// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/async_infer_request.hpp"

#include <atomic>
#include <memory>

#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/plugin_itt.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/variable_state.hpp"

/// @brief Thread-safe global counter for unique inference request IDs that
/// are needed for asynchronous inference.
static std::atomic<uint64_t> g_inference_uid = {1};

namespace {

struct ImmediateStreamsExecutor : public ov::threading::ITaskExecutor {
    explicit ImmediateStreamsExecutor(const std::shared_ptr<ov::threading::IStreamsExecutor>& streamsExecutor)
        : _streamsExecutor{streamsExecutor} {}
    void run(ov::threading::Task task) override {
        if (_streamsExecutor->get_streams_num() > 1) {
            std::vector<ov::threading::Task> tasks{std::move(task)};
            _streamsExecutor->run_and_wait(tasks);
        } else {
            _streamsExecutor->execute(std::move(task));
        }
    }
    std::shared_ptr<ov::threading::IStreamsExecutor> _streamsExecutor;
};

}  // namespace

ov::AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}

ov::AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                                         const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                         const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : m_infer_id(0),
      m_sync_request(request),
      m_request_executor(task_executor),
      m_callback_executor(callback_executor) {
    if (m_request_executor && m_sync_request)
        m_pipeline = {{m_request_executor, [this] {
                           m_sync_request->infer();
                       }}};
    if (m_sync_request)
        m_sync_pipeline = {{std::make_shared<ov::threading::ImmediateExecutor>(), [this] {
                                m_sync_request->infer();
                            }}};
    auto streams_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(m_request_executor);
    if (streams_executor != nullptr) {
        m_sync_pipeline = {{std::make_shared<ImmediateStreamsExecutor>(std::move(streams_executor)), [this] {
                                m_sync_request->infer();
                            }}};
    }
}

void ov::AsyncInferRequest::wait() {
    // Just use the last '_futures' member to wait pipeline completion
    auto future = [this] {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_futures.empty() ? std::shared_future<void>{} : m_futures.back();
    }();
    if (future.valid()) {
        future.get();
    }
}

bool ov::AsyncInferRequest::wait_for(const std::chrono::milliseconds& timeout) {
    OPENVINO_ASSERT(timeout >= std::chrono::milliseconds{0}, "Timeout can't be less than 0 for InferRequest::wait().");

    // Just use the last '_futures' member to wait pipeline completion
    auto future = [this] {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_futures.empty() ? std::shared_future<void>{} : m_futures.back();
    }();

    if (!future.valid()) {
        return false;
    }

    const auto status = future.wait_for(std::chrono::milliseconds{timeout});

    if (std::future_status::ready == status) {
        future.get();
        return true;
    } else {
        return false;
    }
}

void ov::AsyncInferRequest::cancel() {
    std::lock_guard<std::mutex> lock{m_mutex};
    if (m_state == InferState::BUSY) {
        m_state = InferState::CANCELLED;
    }
}

void ov::AsyncInferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    check_state();
    std::lock_guard<std::mutex> lock{m_mutex};
    m_callback = std::move(callback);
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::AsyncInferRequest::query_state() const {
    check_state();
    return m_sync_request->query_state();
}

void ov::AsyncInferRequest::infer_thread_unsafe() {
    run_first_stage(m_sync_pipeline.begin(), m_sync_pipeline.end(), m_sync_callback_executor);
}

void ov::AsyncInferRequest::start_async_thread_unsafe() {
    run_first_stage(m_pipeline.begin(), m_pipeline.end(), m_callback_executor);
}

void ov::AsyncInferRequest::run_first_stage(const Pipeline::iterator itBeginStage,
                                            const Pipeline::iterator itEndStage,
                                            const std::shared_ptr<ov::threading::ITaskExecutor> callbackExecutor) {
    m_infer_id = g_inference_uid++;
    auto& firstStageExecutor = std::get<Stage_e::EXECUTOR>(*itBeginStage);
    OPENVINO_ASSERT(nullptr != firstStageExecutor);
    firstStageExecutor->run(make_next_stage_task(itBeginStage, itEndStage, std::move(callbackExecutor)));
}

ov::threading::Task ov::AsyncInferRequest::make_next_stage_task(
    const Pipeline::iterator itStage,
    const Pipeline::iterator itEndStage,
    const std::shared_ptr<ov::threading::ITaskExecutor> callbackExecutor) {
    return std::bind(
        [this, itStage, itEndStage](std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor) mutable {
            // Propagate the inference ID through all subsequent stages for this instance of the pipeline
            OV_ITT_SCOPED_REGION_BASE(ov::itt::domains::Inference, "Inference::pipeline", "InferenceID", m_infer_id);
            std::exception_ptr currentException = nullptr;
            auto& thisStage = *itStage;
            auto itNextStage = itStage + 1;
            try {
                auto& stageTask = std::get<Stage_e::TASK>(thisStage);
                OPENVINO_ASSERT(nullptr != stageTask);
                stageTask();
                if (itEndStage != itNextStage) {
                    auto& nextStage = *itNextStage;
                    auto& nextStageExecutor = std::get<Stage_e::EXECUTOR>(nextStage);
                    OPENVINO_ASSERT(nullptr != nextStageExecutor);
                    nextStageExecutor->run(make_next_stage_task(itNextStage, itEndStage, std::move(callbackExecutor)));
                }
            } catch (...) {
                currentException = std::current_exception();
            }

            if ((itEndStage == itNextStage) || (nullptr != currentException)) {
                auto lastStageTask = [this, currentException]() mutable {
                    std::promise<void> promise;
                    std::function<void(std::exception_ptr)> callback;
                    {
                        std::lock_guard<std::mutex> lock{m_mutex};
                        m_state = InferState::IDLE;
                        promise = std::move(m_promise);
                        std::swap(callback, m_callback);
                    }
                    if (callback) {
                        try {
                            callback(currentException);
                        } catch (...) {
                            currentException = std::current_exception();
                        }
                        std::lock_guard<std::mutex> lock{m_mutex};
                        if (!m_callback) {
                            std::swap(callback, m_callback);
                        }
                    }
                    if (nullptr == currentException) {
                        promise.set_value();
                    } else {
                        promise.set_exception(currentException);
                    }
                };

                if (nullptr == callbackExecutor) {
                    lastStageTask();
                } else {
                    callbackExecutor->run(std::move(lastStageTask));
                }
            }
        },
        std::move(callbackExecutor));
}

void ov::AsyncInferRequest::start_async() {
    infer_impl([this] {
        start_async_thread_unsafe();
    });
}

void ov::AsyncInferRequest::check_state() const {
    std::lock_guard<std::mutex> lock{m_mutex};
    switch (m_state) {
    case InferState::BUSY:
        ov::Busy::create("Infer Request is busy");
    case InferState::CANCELLED:
        ov::Cancelled::create("Infer Request was canceled");
    default:
        break;
    }
}

void ov::AsyncInferRequest::check_cancelled_state() const {
    std::lock_guard<std::mutex> lock{m_mutex};
    if (m_state == InferState::CANCELLED)
        ov::Cancelled::create("Infer Request was canceled");
}

std::vector<ov::ProfilingInfo> ov::AsyncInferRequest::get_profiling_info() const {
    check_state();
    return m_sync_request->get_profiling_info();
}

ov::SoPtr<ov::ITensor> ov::AsyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    check_state();
    return m_sync_request->get_tensor(port);
}

void ov::AsyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    check_state();
    return m_sync_request->set_tensor(port, tensor);
}

std::vector<ov::SoPtr<ov::ITensor>> ov::AsyncInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    check_state();
    return m_sync_request->get_tensors(port);
}

void ov::AsyncInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                        const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    check_state();
    return m_sync_request->set_tensors(port, tensors);
}

void ov::AsyncInferRequest::stop_and_wait() {
    Futures futures;
    InferState state = InferState::IDLE;
    {
        std::lock_guard<std::mutex> lock{m_mutex};
        state = m_state;
        if (state != InferState::STOP) {
            m_callback = {};
            m_state = InferState::STOP;
            futures = std::move(m_futures);
        }
    }
    if (state != InferState::STOP) {
        for (auto&& future : futures) {
            if (future.valid()) {
                future.wait();
            }
        }
    }
}

void ov::AsyncInferRequest::infer() {
    DisableCallbackGuard disableCallbackGuard{this};
    infer_impl([this] {
        infer_thread_unsafe();
    });
    wait();
}

void ov::AsyncInferRequest::check_tensors() const {
    if (m_sync_request) {
        invoke_check_tensors(*m_sync_request);
    }
}

const std::shared_ptr<const ov::ICompiledModel>& ov::AsyncInferRequest::get_compiled_model() const {
    return m_sync_request->get_compiled_model();
}

const std::vector<ov::Output<const ov::Node>>& ov::AsyncInferRequest::get_inputs() const {
    return m_sync_request->get_inputs();
}
const std::vector<ov::Output<const ov::Node>>& ov::AsyncInferRequest::get_outputs() const {
    return m_sync_request->get_outputs();
}
