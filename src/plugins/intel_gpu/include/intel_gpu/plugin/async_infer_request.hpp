// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/plugin/sync_infer_request.hpp"
#include "openvino/runtime/async_infer_request.hpp"

namespace ov::intel_gpu {

class AsyncInferRequest : public ov::AsyncInferRequest {
public:
    using Parent = ov::AsyncInferRequest;
    AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& infer_request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~AsyncInferRequest() override;

    void start_async() override;

private:
    std::shared_ptr<SyncInferRequest> m_infer_request;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
};

}  // namespace ov::intel_gpu
