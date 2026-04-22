// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime IAsyncInferRequest interface
 * @file openvino/runtime/iasync_infer_request.hpp
 */

#pragma once

#include <chrono>
#include <functional>
#include <memory>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/iinfer_request.hpp"

namespace ov {

/**
 * @brief Interface for asynchronous inference request to be implemented by plugins.
 *
 * Extends IInferRequest with asynchronous lifecycle management:
 * start, wait, cancel, and completion callback registration.
 *
 * Plugins should implement this interface by either:
 *  - Deriving from ov::AsyncInferRequest (the provided concrete base) for pipeline-based requests, or
 *  - Implementing all pure virtual methods directly for fully custom async behaviour.
 *
 * @ingroup ov_dev_api_async_infer_request_api
 */
class OPENVINO_RUNTIME_API IAsyncInferRequest : public IInferRequest {
public:
    virtual ~IAsyncInferRequest() = default;

    /**
     * @brief Start inference of specified input(s) in asynchronous mode
     * @note The method returns immediately. Inference starts also immediately.
     */
    virtual void start_async() = 0;

    /**
     * @brief Waits for the result to become available.
     */
    virtual void wait() = 0;

    /**
     * @brief Waits for the result to become available. Blocks until specified timeout has elapsed or the result
     * becomes available, whichever comes first.
     * @param timeout - maximum duration in milliseconds to block for
     * @return A true if results are ready.
     */
    virtual bool wait_for(const std::chrono::milliseconds& timeout) = 0;

    /**
     * @brief Cancel current inference request execution
     */
    virtual void cancel() = 0;

    /**
     * @brief Set callback function which will be called on success or failure of asynchronous request
     * @param callback - function to be called with the following description:
     */
    virtual void set_callback(std::function<void(std::exception_ptr)> callback) = 0;
};

}  // namespace ov
