// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "executor.hpp"
#include "executor_implementation.hpp"
#include "nodes/executors/graph_emitter.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

/**
 * A stateful (variable) executor
 * Contains two or more executors.
 * Switches between the executors based on provided Memory (more precisely based on in / out shapes)
 */
template <typename Attrs>
class VariableExecutor : public Executor {
public:
    using ExecutorImplementationRef = std::reference_wrapper<const ExecutorImplementation<Attrs>>;

    VariableExecutor(const MemoryArgs& memory,
                     Attrs attrs,
                     ExecutorContext::CPtr context,
                     std::vector<ExecutorImplementationRef> suitableImplementations)
        : m_attrs(std::move(attrs)),
          m_context(std::move(context)),
          m_suitableImplementations(std::move(suitableImplementations)),
          m_executors(m_suitableImplementations.size()) {
        const size_t implId = select(memory, 0);
        m_executors[implId] = create(implId, memory);
        m_implId = implId;
    }

    bool update(const MemoryArgs& memory) override {
        for (auto implId = select(memory, 0); implId < m_suitableImplementations.size();
             implId = select(memory, ++implId)) {
            if (!m_executors[implId]) {
                m_executors[implId] = create(implId, memory);
                if (!m_executors[implId]) {
                    continue;  // skip if creation failed
                }
            }

            if (m_executors[implId]->update(memory)) {
                m_implId = implId;
                return true;
            }
        }

        return false;
    }

    void execute(const MemoryArgs& memory) override {
        m_executors[m_implId]->execute(memory);
    }

    [[nodiscard]] impl_desc_type implType() const override {
        return m_executors[m_implId]->implType();
    }

    void moveMemToNumaNode(int numaID) override {
        m_executors[m_implId]->moveMemToNumaNode(numaID);
    }

private:
    [[nodiscard]] size_t select(const MemoryArgs& memory, const size_t startIdx) const {
        OPENVINO_ASSERT(startIdx < m_suitableImplementations.size(),
                        "Failed to find an implementation since start indx: ",
                        startIdx,
                        " is out of range of the suitable implementations array: ",
                        m_suitableImplementations.size());

        auto startIt = m_suitableImplementations.begin() + startIdx;

        const auto selectedImplementation =
            std::find_if(startIt,
                         m_suitableImplementations.end(),
                         [&memory, this](const ExecutorImplementationRef& implementation) {
                             return implementation.get().acceptsShapes(m_attrs, memory);
                         });

        OPENVINO_ASSERT(selectedImplementation != m_suitableImplementations.end(), "Failed to select an implemetation");

        return std::distance(m_suitableImplementations.begin(), selectedImplementation);
    }

    ExecutorPtr create(const size_t implId, const MemoryArgs& memory) {
        assert(implId < m_executors.size() && implId < m_suitableImplementations.size());

        const auto& impl = m_suitableImplementations[implId].get();
        return impl.create(m_attrs, memory, m_context);
    }

    Attrs m_attrs;
    const ExecutorContext::CPtr m_context;
    std::vector<ExecutorImplementationRef> m_suitableImplementations;
    // executors cache
    std::vector<ExecutorPtr> m_executors;
    size_t m_implId;
};

}  // namespace ov::intel_cpu
