// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <snippets/op/memory_access.hpp>
#include <utility>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_infer_instances.hpp"

namespace ov::snippets::op {

/**
 * @interface BroadcastLoad
 * @brief Is generated for broadcasting by least varying dimension for non-blocked cases and the second varying
 * dimension for blocked
 * @ingroup snippets
 */
class BroadcastLoad : public modifier::MemoryAccess, public ov::op::Op {
public:
    OPENVINO_OP("BroadcastLoad", "SnippetsOpset");

    BroadcastLoad(const Output<Node>& x, ov::Dimension bcast_dimension, size_t offset = 0LU);
    BroadcastLoad() = default;

    size_t get_offset() const {
        return get_input_offset(0);
    }

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    const ov::Dimension& get_bcast_dimension() {
        return bcast_dimension;
    }
    void set_bcast_dimension(const ov::Dimension& new_dim) {
        bcast_dimension = new_dim;
    }

    // Note:BroadcastMove and BroadcastLoad are implemented as separate classes,
    // but have identical shapeInfer semantics. In order to avoid code duplication,
    // we created dummy ShapeInfer classes that are essentially instantiations
    // of a common ShapeInfer class template;
    struct ShapeInfer : public BroadcastShapeInfer<BroadcastLoad> {
        explicit ShapeInfer(const std::shared_ptr<Node>& n) : BroadcastShapeInfer<BroadcastLoad>(n) {}
    };

private:
    ov::Dimension bcast_dimension;
};

}  // namespace ov::snippets::op
