// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/vector_buffer.hpp"

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

VectorBuffer::VectorBuffer(const ov::element::Type element_type) : m_element_type(element_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> VectorBuffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(VectorBuffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<VectorBuffer>(m_element_type);
}

void VectorBuffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(VectorBuffer_validate_and_infer_types);
    set_output_type(0, m_element_type, Shape{1LU});
}

bool VectorBuffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(VectorBuffer_visit_attributes);
    visitor.on_attribute("element_type", m_element_type);
    return true;
}

}  // namespace ov::snippets::op
