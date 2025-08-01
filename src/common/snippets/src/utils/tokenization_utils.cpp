// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/utils/tokenization_utils.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/remarks.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::utils {

using namespace ov::snippets::op;
using namespace ov::snippets::pass;

namespace {
auto has_result_child(const std::shared_ptr<const Node>& node) -> bool {
    const auto& users = node->get_users();
    return std::any_of(users.begin(), users.end(), [](const std::shared_ptr<Node>& child) {
        return ov::is_type<ov::op::v0::Result>(child);
    });
}

auto get_num_result_children(const std::shared_ptr<const Node>& node) -> size_t {
    size_t result = 0;
    for (const auto& child : node->get_users()) {
        if (ov::is_type<ov::opset1::Result>(child)) {
            result++;
        }
    }
    return result;
}

auto outputs_are_not_broadcastable(const std::shared_ptr<const Node>& node) -> bool {
    const auto& outputs = node->outputs();
    if (outputs.size() <= 1) {
        return false;
    }
    ov::PartialShape ref_shape = outputs.front().get_partial_shape();
    bool success = true;
    for (size_t i = 1; i < outputs.size() && success; i++) {
        success &= ov::PartialShape::broadcast_merge_into(ref_shape,
                                                          outputs[i].get_partial_shape(),
                                                          ov::op::AutoBroadcastType::NUMPY);
    }
    return !success;
}
}  // namespace

bool tokenize_node(const std::shared_ptr<ov::Node>& node, const SnippetsTokenization::Config& config) {
    const auto getFusedNames = [](const std::shared_ptr<Node>& n) -> std::string {
        auto rt_info = n->get_rt_info();
        auto it = rt_info.find("originalLayersNames");
        if (it != rt_info.end()) {
            return it->second.as<std::string>() + ",";
        }
        return "";
    };

    auto create_single_node_subgraph = [&](const std::shared_ptr<Node>& node) {
        auto subgraph = op::Subgraph::wrap_node_as_subgraph(node);
        subgraph->get_rt_info()["originalLayersNames"] = getFusedNames(node) + node->get_friendly_name();
        ov::replace_node(node, subgraph);
        op::update_out_tensor_name(subgraph);
    };

    auto abort = [&](const std::string& message) {
        remark(3) << message << '\n';
        create_single_node_subgraph(node);
        return true;
    };
    // inputs that are already subgraphs
    std::unordered_set<std::shared_ptr<Node>> input_subgraphs;
    // clone bodies because we need a rollback if loop is found
    std::map<std::shared_ptr<Node>, std::shared_ptr<ov::Model>> clones;

    ParameterVector body_parameters;
    // inputs to merged subgraph
    OutputVector external_inputs;
    // inputs to the node before merge to subgraph
    OutputVector internal_inputs;
    // nodes whose rt_info should be copied into result subgraph
    NodeVector replaced_nodes{node};

    auto input_values = node->input_values();
    /*
     * Called with subgraph->input_value(i) arg and used to
     * Check that the attached node input subgraph has the same input as the node itself.
     * If true, then ternary merge is initiated.
     *        input
     *        /   \
     *  subgraph--node
     */
    auto is_recurrent = [&input_values](const ov::Output<ov::Node>& to_find) -> bool {
        return std::any_of(input_values.begin(), input_values.end(), [&](const ov::Output<ov::Node>& in) {
            return in == to_find;
        });
    };
    /*
     * Checks if the passed node introduces loop dependency for given topological bounds (pair of maxParentOrder,
     * minChildOrder). The bounds are presumed to be without dependency. The bounds are updated if no dependency is
     * introduced by the node.
     */
    const auto cyclicDependencyIsIntoduced = [&node](const std::shared_ptr<Node>& nodeToExamine,
                                                     std::pair<int64_t, int64_t>& currentBounds) -> bool {
        assert(currentBounds.first < currentBounds.second && "Invalid currentBounds passed");
        const auto& parentNodes = ov::as_node_vector(nodeToExamine->input_values());
        const int64_t maxParentOrder =
            std::accumulate(parentNodes.begin(),
                            parentNodes.end(),
                            currentBounds.first,
                            [](int64_t maxOrder, const std::shared_ptr<Node>& n) {
                                if (ov::is_type_any_of<ov::op::v0::Constant, ov::op::v0::Parameter>(n)) {
                                    return maxOrder;
                                }
                                return std::max(maxOrder, GetTopologicalOrder(n));
                            });
        const auto& childNodes = nodeToExamine->get_users();
        // Skip the node being attached, since it will be a part of subgraph and can't introduce loop dependency
        const int64_t minChildOrder = std::accumulate(childNodes.begin(),
                                                      childNodes.end(),
                                                      currentBounds.second,
                                                      [&node](int64_t minOrder, const std::shared_ptr<Node>& n) {
                                                          if (ov::is_type<ov::op::v0::Result>(n) || n == node) {
                                                              return minOrder;
                                                          }
                                                          return std::min(minOrder, GetTopologicalOrder(n));
                                                      });
        if (maxParentOrder < minChildOrder) {
            currentBounds = std::pair<int64_t, int64_t>(maxParentOrder, minChildOrder);
            return false;
        }
        return true;
    };

    for (const auto& input_node : ov::as_node_vector(input_values)) {
        if (auto subgraph = ov::as_type_ptr<op::Subgraph>(input_node)) {
            if ((clones.count(input_node) == 0U) &&
                GetSnippetsSubgraphType(subgraph) != SnippetsSubgraphType::Completed) {
                auto f = subgraph->body().clone();
                f->set_friendly_name(subgraph->body_ptr()->get_friendly_name());
                clones[input_node] = f;
            }
        }
    }
    //  If there are no input subgraphs no need to go further, just create a new one.
    if (clones.empty()) {
        create_single_node_subgraph(node);
        remark(1) << "Starting subgraph at: " << node->get_friendly_name() << " with " << node->inputs().size()
                  << " inputs and " << node->outputs().size() << " outputs" << '\n';
        return true;
    }
    std::string subgraph_name = node->get_friendly_name();
    std::string fusedNames{};
    size_t num_result_children = 0;
    std::pair<int64_t, int64_t> currentTopoBounds{-1, LONG_MAX};
    cyclicDependencyIsIntoduced(node, currentTopoBounds);
    assert(!cyclicDependencyIsIntoduced(node, currentTopoBounds) &&
           "Cyclic dependency is introduced by the node itself");
    for (const auto& input_value : input_values) {
        auto input_node = input_value.get_node_shared_ptr();
        if (ov::is_type<op::Subgraph>(input_node) && !cyclicDependencyIsIntoduced(input_node, currentTopoBounds)) {
            auto subgraph = std::static_pointer_cast<op::Subgraph>(input_node);
            if (!input_subgraphs.count(input_node)) {
                input_subgraphs.insert(input_node);

                fusedNames += getFusedNames(subgraph);
                replaced_nodes.push_back(subgraph);

                if (has_result_child(subgraph)) {
                    // we set input subgraph name to the current subgraph
                    // in order to save node friendly name before result
                    subgraph_name = subgraph->get_friendly_name();
                    num_result_children += 1;
                }
                auto f = clones[input_node];
                const auto& input_body_parameters = f->get_parameters();
                // Todo:
                //  Some of the input subgraphs might have common parents, so some of the input_parameters might already
                //  be in external_inputs and hence in body_parameters. Here we handle this case and remove repeated
                //  body_parameters. Would it be better to incorporate all inputs first and then remove repeated params.
                for (size_t i = 0; i < input_body_parameters.size(); ++i) {
                    auto found = std::find(external_inputs.begin(), external_inputs.end(), subgraph->input_value(i));
                    if (found != external_inputs.end()) {
                        // Todo: here we rely on friendly_name uniqueness. Propose a different algorithm.
                        size_t current_input_index = body_parameters.size();
                        for (size_t p_ind = 0; p_ind < body_parameters.size(); p_ind++) {
                            const auto& p = body_parameters[p_ind];
                            // unite two body parameters from two input subgraphs only if:
                            // 1. two input subgraphs are connected to the same parent node/subgraph,
                            // 2. and connected to the same output port of this parent node/subgraph.
                            if (p->get_friendly_name() == found->get_node_shared_ptr()->get_friendly_name() &&
                                external_inputs[p_ind] == *found) {
                                current_input_index = p_ind;
                                break;
                            }
                        }

                        if (current_input_index < body_parameters.size()) {
                            remark(13) << "replacing " << *found << " " << current_input_index << " with "
                                       << body_parameters[current_input_index] << '\n';
                            f->replace_parameter(i, body_parameters[current_input_index]);
                        } else {
                            external_inputs.push_back(subgraph->input_value(i));
                            body_parameters.push_back(input_body_parameters[i]);
                        }
                    } else if (is_recurrent(subgraph->input_value(i))) {
                        remark(13) << "ternary merge is conducted " << subgraph->input_value(i).get_node_shared_ptr()
                                   << '\n';

                        auto internal = input_body_parameters[i];
                        auto internal_consumers = internal->outputs();
                        if (auto to_replace_with =
                                ov::as_type_ptr<op::Subgraph>(subgraph->get_input_node_shared_ptr(i))) {
                            // todo: In principle, we can still attach the node to the subgraph if cyclic dependency is
                            // introduced during ternary merge.
                            //  Need to support.
                            if (cyclicDependencyIsIntoduced(to_replace_with, currentTopoBounds)) {
                                return abort(
                                    "Attempt to perform recurrent merge for cyclic-dependent subgraphs. Aborting.");
                            }
                            for (const auto& output : internal_consumers) {
                                for (auto consumer : output.get_target_inputs()) {
                                    auto other_body = clones[subgraph->get_input_node_shared_ptr(i)];
                                    auto other_body_result =
                                        other_body->get_results()[consumer.get_source_output().get_index()];
                                    auto result_producer = other_body_result->input(0).get_source_output();

                                    consumer.replace_source_output(result_producer.get_node_shared_ptr());
                                }
                            }
                        } else {
                            external_inputs.push_back(subgraph->input_value(i));
                            body_parameters.push_back(input_body_parameters[i]);
                        }
                    } else {
                        external_inputs.push_back(subgraph->input_value(i));
                        body_parameters.push_back(input_body_parameters[i]);
                    }
                }
            }

            // this is there stitching happens, get result of a copy of a body of currently processed input and put it
            // to the new inputs internal output index == external output index
            const auto& input_body = clones[input_node];
            const size_t source_output_index = input_value.get_index();
            const auto& source_result = input_body->get_results()[source_output_index];
            internal_inputs.push_back(source_result->input_value(0));
        } else {
            // We need some non-scalar constants inside Subgraph in the following cases:
            // [*] We have to save explicitly FQ Constants to call ConstantFolding after Tokenization.
            //     After ConstantFolding we will move remaining non-scalar Constants from body using
            //     ConvertConstantsToParameters pass
            // [*] We support Transpose with second Constant input (represents order). This Constant will not be
            // scheduled
            //     and will only be used to decompose Transpose into a proper Load, Store and Loop combination.
            if (ov::is_type<ov::opset1::Constant>(input_node) &&
                (ov::shape_size(input_value.get_shape()) == 1 || ov::is_type<ov::op::v0::FakeQuantize>(node) ||
                 op::Subgraph::constant_input_should_be_inside_body(node))) {
                internal_inputs.push_back(input_node->output(0));
            } else {
                external_inputs.push_back(input_value);
                auto new_parameter = std::make_shared<ov::op::v0::Parameter>(input_value.get_element_type(),
                                                                             input_value.get_partial_shape());
                new_parameter->set_friendly_name(input_node->get_friendly_name());
                body_parameters.push_back(new_parameter);
                internal_inputs.push_back(new_parameter->output(0));
            }
        }
    }
    fusedNames += node->get_friendly_name();
    num_result_children += get_num_result_children(node);
    if (num_result_children > 1) {
        return abort("New subgraph is created since too many Result children are detected");
    }

    auto body_node = node->copy_with_new_inputs(internal_inputs);
    body_node->set_friendly_name(node->get_friendly_name());

    remark(1) << "Original node outputs = " << node->get_output_size()
              << " body node outputs = " << body_node->get_output_size() << '\n';

    if (node->get_output_size() != body_node->get_output_size()) {
        OPENVINO_THROW("original node outputs size and extracted node outputs size doesn't much");
    }

    // After some transformations, a different number of Constants for some operations may be created
    // than the actual number of Constants during tokenization.
    // To avoid unsupported number of non-scalar Constants in the future (plugin specific limitation)
    // we should calculate potentional number of non-scalar Constants that will be moved up from body.
    size_t hidden_data_count = 0;
    if (const auto fq_node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node)) {
        hidden_data_count += ov::snippets::utils::get_non_scalar_constant_count_for_fq(fq_node);
    }

    ResultVector body_results;
    std::vector<std::set<Input<Node>>> subgraph_result_inputs;

    ov::NodeVector ops_for_buffer_count;
    for (const auto& subgraph : input_subgraphs) {
        // we should summurize additional needed data count (non-scalar Constants and Buffers) from all input subgraphs
        // because we will collapse them with our node and we should get total count
        const auto subgraph_ptr = ov::as_type_ptr<ov::snippets::op::Subgraph>(subgraph);
        hidden_data_count += subgraph_ptr->get_virtual_port_count();
        // Buffers can be existed only in Subgraphs with domain sensetive ops which
        // requires intermediate memory for data repacking
        // To avoid load time regressions, we verify only these Subgraph with domain sensetive ops
        if (subgraph_ptr->has_domain_sensitive_ops()) {
            const auto ops = subgraph_ptr->body_ptr()->get_ordered_ops();
            ops_for_buffer_count.insert(ops_for_buffer_count.end(), ops.begin(), ops.end());
        }

        for (const auto& output : subgraph->outputs()) {
            bool first_side_consumer = true;

            for (auto target_input : output.get_target_inputs()) {
                auto target_node = target_input.get_node()->shared_from_this();

                if (input_subgraphs.count(target_node)) {
                    remark(13) << "ternary merge is conducted " << subgraph << " -> " << target_node << '\n';
                }

                if (!input_subgraphs.count(target_node) && target_node != node) {
                    if (first_side_consumer) {
                        auto& input_subgraph_body = clones[subgraph];
                        body_results.push_back(std::make_shared<ov::op::v0::Result>(
                            input_subgraph_body->get_results()[output.get_index()]->input_value(0)));
                        subgraph_result_inputs.emplace_back();

                        first_side_consumer = false;
                    }

                    if (!!subgraph_result_inputs.back().count(target_input)) {
                        OPENVINO_THROW("target input added twice!!!");
                    }
                    // save target input port outside the body
                    subgraph_result_inputs.back().insert(target_input);
                }
            }
        }
    }

    if (op::Subgraph::is_domain_sensitive_op(node)) {
        ops_for_buffer_count.push_back(node);
    }

    for (const auto& output : node->outputs()) {
        body_results.push_back(std::make_shared<ov::op::v0::Result>(body_node->output(output.get_index())));
        subgraph_result_inputs.push_back(output.get_target_inputs());
    }

    if (body_results.size() != subgraph_result_inputs.size()) {
        OPENVINO_THROW("body results and node results size mismatch during subgraph collaps");
    }

    // The each data node (Parameter (and non-Scalar Constants), Result, Buffers with the same ID) requires the own
    // unique GPR. At the moment, CPU Plugin has limitation for GPR registers: there are 12 available GPRs, and one of
    // them must be reserved for runtime parameters, so only 11 can be used during kernel execution. This limitation
    // will be resolved once generator supports gprs spills [75622].
    // TODO [75567]: move this plugin-specific constraint to the plugin callback
    const auto unique_buffer_count = op::Subgraph::get_estimated_buffer_count(ops_for_buffer_count);
    const size_t max_data_ptr_count = config.get_data_ptr_gpr_count();
    if (body_parameters.size() + body_results.size() + hidden_data_count + unique_buffer_count > max_data_ptr_count) {
        const std::string message_reset =
            "new subgraph is created. Impossible to schedule subgraph with " + std::to_string(body_parameters.size()) +
            " inputs, " + std::to_string(body_results.size()) + " outputs and " + std::to_string(hidden_data_count) +
            " non-scalar constants and " + std::to_string(unique_buffer_count) + "buffers.";
        return abort(message_reset);
    }

    auto body = op::create_body(node->get_friendly_name(), body_results, body_parameters);
    for (size_t i = 0; i < body->get_parameters().size(); i++) {
        body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
    }
    auto subgraph = op::build_subgraph(node, external_inputs, body, subgraph_name);
    copy_runtime_info(replaced_nodes, subgraph);
    const auto& act_body = subgraph->body();
    for (size_t i = 0; i < act_body.get_parameters().size(); i++) {
        act_body.get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
    }

    if (subgraph->get_output_size() != subgraph_result_inputs.size()) {
        OPENVINO_THROW("newly create subgraph doesn't much number of results");
    }

    if (outputs_are_not_broadcastable(subgraph)) {
        return abort("New subgraph is created due to outputs of a subgraph not broadcastable.");
    }

    for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
        for (auto target_input : subgraph_result_inputs[i]) {
            target_input.replace_source_output(subgraph->output(i));
        }
    }
    op::update_out_tensor_name(subgraph);

    subgraph->validate_and_infer_types();

    const auto& act_body1 = subgraph->body();
    for (size_t i = 0; i < act_body1.get_parameters().size(); i++) {
        act_body1.get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
    }
    subgraph->get_rt_info()["originalLayersNames"] = fusedNames;
    subgraph->set_virtual_port_count(hidden_data_count);

    remark(1) << "Replacement (merge) done for: " << subgraph->get_friendly_name() << " with "
              << subgraph->inputs().size() << " inputs and " << subgraph->outputs().size() << " outputs and "
              << subgraph->body_ptr()->get_ops().size() << " ops total\n";

    return true;
}

std::shared_ptr<ov::snippets::op::Subgraph> tokenize_ordered_nodes(const ov::NodeVector& ordered_ops) {
    OPENVINO_ASSERT(!ordered_ops.empty(), "Nothing to be tokenized!");

    ov::OutputVector body_inputs, subgraph_inputs;
    ov::ParameterVector body_parameters;
    ov::ResultVector body_results;
    std::vector<std::set<Input<Node>>> subgraph_result_inputs;

    auto create_body_inputs = [&](const std::shared_ptr<ov::Node>& node) -> void {
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            const auto input = node->input(i);
            const auto parent = input.get_source_output().get_node_shared_ptr();
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(parent);
            if (constant && (ov::shape_size(input.get_shape()) == 1 || ov::is_type<ov::op::v0::FakeQuantize>(node) ||
                             op::Subgraph::constant_input_should_be_inside_body(node))) {
                // If Constant has one consumer - target node, we add Constant to body_inputs
                // If Constant has several consumers, we should check that all these consumers are inside Subgraph body
                // and if all of them are inside body, we can explicitly add Constant to the body_inputs, otherwise we
                // should make a copy and add copy of Constant to body_inputs For example, this case is especially valid
                // for Transposes nodes
                //     (several Transposes have the same order so there can be the common Constant with this order)
                if (constant->get_output_target_inputs(0).size() == 1) {
                    body_inputs.push_back(input.get_source_output());
                } else {
                    const auto constant_consumers = constant->get_output_target_inputs(0);
                    bool all_consumers_are_inside =
                        std::all_of(constant_consumers.begin(),
                                    constant_consumers.end(),
                                    [&ordered_ops](const ov::Input<ov::Node>& input) {
                                        return std::find(ordered_ops.begin(),
                                                         ordered_ops.end(),
                                                         input.get_node()->shared_from_this()) != ordered_ops.end();
                                    });
                    if (all_consumers_are_inside) {
                        body_inputs.push_back(input.get_source_output());
                    } else {
                        const auto constant_copy = constant->clone_with_new_inputs({});
                        node->set_argument(input.get_index(), constant_copy);
                        body_inputs.emplace_back(constant_copy);
                    }
                }
            } else if (std::find(ordered_ops.begin(), ordered_ops.end(), parent) == ordered_ops.end()) {
                auto parameter =
                    std::make_shared<ov::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
                body_parameters.push_back(parameter);
                body_parameters.back()->set_friendly_name(input.get_node()->get_friendly_name());
                body_inputs.push_back(parameter->output(0));

                subgraph_inputs.push_back(input.get_source_output());

                node->input(i).replace_source_output(parameter);
            }
        }
    };

    size_t hidden_data_count = 0;
    std::string fused_names;
    auto last_node = ordered_ops.back();

    for (const auto& op : ordered_ops) {
        create_body_inputs(op);
        op->clear_control_dependencies();
        fused_names += op->get_friendly_name() + ",";

        if (const auto fq_node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(op)) {
            hidden_data_count += ov::snippets::utils::get_non_scalar_constant_count_for_fq(fq_node);
        }
    }

    for (const auto& output : last_node->outputs()) {
        subgraph_result_inputs.push_back(output.get_target_inputs());
    }
    for (const auto& output : last_node->outputs()) {
        body_results.push_back(std::make_shared<ov::opset1::Result>(last_node->output(output.get_index())));
    }

    if (body_results.size() != subgraph_result_inputs.size()) {
        OPENVINO_THROW("body results and node results size mismatch during subgraph collapse");
    }

    auto body = op::create_body(last_node->get_friendly_name(), body_results, body_parameters);
    auto subgraph = std::make_shared<op::Subgraph>(subgraph_inputs, body);
    // Copy runtime info from last node to subgraph - to copy topological order
    copy_runtime_info(last_node, subgraph);
    subgraph->set_friendly_name(last_node->get_friendly_name());

    for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
        for (const auto& target_input : subgraph_result_inputs[i]) {
            target_input.replace_source_output(subgraph->output(i));
        }
    }
    op::update_out_tensor_name(subgraph);

    subgraph->validate_and_infer_types();

    auto act_body = subgraph->body_ptr();
    for (size_t i = 0; i < act_body->get_parameters().size(); i++) {
        act_body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
    }
    subgraph->get_rt_info()["originalLayersNames"] = fused_names;
    subgraph->set_virtual_port_count(hidden_data_count);

    return subgraph;
}

}  // namespace ov::snippets::utils
