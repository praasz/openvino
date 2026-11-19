// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common_test_utils/test_tools.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

using namespace std;

namespace ov::test {

std::shared_ptr<ov::Model> make_test_graph() {
    auto arg_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_4 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    auto arg_5 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});

    auto t0 = std::make_shared<ov::op::v1::Add>(arg_0, arg_1);
    auto t1 = std::make_shared<ov::op::v0::MatMul>(t0, arg_2);
    auto t2 = std::make_shared<ov::op::v1::Multiply>(t0, arg_3);

    auto t3 = std::make_shared<ov::op::v1::Add>(t1, arg_4);
    auto t4 = std::make_shared<ov::op::v1::Add>(t2, arg_5);

    auto r0 = std::make_shared<ov::op::v1::Add>(t3, t4);

    auto m = std::make_shared<ov::Model>(r0, ov::ParameterVector{arg_0, arg_1, arg_2, arg_3, arg_4, arg_5});

    return m;
}

// This function traverses the vector of ops and verifies that each op's dependencies (its inputs)
// is located earlier in the vector. That is enough to be valid
bool validate_list(const std::vector<std::shared_ptr<ov::Node>>& nodes) {
    bool rc = true;
    for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
        auto node_tmp = *it;
        ov::NodeVector dependencies_tmp;
        for (auto& val : node_tmp->input_values())
            dependencies_tmp.emplace_back(val.get_node_shared_ptr());
        std::vector<ov::Node*> dependencies;

        for (std::shared_ptr<ov::Node> n : dependencies_tmp) {
            dependencies.push_back(n.get());
        }
        auto tmp = it;
        for (tmp++; tmp != nodes.rend(); tmp++) {
            auto dep_tmp = *tmp;
            auto found = find(dependencies.begin(), dependencies.end(), dep_tmp.get());
            if (found != dependencies.end()) {
                dependencies.erase(found);
            }
        }
        if (dependencies.size() > 0) {
            rc = false;
        }
    }
    return rc;
}
class TestMatcherPassTrue : public ov::pass::MatcherPass {
public:
    TestMatcherPassTrue() : MatcherPass() {
        auto any_input = ov::pass::pattern::any_input();
        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher&) {
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(any_input, "TestMatcherPassTrue");
        this->register_matcher(m, callback);
    }
};

class TestMatcherPassFalse : public ov::pass::MatcherPass {
public:
    TestMatcherPassFalse() : MatcherPass() {
        auto any_input = ov::pass::pattern::any_input();
        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher&) {
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(any_input, "TestMatcherPassFalse");
        this->register_matcher(m, callback);
    }
};

class TestModelPassTrue : public pass::ModelPass {
public:
    bool run_on_model(const std::shared_ptr<ov::Model>&) override {
        return true;
    }
};

class TestModelPassFalse : public pass::ModelPass {
public:
    bool run_on_model(const std::shared_ptr<ov::Model>&) override {
        return false;
    }
};

class CountingValidate : public ov::pass::Validate {
public:
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        is_applied = true;
        return Validate::run_on_model(model);
    }

    bool is_applied = false;
};

class ValidateTestManager : public pass::Manager {
protected:
    void push_validate_pass() override {
        push_pass<CountingValidate>();
    }

public:
    int get_num_validate_executed() {
        int total = 0;

        for (const auto& pass : m_pass_list) {
            if (auto counting_validate = ov::as_type_ptr<CountingValidate>(pass)) {
                total += counting_validate->is_applied;
            }
        }

        return total;
    }
};

TEST(pass_manager, Validate_passes_not_applied) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    auto validate = manager.register_pass<CountingValidate>();
    const auto res = manager.run_passes(graph);

    EXPECT_FALSE(res);
    EXPECT_FALSE(validate->is_applied);
}

TEST(pass_manager, Validate_model_pass_applied) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassTrue>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    auto validate = manager.register_pass<CountingValidate>();
    const auto res = manager.run_passes(graph);

    EXPECT_TRUE(res);
    EXPECT_TRUE(validate->is_applied);
}

TEST(pass_manager, Validate_matcher_pass_applied) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    manager.register_pass<TestMatcherPassTrue>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    auto validate = manager.register_pass<CountingValidate>();
    const auto res = manager.run_passes(graph);

    EXPECT_TRUE(res);
    EXPECT_TRUE(validate->is_applied);
}

TEST(pass_manager, Validate_three_validations) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    manager.register_pass<TestMatcherPassTrue>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    auto validate_1 = manager.register_pass<CountingValidate>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    auto validate_2 = manager.register_pass<CountingValidate>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassTrue>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassTrue>();
    manager.register_pass<TestModelPassFalse>();
    auto validate_3 = manager.register_pass<CountingValidate>();
    const auto res = manager.run_passes(graph);

    EXPECT_TRUE(res);
    EXPECT_TRUE(validate_1->is_applied);
    EXPECT_FALSE(validate_2->is_applied);
    EXPECT_TRUE(validate_3->is_applied);
}

TEST(pass_manager, Validate_per_pass) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(true);  // by default true

    auto graph = make_test_graph();

    manager.register_pass<TestMatcherPassTrue>();
    auto validate_1 = manager.register_pass<CountingValidate>();
    auto validate_2 = manager.register_pass<CountingValidate>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassTrue>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassTrue>();
    manager.register_pass<TestModelPassFalse>();
    auto validate_3 = manager.register_pass<CountingValidate>();
    const auto res = manager.run_passes(graph);

    const auto num_true_passes = 3;

    EXPECT_TRUE(res);
    EXPECT_FALSE(validate_1->is_applied);
    EXPECT_FALSE(validate_2->is_applied);
    EXPECT_FALSE(validate_3->is_applied);
    EXPECT_EQ(manager.get_num_validate_executed(), num_true_passes);
}

TEST(pass_manager, Validate_single_validate) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    auto validate = manager.register_pass<CountingValidate>();
    const auto res = manager.run_passes(graph);

    EXPECT_FALSE(res);
    EXPECT_FALSE(validate->is_applied);
}

TEST(pass_manager, Validate_single_validate_per_pass) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(true);  // by default true

    auto graph = make_test_graph();

    auto validate = manager.register_pass<CountingValidate>();
    const auto res = manager.run_passes(graph);

    const auto num_true_passes = 0;

    EXPECT_FALSE(res);
    EXPECT_FALSE(validate->is_applied);
    EXPECT_EQ(manager.get_num_validate_executed(), num_true_passes);
}

TEST(pass_manager, Validate_counter) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(true);  // by default true

    auto graph = make_test_graph();

    manager.register_pass<TestMatcherPassTrue>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassTrue>();
    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassTrue>();
    manager.register_pass<TestModelPassFalse>();
    manager.register_pass<CountingValidate>();
    const auto res = manager.run_passes(graph);

    const auto num_true_passes = 3;

    EXPECT_TRUE(res);
    EXPECT_EQ(manager.get_num_validate_executed(), num_true_passes);
}

TEST(pass_manager, Validate_counter_0) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    manager.register_pass<TestMatcherPassFalse>();
    manager.register_pass<TestModelPassFalse>();

    const auto res = manager.run_passes(graph);

    const auto num_true_passes = 0;

    EXPECT_FALSE(res);
    EXPECT_EQ(manager.get_num_validate_executed(), num_true_passes);
}

TEST(pass_manager, Validate_counter_1) {
    ValidateTestManager manager;
    manager.set_per_pass_validation(false);

    auto graph = make_test_graph();

    manager.register_pass<TestMatcherPassTrue>();
    manager.register_pass<TestModelPassTrue>();

    const auto res = manager.run_passes(graph);

    EXPECT_TRUE(res);
    EXPECT_EQ(manager.get_num_validate_executed(), /*no Validate inserted*/ 0);
}

}  // namespace

TEST(pass_manager, add) {
    pass::Manager pass_manager;

    auto graph = make_test_graph();
    size_t node_count = 0;
    ov::traverse_nodes(graph, [&](shared_ptr<Node> /* node */) {
        node_count++;
    });
    pass_manager.run_passes(graph);
    auto sorted = graph->get_ordered_ops();
    EXPECT_EQ(node_count, sorted.size());
    EXPECT_TRUE(validate_list(sorted));
}


using ::testing::_, ::testing::MockFunction, ::testing::Return;

class PassCreationTestPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PassCreationTestPass");
    explicit PassCreationTestPass(ov::matcher_pass_callback callback = nullptr,
                                  int value = 0,
                                  bool throw_in_ctor = false)
        : value(value) {
        if (throw_in_ctor) {
            throw std::runtime_error("expected exception from pass constructor");
        }
        if (callback) {
            register_matcher(std::make_shared<ov::pass::pattern::Matcher>(
                                 ov::pass::pattern::wrap_type<ov::op::v0::Relu>(),
                                 "PassCreationTestPass"),
                             callback);
        }
    }

    int value;
};

class MockModelPass : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MockModelPass");
    MOCK_METHOD(bool, run_on_model, (const std::shared_ptr<ov::Model>&), (override));
};

// PassBase subobject is not at offset 0, the owner must delete through the right base pointer.
struct LeadingBase {
    virtual ~LeadingBase() = default;
    long payload[4] = {};
};
struct OffsetPass : LeadingBase, PassCreationTestPass {};
struct SharedFromThisPass : PassCreationTestPass, std::enable_shared_from_this<SharedFromThisPass> {};

class NestedRewrite : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("NestedRewrite");
    explicit NestedRewrite(const ov::matcher_pass_callback& callback) {
        add_matcher<PassCreationTestPass>(callback);
        add_matcher<PassCreationTestPass>(callback);
    }
};

std::shared_ptr<ov::Model> make_relu_model() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

template <class Pass>
void expect_non_zero_base_offset(const std::shared_ptr<Pass>& pass) {
    ASSERT_NE(static_cast<void*>(pass.get()), static_cast<void*>(static_cast<ov::pass::PassBase*>(pass.get())));
}

TEST(pass_creation, manager_register_pass_returns_owning_instance) {
    std::weak_ptr<PassCreationTestPass> weak;
    std::shared_ptr<PassCreationTestPass> pass;
    {
        ov::pass::Manager manager;
        pass = manager.register_pass<PassCreationTestPass>(nullptr, 7);
        weak = pass;
        EXPECT_EQ(pass->value, 7);
        EXPECT_GE(pass.use_count(), 2);  // the manager keeps the pass alive too
        EXPECT_EQ(pass->get_pass_config(), manager.get_pass_config());
    }
    EXPECT_EQ(pass.use_count(), 1);
    EXPECT_FALSE(weak.expired());
    pass.reset();
    EXPECT_TRUE(weak.expired());
}

TEST(pass_creation, manager_runs_registered_passes) {
    MockFunction<bool(ov::pass::pattern::Matcher&)> matcher_callback;
    EXPECT_CALL(matcher_callback, Call(_)).WillOnce(Return(false));

    ov::pass::Manager manager;
    manager.register_pass<PassCreationTestPass>(matcher_callback.AsStdFunction());
    auto model_pass = manager.register_pass<MockModelPass>();
    EXPECT_CALL(*model_pass, run_on_model(_)).WillOnce(Return(false));
    manager.run_passes(make_relu_model());
}

TEST(pass_creation, manager_register_disabled_pass) {
    MockFunction<bool(ov::pass::pattern::Matcher&)> matcher_callback;
    EXPECT_CALL(matcher_callback, Call(_)).Times(0);

    ov::pass::Manager manager;
    manager.register_pass<PassCreationTestPass, false>(matcher_callback.AsStdFunction());
    EXPECT_TRUE(manager.get_pass_config()->is_disabled<PassCreationTestPass>());
    manager.run_passes(make_relu_model());
}

TEST(pass_creation, manager_register_pass_with_throwing_constructor) {
    ov::pass::Manager manager;
    EXPECT_THROW(manager.register_pass<PassCreationTestPass>(nullptr, 0, true), std::runtime_error);
    EXPECT_NO_THROW(manager.run_passes(make_relu_model()));
}

TEST(pass_creation, manager_register_pass_with_non_zero_base_offset) {
    std::weak_ptr<OffsetPass> weak;
    {
        ov::pass::Manager manager;
        auto pass = manager.register_pass<OffsetPass>();
        expect_non_zero_base_offset(pass);
        weak = pass;
    }
    EXPECT_TRUE(weak.expired());
}

TEST(pass_creation, manager_register_pass_supports_shared_from_this) {
    ov::pass::Manager manager;
    auto pass = manager.register_pass<SharedFromThisPass>();
    std::shared_ptr<SharedFromThisPass> from_this;
    ASSERT_NO_THROW(from_this = pass->shared_from_this());
    EXPECT_EQ(from_this, pass);
}

TEST(pass_creation, graph_rewrite_add_matcher_returns_owning_instance) {
    MockFunction<bool(ov::pass::pattern::Matcher&)> matcher_callback;
    EXPECT_CALL(matcher_callback, Call(_)).WillOnce(Return(false));

    std::weak_ptr<PassCreationTestPass> weak;
    {
        ov::pass::Manager manager;
        auto rewrite = manager.register_pass<ov::pass::GraphRewrite>();
        auto pass = rewrite->add_matcher<PassCreationTestPass>(matcher_callback.AsStdFunction(), 3);
        weak = pass;
        EXPECT_EQ(pass->value, 3);
        EXPECT_EQ(pass->get_pass_config(), rewrite->get_pass_config());
        manager.run_passes(make_relu_model());
    }
    EXPECT_TRUE(weak.expired());
}

TEST(pass_creation, graph_rewrite_add_disabled_matcher) {
    MockFunction<bool(ov::pass::pattern::Matcher&)> matcher_callback;
    EXPECT_CALL(matcher_callback, Call(_)).Times(0);

    ov::pass::Manager manager;
    auto rewrite = manager.register_pass<ov::pass::GraphRewrite>();
    rewrite->add_matcher<PassCreationTestPass, false>(matcher_callback.AsStdFunction());
    EXPECT_TRUE(rewrite->get_pass_config()->is_disabled<PassCreationTestPass>());
    manager.run_passes(make_relu_model());
}

TEST(pass_creation, graph_rewrite_add_nested_graph_rewrite) {
    MockFunction<bool(ov::pass::pattern::Matcher&)> matcher_callback;
    EXPECT_CALL(matcher_callback, Call(_)).Times(2).WillRepeatedly(Return(false));

    ov::pass::Manager manager;
    auto rewrite = manager.register_pass<ov::pass::GraphRewrite>();
    rewrite->add_matcher<NestedRewrite>(matcher_callback.AsStdFunction());
    manager.run_passes(make_relu_model());
}

TEST(pass_creation, graph_rewrite_add_matcher_with_throwing_constructor) {
    ov::pass::GraphRewrite rewrite;
    EXPECT_THROW(rewrite.add_matcher<PassCreationTestPass>(nullptr, 0, true), std::runtime_error);
}

TEST(pass_creation, graph_rewrite_add_matcher_with_non_zero_base_offset) {
    std::weak_ptr<OffsetPass> weak;
    {
        ov::pass::GraphRewrite rewrite;
        auto pass = rewrite.add_matcher<OffsetPass>();
        expect_non_zero_base_offset(pass);
        weak = pass;
    }
    EXPECT_TRUE(weak.expired());
}

TEST(pass_creation, graph_rewrite_add_matcher_supports_shared_from_this) {
    ov::pass::GraphRewrite rewrite;
    auto pass = rewrite.add_matcher<SharedFromThisPass>();
    std::shared_ptr<SharedFromThisPass> from_this;
    ASSERT_NO_THROW(from_this = pass->shared_from_this());
    EXPECT_EQ(from_this, pass);
}
