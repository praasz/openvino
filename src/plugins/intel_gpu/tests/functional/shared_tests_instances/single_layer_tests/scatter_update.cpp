// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/scatter_update.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ScatterUpdateLayerTest;

const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

const std::vector<ov::element::Type> idxPrecisions = {
        ov::element::i32,
        ov::element::i64,
};

// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape {
    {{10, 16, 12, 15}, {{{2, 2, 2}, {0, 1, 2, 3}}, {{2, 4}, {0, 1, 2, 3}}, {{8}, {0, 1, 2, 3}}}},
    {{10, 9, 10, 9, 10}, {{{8}, {0, 1, 2, 3, 4}}, {{4, 2}, {0, 1, 2, 3, 4}}}},
    {{10, 9, 10, 9, 10, 12}, {{{8}, {0, 1, 2, 3, 4, 5}}}},
    {{10, 16, 12, 15}, {{{2, 4}, {0, 1, 2, 3}}, {{8}, {-1, -2, -3, -4}}}},
    {{10, 9, 10, 9, 10}, {{{8}, {-3, -1, 0, 2, 4}}, {{4, 2}, {-2, 2}}}},
};
//indices should not be random value
const std::vector<std::vector<int64_t>> idxValue = {
        {0, 2, 4, 6, 1, 3, 5, 7}
};

std::vector<ov::test::axisUpdateShapeInShape> combine_shapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::axisUpdateShapeInShape> res_vec;
    for (auto& input_shape : input_shapes) {
        auto src_shape = input_shape.first;
        auto srcRank = src_shape.size();
        for (auto& item : input_shape.second) {
            auto indices_shape = item.first;
            auto indices_rank = indices_shape.size();
            for (auto& axis : item.second) {
                auto axisP = axis < 0 ? axis + srcRank : axis;
                std::vector<size_t> update_shape;
                for (size_t rs = 0; rs < srcRank; rs++) {
                    if (rs != axisP) {
                        update_shape.push_back(src_shape[rs]);
                    } else {
                        for (size_t ri = 0; ri < indices_rank; ri++) {
                            update_shape.push_back(indices_shape[ri]);
                        }
                    }
                }
                std::vector<ov::Shape> in_shapes{src_shape, update_shape};
                res_vec.push_back(
                        ov::test::axisUpdateShapeInShape{
                                ov::test::static_shapes_to_test_representation(in_shapes),
                                ov::Shape{indices_shape},
                                axis});
            }
        }
    }
    return res_vec;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterUpdate,
    ScatterUpdateLayerTest,
    ::testing::Combine(::testing::ValuesIn(combine_shapes(axesShapeInShape)),
                       ::testing::ValuesIn(idxValue),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterUpdateLayerTest::getTestCaseName);

}  // namespace
