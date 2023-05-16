// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, TileTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto param1 = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{3}, std::vector<int>{3, 4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(tile.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({18, 32, 10}));
    // Test Wrong Static Shape
    std::vector<StaticShape> wrong_static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{}},
                             wrong_static_output_shapes = {StaticShape{}};

    ASSERT_THROW(shape_inference(tile.get(), wrong_static_input_shapes, wrong_static_output_shapes), ov::AssertFailure);
}

TEST(StaticShapeInferenceTest, TileFewRepeatsTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto param1 = ov::op::v0::Constant::create(element::i64, Shape{2}, {4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{6, 8, 10}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(tile.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({6, 32, 10}));
}

TEST(StaticShapeInferenceTest, TileSmallDataRankTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto param1 = ov::op::v0::Constant::create(element::i64, Shape{3}, {3, 4, 1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{8, 10}, StaticShape{3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(tile.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 32, 10}));
}

TEST(StaticShapeInferenceTest, TileSmallDataRankTestRepeatsInConstMap) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {
        {1, std::make_shared<HostTensor>(element::i32, Shape{3}, repeats)}};

    // Test Static Shape
    ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}}, output_shapes = {StaticShape{}};
    shape_inference(tile.get(), input_shapes, output_shapes, constant_data);

    ASSERT_EQ(output_shapes.front(), StaticShape({3, 32, 10}));
}

constexpr auto iterations = static_cast<size_t>(100000);

// New implementation

/**
 * @brief Test data accessor performance when const data are stored as ov::TensorVector
 *
 * Data accessor created before shape inference loop (should be fastest)
 */
TEST(StaticShapeInferenceTest, tile_infer_perf_accessor_ov_tensor_vector) {
    using namespace std::chrono;

    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};
    const auto constant_data = ov::TensorVector{{}, {element::i32, Shape{3}, repeats}};
    const auto accessor = make_tensor_accessor(constant_data);

    const ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto op_sh_infer = make_shape_inference<IStaticShapeInfer>(tile);

    const auto start = steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        op_sh_infer->infer(input_shapes, accessor);
    }
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start);

    std::cout << "[NEW] Test ov::Tensor accessor (created before loop), duration: " << duration.count() << " [us]"
              << std::endl;
}

/**
 * @brief Test data accessor performance when const data are stored as ov::TensorVector
 *
 * Data accessor created for each shape inference in loop.
 */
TEST(StaticShapeInferenceTest, tile_infer_perf_accessor_per_iter_ov_tensor_vector) {
    using namespace std::chrono;

    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};
    const auto constant_data = ov::TensorVector{{}, {element::i32, Shape{3}, repeats}};

    const ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto op_sh_infer = make_shape_inference<IStaticShapeInfer>(tile);

    const auto start = steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        op_sh_infer->infer(input_shapes, make_tensor_accessor(constant_data));
    }
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start);

    std::cout << "[NEW] Test ov::Tensor accessor (created before loop), duration: " << duration.count() << " [us]"
              << std::endl;
}

/**
 * @brief Test data accessor performance when const data are stored HostTensor map, accessor created before loop.
 */
TEST(StaticShapeInferenceTest, tile_infer_perf_accessor_ht_map) {
    using namespace std::chrono;

    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};
    const std::map<size_t, HostTensorPtr>& constant_data = {
        {1, std::make_shared<HostTensor>(element::i32, Shape{3}, repeats)}};

    const ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto op_sh_infer = make_shape_inference<IStaticShapeInfer>(tile);

    const auto accessor = make_tensor_accessor(constant_data);

    const auto start = steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        op_sh_infer->infer(input_shapes, accessor);
    }
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start);

    std::cout << "[NEW] Test HostTensor map accessor (created before loop), duration: " << duration.count() << " [us]"
              << std::endl;
}

/**
 * @brief Test data accessor performance when const data are stored HostTensor map, accessor created inside loop.
 */
TEST(StaticShapeInferenceTest, tile_infer_perf_accessor_per_iter_ht_map) {
    using namespace std::chrono;

    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};
    const std::map<size_t, HostTensorPtr>& constant_data = {
        {1, std::make_shared<HostTensor>(element::i32, Shape{3}, repeats)}};

    const ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto op_sh_infer = make_shape_inference<IStaticShapeInfer>(tile);

    const auto start = steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        op_sh_infer->infer(input_shapes, make_tensor_accessor(constant_data));
    }
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start);

    std::cout << "[NEW] Test HostTensor map accessor (created per iteration), duration: " << duration.count() << " [us]"
              << std::endl;
}

/**
 * @brief Test data accessor performance when const data are stored HostTensor vector, accessor created before loop.
 */
TEST(StaticShapeInferenceTest, tile_infer_perf_accessor_ht_vector) {
    using namespace std::chrono;

    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};
    const auto constant_data = HostTensorVector{{}, std::make_shared<HostTensor>(element::i32, Shape{3}, repeats)};

    const ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto op_sh_infer = make_shape_inference<IStaticShapeInfer>(tile);

    const auto start = steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        op_sh_infer->infer(input_shapes, make_tensor_accessor(constant_data));
    }
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start);

    std::cout << "[NEW] Test HostTensor vector accessor (created before loop), duration: " << duration.count()
              << " [us]" << std::endl;
}

template <class T>
class TestAccessor : public ITensorAccessor {
public:
    TestAccessor(std::vector<std::vector<T>>& tensors) : m_tensors{tensors} {}

    Tensor operator()(size_t port) const override {
        const auto size = m_tensors.size();
        if (port < size) {
            return {element::from<T>(), Shape{size}, m_tensors[port].data(), stride};
        } else {
            return make_tensor_accessor()(port);
        }
    }

private:
    std::vector<std::vector<T>>& m_tensors;
    static const ov::Strides stride;
};

template <class T>
const ov::Strides TestAccessor<T>::stride{};

/**
 * @brief Test data accessor performance when for custom accessor data stored in vector, accessor created before loop.
 */
TEST(StaticShapeInferenceTest, tile_infer_perf_accessor_stl_vector) {
    using namespace std::chrono;

    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    auto constant_data = std::vector<std::vector<int32_t>>{{}, {3, 4, 1}};
    const auto accessor = TestAccessor<int32_t>(constant_data);

    const ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto op_sh_infer = make_shape_inference<IStaticShapeInfer>(tile);

    const auto start = steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        op_sh_infer->infer(input_shapes, accessor);
    }
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start);

    std::cout << "[NEW] Test stl vector accessor (created before loop), duration: " << duration.count() << " [us]"
              << std::endl;
}

/**
 * @brief Test data accessor performance when used old version of infer (using HostTensor map) The accessor is created
 * inside shape infer.
 */
TEST(StaticShapeInferenceTest, tile_infer_perf_backward_compatible_iface) {
    using namespace std::chrono;

    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};
    const std::map<size_t, HostTensorPtr>& constant_data = {
        {1, std::make_shared<HostTensor>(element::i32, Shape{3}, repeats)}};

    const ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto op_sh_infer = make_shape_inference<IStaticShapeInfer>(tile);

    const auto start = steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        op_sh_infer->infer(input_shapes, constant_data);
    }
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start);

    std::cout << "[NEW] Test backward compatible (make accessor per iter), duration: " << duration.count() << " [us]"
              << std::endl;
}

/**
 * @brief Test OLD version of shape inference no data accsor created, HostTensor map created before loop.
 */
TEST(StaticShapeInferenceTest, tile_infer_perf_old_map_creation_done_once) {
    using namespace std::chrono;

    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};
    const std::map<size_t, HostTensorPtr>& constant_data = {
        {1, std::make_shared<HostTensor>(element::i32, Shape{3}, repeats)}};

    const ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto op_sh_infer = make_shape_inference<IShapeInferCommon>(tile);

    const auto start = steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        op_sh_infer->infer(input_shapes, constant_data);
    }
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start);

    std::cout << "[OLD] Test HT map created before loop, duration: " << duration.count() << " [us]" << std::endl;
}

/**
 * @brief Test OLD version of shape inference, HostTensor map created inside loop
 * (most similar what is done in NgraphShapeInfer::infer).
 */
TEST(StaticShapeInferenceTest, tile_infer_perf_old_sim_map_creation_per_infer) {
    using namespace std::chrono;

    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    auto tile = std::make_shared<op::v0::Tile>(param0, param1);

    int32_t repeats[] = {3, 4, 1};

    const ShapeVector input_shapes = {StaticShape{8, 10}, StaticShape{3}};
    const auto op_sh_infer = make_shape_inference<IShapeInferCommon>(tile);

    const auto start = steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        const std::map<size_t, HostTensorPtr>& constant_data = {
            {1, std::make_shared<HostTensor>(element::i32, Shape{3}, repeats)}};
        op_sh_infer->infer(input_shapes, constant_data);
    }
    const auto duration = duration_cast<microseconds>(steady_clock::now() - start);

    std::cout << "[OLD] Test HT map created in loop, duration: " << duration.count() << " [us]" << std::endl;
}
