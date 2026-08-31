# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_common_translators")
get_filename_component(root_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

set(LIBRARY_SRC
    ${root_dir}/src/op/angle.cpp
    ${root_dir}/src/op/atan2.cpp
    ${root_dir}/src/op/bincount.cpp
    ${root_dir}/src/op/complex.cpp
    ${root_dir}/src/op/equal.cpp
    ${root_dir}/src/op/erfc.cpp
    ${root_dir}/src/op/unsqueeze.cpp
    ${root_dir}/src/sequence_array_lowering.cpp
    ${root_dir}/src/sequence_concat_replacer.cpp
    ${root_dir}/src/sequence_if_replacer.cpp
    ${root_dir}/src/slot_resolver.cpp
    ${root_dir}/src/unconverted_ops_report.cpp
    ${root_dir}/src/utils.cpp
)
set(LIBRARY_HEADERS
    ${root_dir}/include/common_translators.hpp
    ${root_dir}/include/sequence_array_lowering.hpp
    ${root_dir}/include/sequence_concat_replacer.hpp
    ${root_dir}/include/sequence_if_replacer.hpp
    ${root_dir}/include/unconverted_ops_report.hpp
    ${root_dir}/src/slot_resolver.hpp
)

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${LIBRARY_HEADERS})
add_library(openvino::frontend::common_translators ALIAS ${TARGET_NAME})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_STATIC_LIBRARY)
endif()

# remove -Wmissing-declarations warning, because of frontends implementation specific
if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-missing-declarations)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

target_link_libraries(${TARGET_NAME} PUBLIC openvino::core::dev)

target_include_directories(${TARGET_NAME}
    PUBLIC $<BUILD_INTERFACE:${root_dir}/include>
    PRIVATE ${root_dir}/src)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORIES "${root_dir}/include"
                                       "${root_dir}/src"
                    ADDITIONAL_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:${TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                        $<TARGET_PROPERTY:${TARGET_NAME},INCLUDE_DIRECTORIES>)
