// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief HSM (Header-Sections-Manifest) blob container format contract.
 * @file openvino/runtime/hsm_format.hpp
 *
 * Fixed @ref ov::runtime::HSMHeader, followed by section payloads, followed by a @ref ov::runtime::ManifestEntry
 * table. Defined as Doxygen-documented code (not a separate spec) so docs and implementation can't drift.
 *
 * @verbatim
   +---------------------------+---------------------------+---------------------------+
   |         HSMHeader         |      Section payloads     |          Manifest         |
   |          32 bytes         |     variable size, 0+     | ManifestEntry[], 32B each |
   |          offset 0         |         offset 32         |  offset = manifest_offset |
   +---------------------------+---------------------------+---------------------------+
   @endverbatim
 *
 * Reader (Story 2), writer (Story 3) and the device tag registry (Story 4) are implemented elsewhere. Multi-blob
 * files and @ref ov::runtime::BlobMagic::SHARED_CONTEXT are forward references only (Story 12/13).
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "openvino/util/container_util.hpp"

namespace ov::runtime {

using HSMMagicType = std::array<uint8_t, 5>;  ///< 5 raw ASCII bytes: container magic at offset 0.
using HSMSizeType = uint64_t;                 ///< Container/section size fields.
using HSMOffsetType = uint64_t;               ///< Byte offsets within the container.

/**
 * @brief HSM major version. Bump only for a change that can't stay backward-compatible; new tags/inline values
 * are always fine within a major version. A reader must accept any minor/patch sharing its major version.
 */
inline constexpr uint8_t HSM_FORMAT_VERSION_MAJOR = 1;
inline constexpr uint8_t HSM_FORMAT_VERSION_MINOR = 0;  //!< Minor version written by this codebase.
inline constexpr uint8_t HSM_FORMAT_VERSION_PATCH = 0;  //!< Patch version written by this codebase.

/** @brief Container magic (5 raw ASCII bytes). Struct, not enum, since #HSMMagicType isn't an integral type. */
struct BlobMagic {
    HSMMagicType value{};  ///< Raw bytes. Zero-initialized: an unset BlobMagic never matches a valid magic.

    /**
     * @brief Byte-wise compare.
     * @note Hand-written (not `value == other.value`) because this codebase targets C++17, where
     * `std::array::operator==` is not `constexpr` (that only happens in C++20) - a manual loop keeps this
     * genuinely usable in constant expressions, not just nominally marked `constexpr`.
     */
    constexpr bool operator==(const BlobMagic& other) const noexcept {
        for (size_t i = 0; i < value.size(); ++i) {
            if (value[i] != other.value[i]) {
                return false;
            }
        }
        return true;
    }
    constexpr bool operator!=(const BlobMagic& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Returns a string_view over the raw bytes
     */
    std::string_view as_string_view() const noexcept {
        return {reinterpret_cast<const char*>(value.data()), value.size()};
    }

    static const BlobMagic SINGLE_BLOB;     //!< "OVBLS": single compiled model.
    static const BlobMagic SHARED_CONTEXT;  //!< "OVWSH": multi-blob files, forward-looking.
};

inline constexpr BlobMagic BlobMagic::SINGLE_BLOB{ov::util::make_array<uint8_t>('O', 'V', 'B', 'L', 'S')};
inline constexpr BlobMagic BlobMagic::SHARED_CONTEXT{ov::util::make_array<uint8_t>('O', 'V', 'W', 'S', 'H')};

/**
 * @brief Fixed header at offset 0 of every HSM container.
 * Compatibility lives in sections/manifest entries, not in this header's layout.
 *
 * @verbatim
   +--------+------+-----------------+
   | Offset | Size | Field           |
   +--------+------+-----------------+
   | 0      | 5    | magic           |
   | 5      | 1    | version_major   |
   | 6      | 1    | version_minor   |
   | 7      | 1    | version_patch   |
   | 8      | 8    | total_size      |
   | 16     | 8    | manifest_offset |
   | 24     | 8    | manifest_size   |
   +--------+------+-----------------+
   @endverbatim
 *
 * @note Native byte order, no endian-swap; safe since every supported target (x86/ARM/RISC-V) is little-endian.
 * @note `#pragma pack(1)`: required so every field sits at its documented offset regardless of compiler/ABI,
 * with no implicit padding.
 */
#pragma pack(push, 1)
struct HSMHeader {
    BlobMagic magic;  ///< Must equal #BlobMagic::SINGLE_BLOB or #BlobMagic::SHARED_CONTEXT.

    uint8_t version_major;  ///< See #HSM_FORMAT_VERSION_MAJOR for the compatibility rule.
    uint8_t version_minor;
    uint8_t version_patch;

    HSMSizeType total_size;         ///< Whole container size, in bytes.
    HSMOffsetType manifest_offset;  ///< Byte offset of the first ManifestEntry.
    HSMSizeType manifest_size;      ///< Manifest size in bytes; entry count = manifest_size / sizeof(ManifestEntry).
};
#pragma pack(pop)
static_assert(sizeof(HSMHeader) == 32,
              "HSMHeader layout changed - bump version_major/document the change before touching this struct, "
              "readers and writers (Story 1/2/3) must be updated together.");

}  // namespace ov::runtime
