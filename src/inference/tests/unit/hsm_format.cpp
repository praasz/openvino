// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_format.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace ov::test {
namespace {

// Fixed size of a single manifest entry (see @ref ov::runtime::HSMHeader::manifest_size). ManifestEntry itself
// isn't defined in hsm_format.hpp right now, so tests use this plain byte count instead of sizeof(...).
constexpr size_t k_manifest_entry_size = 32;

// Builds the raw bytes of a structurally-valid HSM container: HSMHeader, then `section_payload` (if any),
// then `manifest` (raw, pre-built manifest table bytes). Only lays out the container's top-level regions
// (header / section payload / manifest); doesn't know or care what any given section payload or manifest
// entry means - callers pass already-serialized bytes, not typed ManifestEntry objects.
std::vector<uint8_t> make_container(const runtime::BlobMagic& magic,
                                    const std::vector<uint8_t>& section_payload,
                                    const std::vector<uint8_t>& manifest) {
    runtime::HSMHeader header{};
    header.magic = magic;
    header.version_major = runtime::HSM_FORMAT_VERSION_MAJOR;
    header.version_minor = runtime::HSM_FORMAT_VERSION_MINOR;
    header.version_patch = runtime::HSM_FORMAT_VERSION_PATCH;
    header.manifest_offset = sizeof(runtime::HSMHeader) + section_payload.size();
    header.manifest_size = manifest.size();
    header.total_size = header.manifest_offset + header.manifest_size;

    std::vector<uint8_t> buffer(header.total_size);
    std::memcpy(buffer.data(), &header, sizeof(header));
    if (!section_payload.empty()) {
        std::memcpy(buffer.data() + sizeof(header), section_payload.data(), section_payload.size());
    }
    if (!manifest.empty()) {
        std::memcpy(buffer.data() + header.manifest_offset, manifest.data(), manifest.size());
    }
    return buffer;
}

// Single compiled model container.
std::vector<uint8_t> make_single_blob_container(const std::vector<uint8_t>& section_payload,
                                                const std::vector<uint8_t>& manifest) {
    return make_container(runtime::BlobMagic::SINGLE_BLOB, section_payload, manifest);
}

// Multi-blob / shared-weights container (Story 12/13, forward-looking).
std::vector<uint8_t> make_shared_context_container(const std::vector<uint8_t>& section_payload,
                                                   const std::vector<uint8_t>& manifest) {
    return make_container(runtime::BlobMagic::SHARED_CONTEXT, section_payload, manifest);
}

// A multi-blob file: `blob_count` independent SHARED_CONTEXT containers concatenated back-to-back, each with
// its own header/manifest. The exact multi-container framing isn't finalized yet (Story 12/13), but every
// container is self-describing via `total_size`, so a reader can hop from one to the next by adding it to
// the current container's start offset - back-to-back concatenation is enough to exercise that here.
// `blob_count == 1` is a valid (degenerate) multi-blob file containing a single blob.
std::vector<uint8_t> make_multi_blob_container(size_t blob_count) {
    std::vector<uint8_t> buffer;
    for (size_t i = 0; i < blob_count; ++i) {
        const auto blob = make_shared_context_container({}, {});
        buffer.insert(buffer.end(), blob.begin(), blob.end());
    }
    return buffer;
}

// Sample container shared by the HsmHeaderTest.* tests below: a header describing 2 manifest entries and a
// 2-byte "OV" section payload. Manifest content is dummy placeholder bytes (just the correct byte count,
// `2 * kManifestEntrySize`) - no test here reads specific entry field values.
std::vector<uint8_t> make_sample_single_blob_container() {
    const std::vector<uint8_t> manifest(2 * k_manifest_entry_size, 0xCD);
    return make_single_blob_container({'O', 'V'}, manifest);
}

}  // namespace

// --- HSM wire-format layout/version compatibility --------------------------------------------------------
// Pinned to HSM_FORMAT_VERSION_MAJOR == 1. A failure in this fixture means the on-disk byte layout changed:
static_assert(runtime::HSM_FORMAT_VERSION_MAJOR == 1,
              "HSM_FORMAT_VERSION_MAJOR changed - review HsmFormatLayoutCompatibilityTest, update the pinned "
              "layout checks below to match the new format, then update this assert.");

class HsmFormatLayoutCompatibilityTest : public ::testing::Test {};

TEST_F(HsmFormatLayoutCompatibilityTest, HSMHeaderLayout) {
    static_assert(sizeof(runtime::HSMMagicType) == 5, "HSMMagicType width changed");
    static_assert(sizeof(runtime::HSMHeader) == 32, "HSMHeader total size changed");
    static_assert(offsetof(runtime::HSMHeader, magic) == 0, "HSMHeader::magic offset changed");
    static_assert(offsetof(runtime::HSMHeader, version_major) == 5, "HSMHeader::version_major offset changed");
    static_assert(offsetof(runtime::HSMHeader, version_minor) == 6, "HSMHeader::version_minor offset changed");
    static_assert(offsetof(runtime::HSMHeader, version_patch) == 7, "HSMHeader::version_patch offset changed");
    static_assert(offsetof(runtime::HSMHeader, total_size) == 8, "HSMHeader::total_size offset changed");
    static_assert(offsetof(runtime::HSMHeader, manifest_offset) == 16, "HSMHeader::manifest_offset offset changed");
    static_assert(offsetof(runtime::HSMHeader, manifest_size) == 24, "HSMHeader::manifest_size offset changed");

    // Runtime echo, for visibility in test reports (the static_asserts above already block the build on a break).
    EXPECT_EQ(sizeof(runtime::HSMHeader), 32u);
    EXPECT_EQ(offsetof(runtime::HSMHeader, manifest_size), 24u);
}

// TEST_F(HsmFormatLayoutCompatibilityTest, ManifestEntryLayout) {
//     static_assert(sizeof(runtime::ManifestEntry) == 32, "ManifestEntry total size changed");
//     static_assert(offsetof(runtime::ManifestEntry, device) == 0, "ManifestEntry::device offset changed");
//     static_assert(offsetof(runtime::ManifestEntry, tag) == 1, "ManifestEntry::tag offset changed");
//     static_assert(offsetof(runtime::ManifestEntry, mode) == 3, "ManifestEntry::mode offset changed");
//     static_assert(offsetof(runtime::ManifestEntry, reserved) == 4, "ManifestEntry::reserved offset changed");
//     static_assert(offsetof(runtime::ManifestEntry, pointer) == 8, "ManifestEntry::pointer offset changed");
//     static_assert(offsetof(runtime::ManifestEntry, inline_bytes) == 8, "ManifestEntry::inline_bytes offset changed");
//     static_assert(offsetof(runtime::ManifestEntry, reserved_tail) == 24, "ManifestEntry::reserved_tail offset
//     changed");

//     EXPECT_EQ(sizeof(runtime::ManifestEntry), 32u);
// }

TEST_F(HsmFormatLayoutCompatibilityTest, BlobMagicIsCompileTimeComparable) {
    // BlobMagic::operator==/!= must be usable in a constant expression, not just nominally marked constexpr.
    static_assert(runtime::BlobMagic::SINGLE_BLOB == runtime::BlobMagic::SINGLE_BLOB,
                  "BlobMagic::operator== must be constexpr-usable");
    static_assert(runtime::BlobMagic::SINGLE_BLOB != runtime::BlobMagic::SHARED_CONTEXT,
                  "BlobMagic::operator!= must be constexpr-usable");
    SUCCEED();
}
// --- HSM wire-format layout/version compatibility end ----------------------------------------------------

TEST(HsmHeaderTest, BlobMagicStrViewMatchesLiteral) {
    EXPECT_EQ(runtime::BlobMagic::SINGLE_BLOB.as_string_view(), "OVBLS");
    EXPECT_EQ(runtime::BlobMagic::SHARED_CONTEXT.as_string_view(), "OVWSH");
    EXPECT_NE(runtime::BlobMagic::SINGLE_BLOB, runtime::BlobMagic::SHARED_CONTEXT);
}

TEST(HsmHeaderTest, CheckSingleBlobHeader) {
    const auto blob = make_sample_single_blob_container();
    ASSERT_GE(blob.size(), sizeof(runtime::HSMHeader));

    const auto header = *reinterpret_cast<const runtime::HSMHeader*>(blob.data());
    EXPECT_EQ(header.magic, runtime::BlobMagic::SINGLE_BLOB);
    EXPECT_EQ(header.version_major, runtime::HSM_FORMAT_VERSION_MAJOR);
    EXPECT_EQ(header.version_minor, runtime::HSM_FORMAT_VERSION_MINOR);
    EXPECT_EQ(header.version_patch, runtime::HSM_FORMAT_VERSION_PATCH);
    EXPECT_EQ(header.total_size, blob.size());
    EXPECT_EQ(header.manifest_offset, sizeof(runtime::HSMHeader) + 2u);  // 2 bytes of section payload
    EXPECT_EQ(header.manifest_size, 2 * k_manifest_entry_size);          // 2 entries in the manifest
};

TEST(HsmHeaderTest, CheckMultiBlobSingleBlob) {
    const auto blob = make_multi_blob_container(1);
    ASSERT_GE(blob.size(), sizeof(runtime::HSMHeader));

    const auto header = *reinterpret_cast<const runtime::HSMHeader*>(blob.data());
    EXPECT_EQ(header.magic, runtime::BlobMagic::SHARED_CONTEXT);
    EXPECT_EQ(header.version_major, runtime::HSM_FORMAT_VERSION_MAJOR);
    EXPECT_EQ(header.version_minor, runtime::HSM_FORMAT_VERSION_MINOR);
    EXPECT_EQ(header.version_patch, runtime::HSM_FORMAT_VERSION_PATCH);
    EXPECT_EQ(header.total_size, blob.size());
    EXPECT_EQ(header.manifest_offset, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header.manifest_size, 0u);  // no entries in the manifest
};

TEST(HsmHeaderTest, CheckMultiBlobTwoBlobs) {
    const auto blob = make_multi_blob_container(2);
    ASSERT_GE(blob.size(), 2 * sizeof(runtime::HSMHeader));

    const auto header1 = *reinterpret_cast<const runtime::HSMHeader*>(blob.data());
    EXPECT_EQ(header1.magic, runtime::BlobMagic::SHARED_CONTEXT);
    EXPECT_EQ(header1.version_major, runtime::HSM_FORMAT_VERSION_MAJOR);
    EXPECT_EQ(header1.version_minor, runtime::HSM_FORMAT_VERSION_MINOR);
    EXPECT_EQ(header1.version_patch, runtime::HSM_FORMAT_VERSION_PATCH);
    EXPECT_EQ(header1.total_size, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header1.manifest_offset, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header1.manifest_size, 0u);  // no entries in the manifest

    const auto header2 = *reinterpret_cast<const runtime::HSMHeader*>(blob.data() + header1.total_size);
    EXPECT_EQ(header2.magic, runtime::BlobMagic::SHARED_CONTEXT);
    EXPECT_EQ(header2.version_major, runtime::HSM_FORMAT_VERSION_MAJOR);
    EXPECT_EQ(header2.version_minor, runtime::HSM_FORMAT_VERSION_MINOR);
    EXPECT_EQ(header2.version_patch, runtime::HSM_FORMAT_VERSION_PATCH);
    EXPECT_EQ(header2.total_size, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header2.manifest_offset, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header2.manifest_size, 0u);  // no entries in the manifest
}

}  // namespace ov::test
