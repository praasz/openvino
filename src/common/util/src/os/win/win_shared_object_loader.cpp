// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

//
// LoadLibraryA, LoadLibraryW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// GetModuleHandleExA, GetModuleHandleExW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// GetModuleHandleA, GetModuleHandleW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// SetDllDirectoryA, SetDllDirectoryW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//
// GetDllDirectoryA, GetDllDirectoryW:
//  WINAPI_FAMILY_DESKTOP_APP - FAIL
//  WINAPI_FAMILY_PC_APP - FAIL (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL
//  WINAPI_FAMILY_GAMES - FAIL
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//
// SetupDiGetClassDevsA, SetupDiEnumDeviceInfo, SetupDiGetDeviceInstanceIdA, SetupDiDestroyDeviceInfoList:
//  WINAPI_FAMILY_DESKTOP_APP - FAIL (default)
//  WINAPI_FAMILY_PC_APP - FAIL (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL
//  WINAPI_FAMILY_GAMES - FAIL
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//

#if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#    error "Only WINAPI_PARTITION_DESKTOP is supported, because of LoadLibrary[A|W]"
#endif

#include <direct.h>

#include <mutex>

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <windows.h>

namespace ov {
namespace util {
std::shared_ptr<void> load_shared_object(const std::filesystem::path& path) {
    void* shared_object = nullptr;
    using GetDllDirectoryW_Fnc = DWORD (*)(DWORD, LPWSTR);
    static GetDllDirectoryW_Fnc IEGetDllDirectoryW = nullptr;
    if (HMODULE hm = GetModuleHandleW(L"kernel32.dll")) {
        IEGetDllDirectoryW = reinterpret_cast<GetDllDirectoryW_Fnc>(GetProcAddress(hm, "GetDllDirectoryW"));
    }
    // ExcludeCurrentDirectory
#if !WINAPI_PARTITION_SYSTEM
    if (IEGetDllDirectoryW && IEGetDllDirectoryW(0, NULL) <= 1) {
        SetDllDirectoryW(L"");
    }
    if (IEGetDllDirectoryW) {
        DWORD nBufferLength = IEGetDllDirectoryW(0, NULL);
        std::vector<WCHAR> lpBuffer(nBufferLength);
        IEGetDllDirectoryW(nBufferLength, &lpBuffer.front());

        auto dirname = get_directory(path);

        SetDllDirectoryW(dirname.c_str());

        shared_object = LoadLibraryW(path);

        SetDllDirectoryW(&lpBuffer.front());
    }
#endif
    if (!shared_object) {
        shared_object = LoadLibraryW(path);
    }
    if (!shared_object) {
        char cwd[1024];
        std::stringstream ss;
        ss << "Cannot load library '" << path.string() << "': " << GetLastError()
           << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        throw std::runtime_error(ss.str());
    }
    return {shared_object, [](void* shared_object) {
                FreeLibrary(reinterpret_cast<HMODULE>(shared_object));
            }};
}

void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbol_name) {
    if (!shared_object) {
        std::stringstream ss;
        ss << "Cannot get '" << symbol_name << "' content from unknown library!";
        throw std::runtime_error(ss.str());
    }
    auto procAddr = reinterpret_cast<void*>(
        GetProcAddress(reinterpret_cast<HMODULE>(const_cast<void*>(shared_object.get())), symbol_name));
    if (procAddr == nullptr) {
        std::stringstream ss;
        ss << "GetProcAddress cannot locate method '" << symbol_name << "': " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    return procAddr;
}
}  // namespace util
}  // namespace ov
