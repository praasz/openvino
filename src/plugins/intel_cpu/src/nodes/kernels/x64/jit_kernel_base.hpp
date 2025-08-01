// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cassert>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/visibility.hpp"
#include "utils/cpu_utils.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/jit_generator.hpp"
#    include "registers_pool.hpp"
#endif  // OPENVINO_ARCH_X86_64

namespace ov::intel_cpu::kernel {

class JitKernelBase;

#if defined(OPENVINO_ARCH_X86_64)

#    define getReg64() RegistersPool::Reg<Xbyak::Reg64>(registersPool)
#    define getReg32() RegistersPool::Reg<Xbyak::Reg32>(registersPool)
#    define getVmm()   RegistersPool::Reg<Vmm>(registersPool)
#    define getMask()  RegistersPool::Reg<Vmask>(registersPool)

class JitKernelBase : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    JitKernelBase(const char* name, dnnl::impl::cpu::x64::cpu_isa_t max_cpu_isa);

    dnnl::impl::cpu::x64::cpu_isa_t getIsa() {
        return m_isa;
    }

    size_t getVectorLen() const {
        return vlen;
    }

    void uni_vfmsub132ps(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op);

    void uni_vfnmadd132ps(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op);

    void uni_vfmsub231ps(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op);

    void uni_vpaddd(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op) {
        jit_generator_t::uni_vpaddd(v_dst, v_src, op);
    }

    void uni_vpaddd(const Xbyak::Ymm& v_dst, const Xbyak::Ymm& v_src, const Xbyak::Operand& op);

    void uni_vpaddq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op);

    void uni_vpsubd(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op) {
        jit_generator_t::uni_vpsubd(v_dst, v_src, op);
    }

    void uni_vpsubd(const Xbyak::Ymm& v_dst, const Xbyak::Ymm& v_src, const Xbyak::Operand& op);

    void uni_vsubpd(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op);

    void uni_vmulpd(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op);

    void uni_vpmuludq(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op_2);

    void uni_vdivps(const Xbyak::Xmm& v_dst, const Xbyak::Operand& op1, const Xbyak::Operand& op2);

    void uni_vdivpd(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op2);

    void uni_vandps(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& vSrs, const Xbyak::Operand& op);

    void uni_vandnps(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& vSrs, const Xbyak::Operand& op);

    void uni_kmovd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc) {
        kmovd(kDst, kSrc);
    }

    void uni_kmovd(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src) {
        uni_vmovups(v_dst, v_src);
    }

    void uni_kandd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc1, const Xbyak::Opmask& kSrc2) {
        kandd(kDst, kSrc1, kSrc2);
    }

    void uni_kandd(const Xbyak::Xmm& kDst, const Xbyak::Xmm& kSrc1, const Xbyak::Xmm& kSrc2) {
        uni_vandps(kDst, kSrc1, kSrc2);
    }

    void uni_vpbroadcastd(const Xbyak::Xmm& x, const Xbyak::Operand& op);

    void uni_vpbroadcastd(const Xbyak::Ymm& x, const Xbyak::Operand& op);

    void uni_vpbroadcastq(const Xbyak::Xmm& x, const Xbyak::Operand& op);

    void uni_vroundpd(const Xbyak::Xmm& v_dst, const Xbyak::Operand& op, uint8_t imm);

    void uni_vcvtdq2pd(const Xbyak::Xmm& v_dst, const Xbyak::Operand& op);

    void uni_vcvtpd2dq(const Xbyak::Xmm& v_dst, const Xbyak::Operand& op);

    void uni_vpmovzxdq(const Xbyak::Xmm& v_dst, const Xbyak::Operand& op);

    void uni_vshufpd(const Xbyak::Xmm& v_dst, const Xbyak::Xmm& v_src, const Xbyak::Operand& op, uint8_t imm);

    void gatherdd(const Xbyak::Xmm& v_dst,
                  const Xbyak::Reg64& rSrcPtr,
                  const Xbyak::Xmm& vSrcShift,
                  const Xbyak::Opmask& kReadMask,
                  bool useMask = true,
                  bool zeroFill = false);

    void gatherdd(const Xbyak::Xmm& v_dst,
                  const Xbyak::Reg64& rSrcPtr,
                  const Xbyak::Xmm& vSrcShift,
                  const Xbyak::Xmm& vReadMask,
                  bool useMask = true,
                  bool zeroFill = false);

    void gatherdd(const Xbyak::Ymm& v_dst,
                  const Xbyak::Reg64& rSrcPtr,
                  const Xbyak::Ymm& vSrcShift,
                  const Xbyak::Ymm& vReadMask,
                  bool useMask = true,
                  bool zeroFill = false);

    void fillRestWorkMask(const Xbyak::Opmask& kDstMask, const Xbyak::Reg64& rWorkRest);

    void fillRestWorkMask(const Xbyak::Xmm& xmmDstMask, const Xbyak::Reg64& rWorkRest, uint64_t typeSize = 4);

    void fillRestWorkMask(const Xbyak::Ymm& ymmDstMask, const Xbyak::Reg64& rWorkRest, uint64_t typeSize = 4);

    void load(const Xbyak::Xmm& v_dst,
              const Xbyak::Address& srcAddr,
              const Xbyak::Reg64& rLoadNum,
              size_t typeSize,
              bool zeroFill = false);

    void load(const Xbyak::Ymm& v_dst,
              const Xbyak::Address& srcAddr,
              const Xbyak::Reg64& rLoadNum,
              size_t typeSize,
              bool zeroFill = false);

    void store(const Xbyak::Address& dstAddr,
               const Xbyak::Xmm& v_src,
               const Xbyak::Reg64& rToStoreNum,
               size_t typeSize);

    void store(const Xbyak::Address& dstAddr,
               const Xbyak::Ymm& v_src,
               const Xbyak::Reg64& rToStoreNum,
               size_t typeSize);

    // Makes gather from memory under the vReadMask and writes to the memory m128.
    void memMovDD(const Xbyak::Reg64& rDst,
                  const Xbyak::Reg64& rSrc,
                  const Xbyak::Xmm& vReadMask,
                  const Xbyak::Xmm& vSrcShift,
                  const Xbyak::Reg64& rToStoreNum,
                  bool useMask = true,
                  bool zeroFill = false);

    // Makes gather from the memory under the vReadMask and writes to the memory m256.
    void memMovDD(const Xbyak::Reg64& rDst,
                  const Xbyak::Reg64& rSrc,
                  const Xbyak::Ymm& vReadMask,
                  const Xbyak::Ymm& vSrcShift,
                  const Xbyak::Reg64& rToStoreNum,
                  bool useMask = true,
                  bool zeroFill = false);

protected:
    static bool isValidIsa(dnnl::impl::cpu::x64::cpu_isa_t isa) {
        return dnnl::impl::cpu::x64::mayiuse(isa);
    }

    const dnnl::impl::cpu::x64::cpu_isa_t m_isa;
    RegistersPool::Ptr registersPool;
    size_t vlen;

    enum : uint8_t {
        // Comparison predicate operand (immediate byte) for single-precision floating-point values.
        CMP_EQ_PS = 0,  // Equal (ordered, non-signaling)
        CMP_LT_PS,      // Less-than (ordered, signaling)
        CMP_LE_PS,      // Less-than-or-equal (ordered, signaling)
        CMP_UNORD_PS,   // Unordered (non-signaling)
        CMP_NEQ_PS,     // Not-equal (unordered, non-signaling)
        CMP_NLT_PS,     // Not-less-than (unordered, signaling)
        CMP_NLE_PS,     // Not-less-than-or-equal (unordered, signaling)
        CMP_ORD_PS      // Ordered (non-signaling)
    };
};

template <typename CompileParams, typename CallArgs>
class JitKernel : public JitKernelBase {
public:
    using KernelFunc = void (*)(const CallArgs*);

    explicit JitKernel(const char* name, const CompileParams& jcp, dnnl::impl::cpu::x64::cpu_isa_t max_cpu_isa)
        : JitKernelBase{name, max_cpu_isa},
          m_jcp{jcp} {}

    ~JitKernel() override = default;

    dnnl::impl::status_t create_kernel() override {
        const dnnl::impl::status_t code = jit_generator_t::create_kernel();
        OPENVINO_ASSERT(code == dnnl::impl::status::success,
                        "Could not create kernel. Error code: ",
                        std::to_string(code),
                        ". ",
                        "Xbyak error code: ",
                        Xbyak::ConvertErrorToString(Xbyak::GetError()));
        m_func = jit_kernel_cast<decltype(m_func)>(jit_ker());
        return code;
    }

    void operator()(const CallArgs* args) const {
        assert(m_func);
        m_func(args);
    }

    void operator()(const CallArgs& args) const {
        this->operator()(&args);
    }

    template <template <dnnl::impl::cpu::x64::cpu_isa_t isa> class KernelT>
    static std::shared_ptr<JitKernel<CompileParams, CallArgs>> createInstance(const CompileParams& jcp) {
        std::shared_ptr<JitKernel<CompileParams, CallArgs>> res;

        try {
#    define IF_ISA_CASE(ISA)                    \
        if (dnnl::impl::cpu::x64::mayiuse(ISA)) \
            res.reset(new KernelT<ISA>(jcp));   \
        else

            IF_ISA_CASE(dnnl::impl::cpu::x64::avx512_core)
            IF_ISA_CASE(dnnl::impl::cpu::x64::avx2)
            IF_ISA_CASE(dnnl::impl::cpu::x64::sse41);

#    undef IF_ISA_CASE

            if (res) {
                res->create_kernel();
            }
        } catch (...) {
            return nullptr;
        }

        return res;
    }

protected:
    CompileParams m_jcp;

private:
    KernelFunc m_func{nullptr};
};

#endif  // OPENVINO_ARCH_X86_64

}  // namespace ov::intel_cpu::kernel
