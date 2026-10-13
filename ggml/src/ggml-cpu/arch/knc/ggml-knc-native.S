/* SPDX-License-Identifier: MIT
 * Native Knights Corner MVEX/IMCI, assembled by MPSS GNU binutils.
 * No Intel intrinsics and no AVX-512 encoding. All vector registers are
 * caller-saved in the K1OM ABI. zmm0..zmm3 are the only accumulation chains.
 */
#if !defined(__k1om__) || !defined(__KNC__)
#error "Assemble with the MPSS GCC K1OM cross-compiler and -mknc"
#endif
#if defined(__AVX512F__) || defined(__AVX512BW__) || defined(__AVX2__)
#error "AVX-512/AVX2 are incompatible with Knights Corner IMCI"
#endif
#if defined(TARGET_KNC_3120A) == defined(TARGET_KNC_7120P)
#error "Select exactly one KNC hardware profile"
#endif
#ifdef TARGET_KNC_3120A
#define KNC_L1_ITERS 4
#define KNC_L2_ITERS 16
#endif
#ifdef TARGET_KNC_7120P
#define KNC_L1_ITERS 8
#define KNC_L2_ITERS 32
#endif

/* Each logical iteration is sixteen values. Packed offsets use the actual
 * Q5_K/Q8_K layout, never an FP32 byte distance. r10 is remaining blocks;
 * r8/r9 point to the current packed blocks. No out-of-range prefetch address
 * is formed or issued. These macros expand entirely to immediate offsets. */
.macro KNC_PREFETCH_PACKED at, distance, hint
    cmpq $((\at + 16*\distance)/256), %r10
    jbe .Lpacked_skip\@
    \hint (((\at+16*\distance)/256)*176+48+(((\at+16*\distance)%256)/64)*32+(\at+16*\distance)%32)(%r8)
    \hint (((\at+16*\distance)/256)*176+16+(\at+16*\distance)%32)(%r8)
    \hint (((\at+16*\distance)/256)*292+4+(\at+16*\distance)%256)(%r9)
.Lpacked_skip\@:
.endm
.macro KNC_PREFETCH_STREAM at
    KNC_PREFETCH_PACKED \at, KNC_L1_ITERS, vprefetch0
    KNC_PREFETCH_PACKED \at, KNC_L2_ITERS, vprefetch1
.endm

/* Four independent sixteen-lane streams per group of 64 logical values.
 * zmm4/5: nibble/bit masks; zmm6/7: immutable high-bit planes.
 * zmm8/9: packed nibbles; zmm10..13: decoded values; zmm14..17: Q8 FP32;
 * zmm18/19: scales; zmm20..23: high-bit temporaries. No stack vector spills.
 * Converted byte loads require only sixteen-byte source alignment. */
.macro KNC_Q5_GROUP group, q4off, q8off
    vmovdqa32 \q4off(%rdi){uint8}, %zmm8
    vmovdqa32 (\q4off+16)(%rdi){uint8}, %zmm9
    vmovaps \q8off(%rdx){sint8}, %zmm14
    vmovaps (\q8off+16)(%rdx){sint8}, %zmm15
    vmovaps (\q8off+32)(%rdx){sint8}, %zmm16
    vmovaps (\q8off+48)(%rdx){sint8}, %zmm17

    vpandd %zmm4, %zmm8, %zmm10
    vpandd %zmm4, %zmm9, %zmm11
    vpsrld $4, %zmm8, %zmm12
    vpsrld $4, %zmm9, %zmm13
    vpsrld $\group, %zmm6, %zmm20
    vpsrld $\group, %zmm7, %zmm21
    vpsrld $(\group+1), %zmm6, %zmm22
    vpsrld $(\group+1), %zmm7, %zmm23
    vpandd %zmm5, %zmm20, %zmm20
    vpandd %zmm5, %zmm21, %zmm21
    vpandd %zmm5, %zmm22, %zmm22
    vpandd %zmm5, %zmm23, %zmm23
    vpslld $4, %zmm20, %zmm20
    vpslld $4, %zmm21, %zmm21
    vpslld $4, %zmm22, %zmm22
    vpslld $4, %zmm23, %zmm23
    vpord %zmm20, %zmm10, %zmm10
    vpord %zmm21, %zmm11, %zmm11
    vpord %zmm22, %zmm12, %zmm12
    vpord %zmm23, %zmm13, %zmm13

    vcvtfxpntdq2ps $0, (4*\group)(%rcx){1to16}, %zmm18
    vcvtfxpntdq2ps $0, (4*(\group+1))(%rcx){1to16}, %zmm19
    vcvtfxpntdq2ps $0, %zmm10, %zmm10
    vcvtfxpntdq2ps $0, %zmm11, %zmm11
    vcvtfxpntdq2ps $0, %zmm12, %zmm12
    vcvtfxpntdq2ps $0, %zmm13, %zmm13
    vmulps %zmm18, %zmm10, %zmm10
    vmulps %zmm18, %zmm11, %zmm11
    vmulps %zmm19, %zmm12, %zmm12
    vmulps %zmm19, %zmm13, %zmm13
    KNC_PREFETCH_STREAM \q8off
    vfmadd231ps %zmm14, %zmm10, %zmm0
    KNC_PREFETCH_STREAM (\q8off+16)
    vfmadd231ps %zmm15, %zmm11, %zmm1
    KNC_PREFETCH_STREAM (\q8off+32)
    vfmadd231ps %zmm16, %zmm12, %zmm2
    KNC_PREFETCH_STREAM (\q8off+48)
    vfmadd231ps %zmm17, %zmm13, %zmm3
.endm

.text
.p2align 4
.globl ggml_knc_q5_block_native
.type ggml_knc_q5_block_native, @function
ggml_knc_q5_block_native:
    /* q4=rdi, qh=rsi, q8=rdx, scales=rcx, packed x=r8, packed y=r9;
     * remaining blocks and aligned output lanes are stack arguments 7/8. */
    movq 8(%rsp), %r10
    movq 16(%rsp), %r11
    vpxord %zmm0, %zmm0, %zmm0
    vpxord %zmm1, %zmm1, %zmm1
    vpxord %zmm2, %zmm2, %zmm2
    vpxord %zmm3, %zmm3, %zmm3
    vpbroadcastd .Lmask15(%rip), %zmm4
    vpbroadcastd .Lmask1(%rip), %zmm5
    vmovdqa32 (%rsi){uint8}, %zmm6
    vmovdqa32 16(%rsi){uint8}, %zmm7
    KNC_Q5_GROUP 0, 0, 0
    KNC_Q5_GROUP 2, 32, 64
    KNC_Q5_GROUP 4, 64, 128
    KNC_Q5_GROUP 6, 96, 192
    /* Every combined lane is an exact integer with |lane| <= 3,999,744.
     * Convert before horizontal reduction: the full sum may exceed 2^24. */
    vaddps %zmm1, %zmm0, %zmm0
    vaddps %zmm3, %zmm2, %zmm2
    vaddps %zmm2, %zmm0, %zmm0
    vcvtfxpntps2dq $0, %zmm0, %zmm0
    vmovdqa32 %zmm0, (%r11)
    ret
.size ggml_knc_q5_block_native, .-ggml_knc_q5_block_native

/* n=rdi, x=rsi, y=rdx, aligned output lanes=rcx; i=rax, n-i=r8.
 * The bound is tested before computing each future FP32 address. */
.macro KNC_PREFETCH_F32 offset, distance, hint
    cmpq $(\offset+16*\distance), %r8
    jbe .Lf32_skip\@
    \hint (4*(\offset+16*\distance))(%rsi,%rax,4)
    \hint (4*(\offset+16*\distance))(%rdx,%rax,4)
.Lf32_skip\@:
.endm
.macro KNC_F32_STREAM offset, xreg, yreg, accumulator
    KNC_PREFETCH_F32 \offset, KNC_L1_ITERS, vprefetch0
    KNC_PREFETCH_F32 \offset, KNC_L2_ITERS, vprefetch1
    vfmadd231ps %\yreg, %\xreg, %\accumulator
.endm

.p2align 4
.globl ggml_knc_f32_vectors_native
.type ggml_knc_f32_vectors_native, @function
ggml_knc_f32_vectors_native:
    xorl %eax, %eax
    movq %rdi, %r8
    vpxord %zmm0, %zmm0, %zmm0
    vpxord %zmm1, %zmm1, %zmm1
    vpxord %zmm2, %zmm2, %zmm2
    vpxord %zmm3, %zmm3, %zmm3
    cmpq $64, %r8
    jb .Lf32_tail_vectors
.Lf32_four_streams:
    vmovaps (%rsi,%rax,4), %zmm4
    vmovaps 64(%rsi,%rax,4), %zmm5
    vmovaps 128(%rsi,%rax,4), %zmm6
    vmovaps 192(%rsi,%rax,4), %zmm7
    vmovaps (%rdx,%rax,4), %zmm8
    vmovaps 64(%rdx,%rax,4), %zmm9
    vmovaps 128(%rdx,%rax,4), %zmm10
    vmovaps 192(%rdx,%rax,4), %zmm11
    KNC_F32_STREAM 0, zmm4, zmm8, zmm0
    KNC_F32_STREAM 16, zmm5, zmm9, zmm1
    KNC_F32_STREAM 32, zmm6, zmm10, zmm2
    KNC_F32_STREAM 48, zmm7, zmm11, zmm3
    addq $64, %rax
    subq $64, %r8
    cmpq $64, %r8
    jae .Lf32_four_streams
.Lf32_tail_vectors:
    cmpq $16, %r8
    jb .Lf32_reduce
.Lf32_one_stream:
    vmovaps (%rsi,%rax,4), %zmm4
    vmovaps (%rdx,%rax,4), %zmm8
    KNC_F32_STREAM 0, zmm4, zmm8, zmm0
    addq $16, %rax
    subq $16, %r8
    cmpq $16, %r8
    jae .Lf32_one_stream
.Lf32_reduce:
    vaddps %zmm1, %zmm0, %zmm0
    vaddps %zmm3, %zmm2, %zmm2
    vaddps %zmm2, %zmm0, %zmm0
    vmovaps %zmm0, (%rcx)
    ret
.size ggml_knc_f32_vectors_native, .-ggml_knc_f32_vectors_native

.section .rodata
.p2align 2
.Lmask15:
    .long 15
.Lmask1:
    .long 1
.section .note.GNU-stack,"",@progbits
