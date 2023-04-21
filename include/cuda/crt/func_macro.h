/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2008-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#if defined(_MSC_VER)
#pragma message("crt/func_macro.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/func_macro.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_FUNC_MACRO_H__
#endif

#if !defined(__FUNC_MACRO_H__)
#define __FUNC_MACRO_H__

#if !defined(__CUDA_INTERNAL_COMPILATION__)

#error -- incorrect inclusion of a cudart header file

#endif /* !__CUDA_INTERNAL_COMPILATION__ */

#if defined(__GNUC__)

#define __func__(decl) \
        inline decl

#define __device_func__(decl) \
        static __attribute__((__unused__)) decl

#elif defined(_WIN32)

#define __func__(decl) \
        static inline decl

#define __device_func__(decl) \
        static decl

#endif /* __GNUC__ */

#endif /* __FUNC_MACRO_H__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_FUNC_MACRO_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_FUNC_MACRO_H__
#endif
