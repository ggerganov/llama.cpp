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
#pragma message("crt/device_functions.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/device_functions.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_RUNTIME_H__
#endif

#if !defined(__CUDA_INTERNAL_COMPILATION__)

#define __CUDA_INTERNAL_COMPILATION__
#define __text__
#define __surf__
#define __name__shadow_var(c, cpp) \
        #c
#define __name__text_var(c, cpp) \
        #cpp
#define __host__shadow_var(c, cpp) \
        cpp
#define __text_var(c, cpp) \
        cpp
#define __device_fun(fun) \
        #fun
#define __device_var(var) \
        #var
#define __device__text_var(c, cpp) \
        #c
#define __device__shadow_var(c, cpp) \
        #c

#if defined(_WIN32) && !defined(_WIN64)

#define __pad__(f) \
        f

#else /* _WIN32 && !_WIN64 */

#define __pad__(f)

#endif /* _WIN32 && !_WIN64 */

#include "builtin_types.h"
#include "storage_class.h"

#else /* !__CUDA_INTERNAL_COMPILATION__ */

template <typename T>
static inline T *__cudaAddressOf(T &val) 
{
    return (T *)((void *)(&(const_cast<char &>(reinterpret_cast<const volatile char &>(val)))));
}

#define __cudaRegisterBinary(X)                                                   \
        __cudaFatCubinHandle = __cudaRegisterFatBinary((void*)&__fatDeviceText); \
        { void (*callback_fp)(void **) =  (void (*)(void **))(X); (*callback_fp)(__cudaFatCubinHandle); __cudaRegisterFatBinaryEnd(__cudaFatCubinHandle); }\
        atexit(__cudaUnregisterBinaryUtil)
        
#define __cudaRegisterVariable(handle, var, ext, size, constant, global) \
        __cudaRegisterVar(handle, (char*)&__host##var, (char*)__device##var, __name##var, ext, size, constant, global)
#define __cudaRegisterManagedVariable(handle, var, ext, size, constant, global) \
        __cudaRegisterManagedVar(handle, (void **)&__host##var, (char*)__device##var, __name##var, ext, size, constant, global)

#define __cudaRegisterGlobalTexture(handle, tex, dim, norm, ext) \
        __cudaRegisterTexture(handle, (const struct textureReference*)&tex, (const void**)(void*)__device##tex, __name##tex, dim, norm, ext)
#define __cudaRegisterGlobalSurface(handle, surf, dim, ext) \
        __cudaRegisterSurface(handle, (const struct surfaceReference*)&surf, (const void**)(void*)__device##surf, __name##surf, dim, ext)
#define __cudaRegisterEntry(handle, funptr, fun, thread_limit) \
        __cudaRegisterFunction(handle, (const char*)funptr, (char*)__device_fun(fun), #fun, -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0)

extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(
  dim3         *gridDim,
  dim3         *blockDim,
  size_t       *sharedMem,
  void         *stream
);

#define __cudaLaunchPrologue(size) \
        void * __args_arr[size]; \
        int __args_idx = 0
        
#define __cudaSetupArg(arg, offset) \
        __args_arr[__args_idx] = (void *)__cudaAddressOf(arg); ++__args_idx
          
#define __cudaSetupArgSimple(arg, offset) \
        __args_arr[__args_idx] = (void *)(char *)&arg; ++__args_idx
        
#if defined(__GNUC__)
#define __NV_ATTR_UNUSED_FOR_LAUNCH __attribute__((unused))
#else  /* !__GNUC__ */
#define __NV_ATTR_UNUSED_FOR_LAUNCH
#endif  /* __GNUC__ */

/* the use of __args_idx in the expression below avoids host compiler warning about it being an
   unused variable when the launch has no arguments */
#define __cudaLaunch(fun) \
        { volatile static char *__f __NV_ATTR_UNUSED_FOR_LAUNCH;  __f = fun; \
          dim3 __gridDim, __blockDim;\
          size_t __sharedMem; \
          cudaStream_t __stream; \
          if (__cudaPopCallConfiguration(&__gridDim, &__blockDim, &__sharedMem, &__stream) != cudaSuccess) \
            return; \
          if (__args_idx == 0) {\
            (void)cudaLaunchKernel(fun, __gridDim, __blockDim, &__args_arr[__args_idx], __sharedMem, __stream);\
          } else { \
            (void)cudaLaunchKernel(fun, __gridDim, __blockDim, &__args_arr[0], __sharedMem, __stream);\
          }\
        }

#if defined(__GNUC__)
#define __nv_dummy_param_ref(param) \
        { volatile static void **__ref __attribute__((unused)); __ref = (volatile void **)param; }
#else /* __GNUC__ */
#define __nv_dummy_param_ref(param) \
        { volatile static void **__ref; __ref = (volatile void **)param; }
#endif /* __GNUC__ */

static void ____nv_dummy_param_ref(void *param) __nv_dummy_param_ref(param)

#define __REGISTERFUNCNAME_CORE(X) __cudaRegisterLinkedBinary##X
#define __REGISTERFUNCNAME(X) __REGISTERFUNCNAME_CORE(X)

extern "C" {
void __REGISTERFUNCNAME( __NV_MODULE_ID ) ( void (*)(void **), void *, void *, void (*)(void *));
}

#define __TO_STRING_CORE(X) #X
#define __TO_STRING(X) __TO_STRING_CORE(X)

extern "C" {
#if defined(_WIN32)
#pragma data_seg("__nv_module_id")
  static const __declspec(allocate("__nv_module_id")) unsigned char __module_id_str[] = __TO_STRING(__NV_MODULE_ID);
#pragma data_seg()
#elif defined(__APPLE__)
  static const unsigned char __module_id_str[] __attribute__((section ("__NV_CUDA,__nv_module_id"))) = __TO_STRING(__NV_MODULE_ID);
#else
  static const unsigned char __module_id_str[] __attribute__((section ("__nv_module_id"))) = __TO_STRING(__NV_MODULE_ID);
#endif

#undef __FATIDNAME_CORE
#undef __FATIDNAME
#define __FATIDNAME_CORE(X) __fatbinwrap##X
#define __FATIDNAME(X) __FATIDNAME_CORE(X)

#define  ____cudaRegisterLinkedBinary(X) \
{ __REGISTERFUNCNAME(__NV_MODULE_ID) (( void (*)(void **))(X), (void *)&__FATIDNAME(__NV_MODULE_ID), (void *)&__module_id_str, (void (*)(void *))&____nv_dummy_param_ref); }

}

extern "C" {
extern void** CUDARTAPI __cudaRegisterFatBinary(
  void *fatCubin
);

extern void CUDARTAPI __cudaRegisterFatBinaryEnd(
  void **fatCubinHandle
);

extern void CUDARTAPI __cudaUnregisterFatBinary(
  void **fatCubinHandle
);

extern void CUDARTAPI __cudaRegisterVar(
        void **fatCubinHandle,
        char  *hostVar,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global
);

extern void CUDARTAPI __cudaRegisterManagedVar(
        void **fatCubinHandle,
        void **hostVarPtrAddress,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global
);

extern char CUDARTAPI __cudaInitModule(
        void **fatCubinHandle
);

extern void CUDARTAPI __cudaRegisterTexture(
        void                    **fatCubinHandle,
  const struct textureReference  *hostVar,
  const void                    **deviceAddress,
  const char                     *deviceName,
        int                       dim,       
        int                       norm,      
        int                        ext        
);

extern void CUDARTAPI __cudaRegisterSurface(
        void                    **fatCubinHandle,
  const struct surfaceReference  *hostVar,
  const void                    **deviceAddress,
  const char                     *deviceName,
        int                       dim,       
        int                       ext        
);

extern void CUDARTAPI __cudaRegisterFunction(
        void   **fatCubinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize
);

#if defined(__APPLE__)
extern "C" int atexit(void (*)(void));

#elif  defined(__GNUC__) && !defined(__ANDROID__) && !defined(__HORIZON__)
extern int atexit(void(*)(void)) throw();

#elif defined(__HORIZON__)

// __TEMP_WAR__ 200132570 HOS : Disable atexit call until it works
#define atexit(p)

#else /* __GNUC__ && !__ANDROID__ */
extern int __cdecl atexit(void(__cdecl *)(void));
#endif

}

static void **__cudaFatCubinHandle;

static void __cdecl __cudaUnregisterBinaryUtil(void)
{
  ____nv_dummy_param_ref((void *)&__cudaFatCubinHandle);
  __cudaUnregisterFatBinary(__cudaFatCubinHandle);
}

static char __nv_init_managed_rt_with_module(void **handle)
{
  return __cudaInitModule(handle);
}

#include "common_functions.h"

#pragma pack()

#if defined(_WIN32)

#pragma warning(disable: 4099)

#if !defined(_WIN64)

#pragma warning(disable: 4408)

#endif /* !_WIN64 */

#endif /* _WIN32 */

#endif /* !__CUDA_INTERNAL_COMPILATION__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_RUNTIME_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_RUNTIME_H__
#endif
