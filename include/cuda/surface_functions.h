/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__SURFACE_FUNCTIONS_H__)
#define __SURFACE_FUNCTIONS_H__

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"
#include "cuda_surface_types.h"

#if defined(_WIN32)
# define __DEPRECATED__ __declspec(deprecated)
#else
# define __DEPRECATED__  __attribute__((deprecated))
#endif



#ifdef __CUDA_ARCH__
template <typename T> struct __nv_surf_trait {  typedef void * cast_type; };

template<> struct __nv_surf_trait<char> {  typedef char * cast_type; };
template<> struct __nv_surf_trait<signed char> {  typedef signed char * cast_type; };
template<> struct __nv_surf_trait<unsigned char> {  typedef unsigned char * cast_type; };
template<> struct __nv_surf_trait<char1> {  typedef char1 * cast_type; };
template<> struct __nv_surf_trait<uchar1> {  typedef uchar1 * cast_type; };
template<> struct __nv_surf_trait<char2> {  typedef char2 * cast_type; };
template<> struct __nv_surf_trait<uchar2> {  typedef uchar2 * cast_type; };
template<> struct __nv_surf_trait<char4> {  typedef char4 * cast_type; };
template<> struct __nv_surf_trait<uchar4> {  typedef uchar4 * cast_type; };
template<> struct __nv_surf_trait<short> {  typedef short * cast_type; };
template<> struct __nv_surf_trait<unsigned short> {  typedef unsigned short * cast_type; };
template<> struct __nv_surf_trait<short1> {  typedef short1 * cast_type; };
template<> struct __nv_surf_trait<ushort1> {  typedef ushort1 * cast_type; };
template<> struct __nv_surf_trait<short2> {  typedef short2 * cast_type; };
template<> struct __nv_surf_trait<ushort2> {  typedef ushort2 * cast_type; };
template<> struct __nv_surf_trait<short4> {  typedef short4 * cast_type; };
template<> struct __nv_surf_trait<ushort4> {  typedef ushort4 * cast_type; };
template<> struct __nv_surf_trait<int> {  typedef int * cast_type; };
template<> struct __nv_surf_trait<unsigned int> {  typedef unsigned int * cast_type; };
template<> struct __nv_surf_trait<int1> {  typedef int1 * cast_type; };
template<> struct __nv_surf_trait<uint1> {  typedef uint1 * cast_type; };
template<> struct __nv_surf_trait<int2> {  typedef int2 * cast_type; };
template<> struct __nv_surf_trait<uint2> {  typedef uint2 * cast_type; };
template<> struct __nv_surf_trait<int4> {  typedef int4 * cast_type; };
template<> struct __nv_surf_trait<uint4> {  typedef uint4 * cast_type; };
template<> struct __nv_surf_trait<long long> {  typedef long long * cast_type; };
template<> struct __nv_surf_trait<unsigned long long> {  typedef unsigned long long * cast_type; };
template<> struct __nv_surf_trait<longlong1> {  typedef longlong1 * cast_type; };
template<> struct __nv_surf_trait<ulonglong1> {  typedef ulonglong1 * cast_type; };
template<> struct __nv_surf_trait<longlong2> {  typedef longlong2 * cast_type; };
template<> struct __nv_surf_trait<ulonglong2> {  typedef ulonglong2 * cast_type; };
#if !defined(__LP64__)
template<> struct __nv_surf_trait<long> {  typedef int * cast_type; };
template<> struct __nv_surf_trait<unsigned long> {  typedef unsigned int * cast_type; };
template<> struct __nv_surf_trait<long1> {  typedef int1 * cast_type; };
template<> struct __nv_surf_trait<ulong1> {  typedef uint1 * cast_type; };
template<> struct __nv_surf_trait<long2> {  typedef int2 * cast_type; };
template<> struct __nv_surf_trait<ulong2> {  typedef uint2 * cast_type; };
template<> struct __nv_surf_trait<long4> {  typedef uint4 * cast_type; };
template<> struct __nv_surf_trait<ulong4> {  typedef int4 * cast_type; };
#endif
template<> struct __nv_surf_trait<float> {  typedef float * cast_type; };
template<> struct __nv_surf_trait<float1> {  typedef float1 * cast_type; };
template<> struct __nv_surf_trait<float2> {  typedef float2 * cast_type; };
template<> struct __nv_surf_trait<float4> {  typedef float4 * cast_type; };
#endif /* defined(__CUDA_ARCH__) */

template <typename T>
static __DEPRECATED__ __device__  __forceinline__ void surf1Dread(T *res, surface<void, cudaSurfaceType1D> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf1Dread_v2", (void *)res, s, surf, x, mode);
#endif     
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__  T surf1Dread(surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  T temp;
  __nv_tex_surf_handler("__surf1Dread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, mode);
  return temp;
#endif
}
  
template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf1Dread(T *res, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  *res = surf1Dread<T>(surf, x, mode);
#endif /* __CUDA_ARCH__ */  
}


template <typename T>
static __DEPRECATED__ __device__  __forceinline__ void surf2Dread(T *res, surface<void, cudaSurfaceType2D> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf2Dread_v2", (void *)res, s, surf, x, y, mode);
#endif     
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__  T surf2Dread(surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  T temp;
  __nv_tex_surf_handler("__surf2Dread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, mode);
  return temp;
#endif
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf2Dread(T *res, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  *res = surf2Dread<T>(surf, x, y, mode);
#endif /* __CUDA_ARCH__ */  
}


template <typename T>
static __DEPRECATED__ __device__  __forceinline__ void surf3Dread(T *res, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf3Dread_v2", (void *)res, s, surf, x, y, z, mode);
#endif     
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__  T surf3Dread(surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  T temp;
  __nv_tex_surf_handler("__surf3Dread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, z, mode);
  return temp;
#endif
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf3Dread(T *res, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  *res = surf3Dread<T>(surf, x, y, z, mode);
#endif /* __CUDA_ARCH__ */  
}



template <typename T>
static __DEPRECATED__ __device__  __forceinline__ void surf1DLayeredread(T *res, surface<void, cudaSurfaceType1DLayered> surf, int x, int  layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf1DLayeredread_v2", (void *)res, s, surf, x,  layer, mode);
#endif     
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__  T surf1DLayeredread(surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  T temp;
  __nv_tex_surf_handler("__surf1DLayeredread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, layer, mode);
  return temp;
#endif
}


template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf1DLayeredread(T *res, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  *res = surf1DLayeredread<T>(surf, x, layer, mode);
#endif /* __CUDA_ARCH__ */  
}


template <typename T>
static __DEPRECATED__ __device__  __forceinline__ void surf2DLayeredread(T *res, surface<void, cudaSurfaceType2DLayered> surf, int x,  int y, int  layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf2DLayeredread_v2", (void *)res, s, surf, x, y, layer, mode);
#endif     
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__  T surf2DLayeredread(surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  T temp;
  __nv_tex_surf_handler("__surf2DLayeredread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, layer, mode);
  return temp;
#endif
}


template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf2DLayeredread(T *res, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  *res = surf2DLayeredread<T>(surf, x, y, layer, mode);
#endif /* __CUDA_ARCH__ */  
}


template <typename T>
static __device__  __forceinline__ void surfCubemapread(T *res, surface<void, cudaSurfaceTypeCubemap> surf, int x,  int y, int  face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surfCubemapread_v2", (void *)res, s, surf, x, y, face, mode);
#endif     
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__  T surfCubemapread(surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  T temp;

  __nv_tex_surf_handler("__surfCubemapread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, face, mode);
  return temp;
#endif
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surfCubemapread(T *res, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__  
  *res = surfCubemapread<T>(surf, x, y, face, mode);
#endif /* __CUDA_ARCH__ */  
}


template <typename T>
static __DEPRECATED__ __device__  __forceinline__ void surfCubemapLayeredread(T *res, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x,  int y, int  layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surfCubemapLayeredread_v2", (void *)res, s, surf, x, y, layerFace, mode);
#endif     
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__  T surfCubemapLayeredread(surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  T temp;
  __nv_tex_surf_handler("__surfCubemapLayeredread_v2", (typename __nv_surf_trait<T>::cast_type)&temp, (int)sizeof(T), surf, x, y, layerFace, mode);
  return temp;
#endif
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surfCubemapLayeredread(T *res, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__  
  *res = surfCubemapLayeredread<T>(surf, x, y, layerFace, mode);
#endif /* __CUDA_ARCH__ */  
}

//surf1Dwrite
template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf1Dwrite(T val, surface<void, cudaSurfaceType1D> surf, int x, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf1Dwrite_v2", (void *)&val, s, surf, x, mode);
#endif  
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf1Dwrite(T val, surface<void, cudaSurfaceType1D> surf, int x, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  __nv_tex_surf_handler("__surf1Dwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x,  mode);
#endif /* __CUDA_ARCH__ */  
}


//surf2Dwrite
template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf2Dwrite_v2", (void *)&val,  s, surf, x, y, mode);
#endif  
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, int x, int y, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  __nv_tex_surf_handler("__surf2Dwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y,  mode);
#endif /* __CUDA_ARCH__ */  
}

//surf3Dwrite
template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf3Dwrite(T val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf3Dwrite_v2", (void *)&val,  s, surf, x, y, z,mode);
#endif  
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf3Dwrite(T val, surface<void, cudaSurfaceType3D> surf, int x, int y, int z, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  __nv_tex_surf_handler("__surf3Dwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, z,  mode);
#endif /* __CUDA_ARCH__ */  
}

//surf1DLayeredwrite
template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf1DLayeredwrite(T val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf1DLayeredwrite_v2", (void *)&val,  s, surf, x, layer,mode);
#endif  
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf1DLayeredwrite(T val, surface<void, cudaSurfaceType1DLayered> surf, int x, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  __nv_tex_surf_handler("__surf1DLayeredwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val,  (int)sizeof(T), surf, x, layer, mode);
#endif /* __CUDA_ARCH__ */  
}

//surf2DLayeredwrite
template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf2DLayeredwrite(T val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surf2DLayeredwrite_v2", (void *)&val, s, surf, x, y, layer,mode);
#endif  
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surf2DLayeredwrite(T val, surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  __nv_tex_surf_handler("__surf2DLayeredwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val,  (int)sizeof(T), surf, x, y, layer, mode);
#endif /* __CUDA_ARCH__ */  
}

//surfCubemapwrite
template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surfCubemapwrite(T val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surfCubemapwrite_v2", (void *)&val, s, surf, x, y, face, mode);
#endif  
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surfCubemapwrite(T val, surface<void, cudaSurfaceTypeCubemap> surf, int x, int y, int face, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  __nv_tex_surf_handler("__surfCubemapwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, face,  mode);
#endif /* __CUDA_ARCH__ */  
}


//surfCubemapLayeredwrite
template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surfCubemapLayeredwrite(T val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__surfCubemapLayeredwrite_v2", (void *)&val, s, surf, x, y, layerFace,  mode);
#endif  
}

template<class T>
static __DEPRECATED__ __device__ __forceinline__ void surfCubemapLayeredwrite(T val, surface<void, cudaSurfaceTypeCubemapLayered> surf, int x, int y, int layerFace, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__ 
  __nv_tex_surf_handler("__surfCubemapLayeredwrite_v2", (typename __nv_surf_trait<T>::cast_type)&val, (int)sizeof(T), surf, x, y, layerFace,  mode);
#endif /* __CUDA_ARCH__ */  
}

#undef __DEPRECATED__


#endif /* __cplusplus && __CUDACC__ */
#endif /* !__SURFACE_FUNCTIONS_H__ */
