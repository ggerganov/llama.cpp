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


#ifndef __SURFACE_INDIRECT_FUNCTIONS_H__
#define __SURFACE_INDIRECT_FUNCTIONS_H__


#if defined(__cplusplus) && defined(__CUDACC__)

#include "cuda_runtime_api.h"

template<typename T> struct __nv_isurf_trait { };
template<> struct __nv_isurf_trait<char> { typedef void type; };
template<> struct __nv_isurf_trait<signed char> { typedef void type; };
template<> struct __nv_isurf_trait<char1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned char> { typedef void type; };
template<> struct __nv_isurf_trait<uchar1> { typedef void type; };
template<> struct __nv_isurf_trait<short> { typedef void type; };
template<> struct __nv_isurf_trait<short1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned short> { typedef void type; };
template<> struct __nv_isurf_trait<ushort1> { typedef void type; };
template<> struct __nv_isurf_trait<int> { typedef void type; };
template<> struct __nv_isurf_trait<int1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned int> { typedef void type; };
template<> struct __nv_isurf_trait<uint1> { typedef void type; };
template<> struct __nv_isurf_trait<long long> { typedef void type; };
template<> struct __nv_isurf_trait<longlong1> { typedef void type; };
template<> struct __nv_isurf_trait<unsigned long long> { typedef void type; };
template<> struct __nv_isurf_trait<ulonglong1> { typedef void type; };
template<> struct __nv_isurf_trait<float> { typedef void type; };
template<> struct __nv_isurf_trait<float1> { typedef void type; };

template<> struct __nv_isurf_trait<char2> { typedef void type; };
template<> struct __nv_isurf_trait<uchar2> { typedef void type; };
template<> struct __nv_isurf_trait<short2> { typedef void type; };
template<> struct __nv_isurf_trait<ushort2> { typedef void type; };
template<> struct __nv_isurf_trait<int2> { typedef void type; };
template<> struct __nv_isurf_trait<uint2> { typedef void type; };
template<> struct __nv_isurf_trait<longlong2> { typedef void type; };
template<> struct __nv_isurf_trait<ulonglong2> { typedef void type; };
template<> struct __nv_isurf_trait<float2> { typedef void type; };

template<> struct __nv_isurf_trait<char4> { typedef void type; };
template<> struct __nv_isurf_trait<uchar4> { typedef void type; };
template<> struct __nv_isurf_trait<short4> { typedef void type; };
template<> struct __nv_isurf_trait<ushort4> { typedef void type; };
template<> struct __nv_isurf_trait<int4> { typedef void type; };
template<> struct __nv_isurf_trait<uint4> { typedef void type; };
template<> struct __nv_isurf_trait<float4> { typedef void type; };


template <typename T>
static __device__ typename __nv_isurf_trait<T>::type  surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf1Dread", ptr, obj, x, mode);
#endif /* __CUDA_ARCH__ */
}

template <class T>
static __device__ T surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__  
   T ret;
   surf1Dread(&ret, surfObject, x, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

template <typename T>
static __device__ typename __nv_isurf_trait<T>::type  surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf2Dread", ptr, obj, x, y, mode);
#endif /* __CUDA_ARCH__ */
}

template <class T>
static __device__ T surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surf2Dread(&ret, surfObject, x, y, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}


template <typename T>
static __device__ typename  __nv_isurf_trait<T>::type  surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf3Dread", ptr, obj, x, y, z, mode);
#endif /* __CUDA_ARCH__ */
}

template <class T>
static __device__ T surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surf3Dread(&ret, surfObject, x, y, z, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

template <typename T>
static __device__ typename  __nv_isurf_trait<T>::type  surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf1DLayeredread", ptr, obj, x, layer, mode);
#endif /* __CUDA_ARCH__ */
}

template <class T>
static __device__ T surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surf1DLayeredread(&ret, surfObject, x, layer, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

template <typename T>
static __device__  typename __nv_isurf_trait<T>::type  surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf2DLayeredread", ptr, obj, x, y, layer, mode);
#endif /* __CUDA_ARCH__ */
}

template <class T>
static __device__ T surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surf2DLayeredread(&ret, surfObject, x, y, layer, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

template <typename T>
static __device__ typename __nv_isurf_trait<T>::type  surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurfCubemapread", ptr, obj, x, y, face, mode);
#endif /* __CUDA_ARCH__ */
}

template <class T>
static __device__ T surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surfCubemapread(&ret, surfObject, x, y, face, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

template <typename T>
static __device__  typename __nv_isurf_trait<T>::type  surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurfCubemapLayeredread", ptr, obj, x, y, layerface, mode);
#endif /* __CUDA_ARCH__ */
}

template <class T>
static __device__ T surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__   
   T ret;
   surfCubemapLayeredread(&ret, surfObject, x, y, layerface, boundaryMode);
   return ret;
#endif /* __CUDA_ARCH__ */   
}

template <typename T>
static __device__ typename __nv_isurf_trait<T>::type surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf1Dwrite_v2", &val, obj, x, mode);
#endif /* __CUDA_ARCH__ */  
}

template <typename T>
static __device__ typename __nv_isurf_trait<T>::type surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf2Dwrite_v2", &val, obj, x, y, mode);
#endif /* __CUDA_ARCH__ */ 
}

template <typename T>
static __device__ typename __nv_isurf_trait<T>::type surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf3Dwrite_v2", &val, obj, x, y, z, mode);
#endif /* __CUDA_ARCH__ */
}

template <typename T>
static __device__ typename __nv_isurf_trait<T>::type surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf1DLayeredwrite_v2", &val, obj, x, layer, mode);
#endif /* __CUDA_ARCH__ */
}

template <typename T>
static __device__ typename __nv_isurf_trait<T>::type surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurf2DLayeredwrite_v2", &val, obj, x, y, layer, mode);
#endif /* __CUDA_ARCH__ */
}

template <typename T>
static __device__ typename __nv_isurf_trait<T>::type surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurfCubemapwrite_v2", &val, obj, x, y, face, mode);
#endif /* __CUDA_ARCH__ */
}

template <typename T>
static __device__ typename __nv_isurf_trait<T>::type surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__isurfCubemapLayeredwrite_v2", &val, obj, x, y, layerface, mode);
#endif /* __CUDA_ARCH__ */
}

#endif // __cplusplus && __CUDACC__

#endif // __SURFACE_INDIRECT_FUNCTIONS_H__


