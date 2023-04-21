/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
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


#ifndef __TEXTURE_INDIRECT_FUNCTIONS_H__
#define __TEXTURE_INDIRECT_FUNCTIONS_H__


#if defined(__cplusplus) && defined(__CUDACC__)

#include "cuda_runtime_api.h"


#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 600)
#define __NV_TEX_SPARSE 1
#endif  /* endif */

template <typename T> struct __nv_itex_trait {   };
template<> struct __nv_itex_trait<char> { typedef void type; };
template<> struct __nv_itex_trait<signed char> { typedef void type; };
template<> struct __nv_itex_trait<char1> { typedef void type; };
template<> struct __nv_itex_trait<char2> { typedef void type; };
template<> struct __nv_itex_trait<char4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned char> { typedef void type; };
template<> struct __nv_itex_trait<uchar1> { typedef void type; };
template<> struct __nv_itex_trait<uchar2> { typedef void type; };
template<> struct __nv_itex_trait<uchar4> { typedef void type; };
template<> struct __nv_itex_trait<short> { typedef void type; };
template<> struct __nv_itex_trait<short1> { typedef void type; };
template<> struct __nv_itex_trait<short2> { typedef void type; };
template<> struct __nv_itex_trait<short4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned short> { typedef void type; };
template<> struct __nv_itex_trait<ushort1> { typedef void type; };
template<> struct __nv_itex_trait<ushort2> { typedef void type; };
template<> struct __nv_itex_trait<ushort4> { typedef void type; };
template<> struct __nv_itex_trait<int> { typedef void type; };
template<> struct __nv_itex_trait<int1> { typedef void type; };
template<> struct __nv_itex_trait<int2> { typedef void type; };
template<> struct __nv_itex_trait<int4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned int> { typedef void type; };
template<> struct __nv_itex_trait<uint1> { typedef void type; };
template<> struct __nv_itex_trait<uint2> { typedef void type; };
template<> struct __nv_itex_trait<uint4> { typedef void type; };
#if !defined(__LP64__)
template<> struct __nv_itex_trait<long> { typedef void type; };
template<> struct __nv_itex_trait<long1> { typedef void type; };
template<> struct __nv_itex_trait<long2> { typedef void type; };
template<> struct __nv_itex_trait<long4> { typedef void type; };
template<> struct __nv_itex_trait<unsigned long> { typedef void type; };
template<> struct __nv_itex_trait<ulong1> { typedef void type; };
template<> struct __nv_itex_trait<ulong2> { typedef void type; };
template<> struct __nv_itex_trait<ulong4> { typedef void type; };
#endif /* !__LP64__ */
template<> struct __nv_itex_trait<float> { typedef void type; };
template<> struct __nv_itex_trait<float1> { typedef void type; };
template<> struct __nv_itex_trait<float2> { typedef void type; };
template<> struct __nv_itex_trait<float4> { typedef void type; };



template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x)
{
#ifdef __CUDA_ARCH__
   __nv_tex_surf_handler("__itex1Dfetch", ptr, obj, x);
#endif   
}

template <class T>
static __device__ T tex1Dfetch(cudaTextureObject_t texObject, int x)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex1Dfetch(&ret, texObject, x);
  return ret;
#endif  
}

template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex1D(T *ptr, cudaTextureObject_t obj, float x)
{
#ifdef __CUDA_ARCH__
   __nv_tex_surf_handler("__itex1D", ptr, obj, x);
#endif
}


template <class T>
static __device__  T tex1D(cudaTextureObject_t texObject, float x)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex1D(&ret, texObject, x);
  return ret;
#endif
}


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2D(T *ptr, cudaTextureObject_t obj, float x, float y)
{
#ifdef __CUDA_ARCH__
   __nv_tex_surf_handler("__itex2D", ptr, obj, x, y);
#endif
}

template <class T>
static __device__  T tex2D(cudaTextureObject_t texObject, float x, float y)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2D(&ret, texObject, x, y);
  return ret;
#endif  
}

#if __NV_TEX_SPARSE
template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2D(T *ptr, cudaTextureObject_t obj, float x, float y, 
                                                          bool* isResident)
{
#ifdef __CUDA_ARCH__
  unsigned char res;
   __nv_tex_surf_handler("__itex2D_sparse", ptr, obj, x, y, &res);
   *isResident = (res != 0);
#endif
}

template <class T>
static __device__  T tex2D(cudaTextureObject_t texObject, float x, float y, bool* isResident)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2D(&ret, texObject, x, y, isResident);
  return ret;
#endif  
}

#endif  /* __NV_TEX_SPARSE */


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z)
{
#ifdef __CUDA_ARCH__
   __nv_tex_surf_handler("__itex3D", ptr, obj, x, y, z);
#endif
}

template <class T>
static __device__  T tex3D(cudaTextureObject_t texObject, float x, float y, float z)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex3D(&ret, texObject, x, y, z);
  return ret;
#endif
}

#if __NV_TEX_SPARSE
template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z, 
                                                          bool* isResident)
{
#ifdef __CUDA_ARCH__
  unsigned char res;
   __nv_tex_surf_handler("__itex3D_sparse", ptr, obj, x, y, z, &res);
   *isResident = (res != 0);
#endif
}

template <class T>
static __device__  T tex3D(cudaTextureObject_t texObject, float x, float y, float z, bool* isResident)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex3D(&ret, texObject, x, y, z, isResident);
  return ret;
#endif
}
#endif  /* __NV_TEX_SPARSE */


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer)
{
#ifdef __CUDA_ARCH__
   __nv_tex_surf_handler("__itex1DLayered", ptr, obj, x, layer);
#endif
}

template <class T>
static __device__  T tex1DLayered(cudaTextureObject_t texObject, float x, int layer)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex1DLayered(&ret, texObject, x, layer);
  return ret;
#endif
}

template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex2DLayered", ptr, obj, x, y, layer);
#endif
}

template <class T>
static __device__  T tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DLayered(&ret, texObject, x, y, layer);
  return ret;
#endif
}

#if __NV_TEX_SPARSE
template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, bool* isResident)
{
#ifdef __CUDA_ARCH__
  unsigned char res;
  __nv_tex_surf_handler("__itex2DLayered_sparse", ptr, obj, x, y, layer, &res);
  *isResident = (res != 0);
#endif
}

template <class T>
static __device__  T tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer, bool* isResident)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DLayered(&ret, texObject, x, y, layer, isResident);
  return ret;
#endif
}
#endif  /* __NV_TEX_SPARSE */


template <typename T>
static __device__ typename __nv_itex_trait<T>::type texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itexCubemap", ptr, obj, x, y, z);
#endif
}


template <class T>
static __device__  T texCubemap(cudaTextureObject_t texObject, float x, float y, float z)
{
#ifdef __CUDA_ARCH__
  T ret;
  texCubemap(&ret, texObject, x, y, z);
  return ret;
#endif
}


template <typename T>
static __device__ typename __nv_itex_trait<T>::type texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itexCubemapLayered", ptr, obj, x, y, z, layer);
#endif
}

template <class T>
static __device__  T texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer)
{
#ifdef __CUDA_ARCH__
  T ret;
  texCubemapLayered(&ret, texObject, x, y, z, layer);
  return ret;
#endif  
}

template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex2Dgather", ptr, obj, x, y, comp);
#endif
}

template <class T>
static __device__  T tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2Dgather(&ret, to, x, y, comp);
  return ret;
#endif
}

#if __NV_TEX_SPARSE
template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, bool* isResident, int comp = 0)
{
#ifdef __CUDA_ARCH__
  unsigned char res;
  __nv_tex_surf_handler("__itex2Dgather_sparse", ptr, obj, x, y, comp,  &res);
  *isResident = (res != 0);
#endif
}

template <class T>
static __device__  T tex2Dgather(cudaTextureObject_t to, float x, float y, bool* isResident, int comp = 0)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2Dgather(&ret, to, x, y,  isResident, comp);
  return ret;
#endif
}

#endif  /* __NV_TEX_SPARSE */

template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex1DLod", ptr, obj, x, level);
#endif
}

template <class T>
static __device__  T tex1DLod(cudaTextureObject_t texObject, float x, float level)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex1DLod(&ret, texObject, x, level);
  return ret;
#endif  
}


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex2DLod", ptr, obj, x, y, level);
#endif
}

template <class T>
static __device__  T tex2DLod(cudaTextureObject_t texObject, float x, float y, float level)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DLod(&ret, texObject, x, y, level);
  return ret;
#endif  
}

#if __NV_TEX_SPARSE

template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level, bool* isResident)
{
#ifdef __CUDA_ARCH__
  unsigned char res;
  __nv_tex_surf_handler("__itex2DLod_sparse", ptr, obj, x, y, level, &res);
  *isResident = (res != 0);
#endif
}

template <class T>
static __device__  T tex2DLod(cudaTextureObject_t texObject, float x, float y, float level, bool* isResident)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DLod(&ret, texObject, x, y, level, isResident);
  return ret;
#endif  
}

#endif  /* __NV_TEX_SPARSE */


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex3DLod", ptr, obj, x, y, z, level);
#endif
}

template <class T>
static __device__  T tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex3DLod(&ret, texObject, x, y, z, level);
  return ret;
#endif  
}

#if __NV_TEX_SPARSE
template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level, bool* isResident)
{ 
#ifdef __CUDA_ARCH__
  unsigned char res;
  __nv_tex_surf_handler("__itex3DLod_sparse", ptr, obj, x, y, z, level, &res);
  *isResident = (res != 0);
#endif
}

template <class T>
static __device__  T tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level, bool* isResident)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex3DLod(&ret, texObject, x, y, z, level, isResident);
  return ret;
#endif  
}

#endif  /* __NV_TEX_SPARSE */


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex1DLayeredLod", ptr, obj, x, layer, level);
#endif
}

template <class T>
static __device__  T tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex1DLayeredLod(&ret, texObject, x, layer, level);
  return ret;
#endif  
}


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex2DLayeredLod", ptr, obj, x, y, layer, level);
#endif
}

template <class T>
static __device__  T tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DLayeredLod(&ret, texObject, x, y, layer, level);
  return ret;
#endif  
}

#if __NV_TEX_SPARSE
template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level, bool* isResident)
{ 
#ifdef __CUDA_ARCH__
  unsigned char res;
  __nv_tex_surf_handler("__itex2DLayeredLod_sparse", ptr, obj, x, y, layer, level, &res);
  *isResident = (res != 0);
#endif
}

template <class T>
static __device__  T tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level, bool* isResident)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DLayeredLod(&ret, texObject, x, y, layer, level, isResident);
  return ret;
#endif  
}
#endif  /* __NV_TEX_SPARSE */

template <typename T>
static __device__ typename __nv_itex_trait<T>::type texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itexCubemapLod", ptr, obj, x, y, z, level);
#endif
}

template <class T>
static __device__  T texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level)
{
#ifdef __CUDA_ARCH__
  T ret;
  texCubemapLod(&ret, texObject, x, y, z, level);
  return ret;
#endif  
}


template <typename T>
static __device__ typename __nv_itex_trait<T>::type texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itexCubemapGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy);
#endif
}

template <class T>
static __device__  T texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
#ifdef __CUDA_ARCH__
  T ret;
  texCubemapGrad(&ret, texObject, x, y, z, dPdx, dPdy);
  return ret;
#endif  
}

template <typename T>
static __device__ typename __nv_itex_trait<T>::type texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itexCubemapLayeredLod", ptr, obj, x, y, z, layer, level);
#endif
}

template <class T>
static __device__  T texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level)
{
#ifdef __CUDA_ARCH__
  T ret;
  texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level);
  return ret;
#endif  
}

template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex1DGrad", ptr, obj, x, dPdx, dPdy);
#endif
}

template <class T>
static __device__  T tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex1DGrad(&ret, texObject, x, dPdx, dPdy);
  return ret;
#endif  
}


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex2DGrad_v2", ptr, obj, x, y, &dPdx, &dPdy);
#endif

}

template <class T>
static __device__  T tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DGrad(&ret, texObject, x, y, dPdx, dPdy);
  return ret;
#endif  
}

#if __NV_TEX_SPARSE
template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy, bool* isResident)
{ 
#ifdef __CUDA_ARCH__
  unsigned char res;
  __nv_tex_surf_handler("__itex2DGrad_sparse", ptr, obj, x, y, &dPdx, &dPdy, &res);
  *isResident = (res != 0);
#endif

}

template <class T>
static __device__  T tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy, bool* isResident)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DGrad(&ret, texObject, x, y, dPdx, dPdy, isResident);
  return ret;
#endif  
}
#endif  /* __NV_TEX_SPARSE */


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex3DGrad_v2", ptr, obj, x, y, z, &dPdx, &dPdy);
#endif
}

template <class T>
static __device__  T tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy);
  return ret;
#endif  
}

#if __NV_TEX_SPARSE
template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy, bool* isResident)
{ 
#ifdef __CUDA_ARCH__
  unsigned char res;
  __nv_tex_surf_handler("__itex3DGrad_sparse", ptr, obj, x, y, z, &dPdx, &dPdy, &res);
  *isResident = (res != 0);
#endif
}

template <class T>
static __device__  T tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy, bool* isResident)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy, isResident);
  return ret;
#endif  
}

#endif  /* __NV_TEX_SPARSE */


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex1DLayeredGrad", ptr, obj, x, layer, dPdx, dPdy);
#endif
}

template <class T>
static __device__  T tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy);
  return ret;
#endif  
}


template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DLayeredGrad(T * ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy)
{ 
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itex2DLayeredGrad_v2", ptr, obj, x, y, layer, &dPdx, &dPdy);
#endif
}

template <class T>
static __device__  T tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy);
  return ret;
#endif  
}

#if __NV_TEX_SPARSE
template <typename T>
static __device__ typename __nv_itex_trait<T>::type tex2DLayeredGrad(T * ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy, bool* isResident)
{ 
#ifdef __CUDA_ARCH__
  unsigned char res;
  __nv_tex_surf_handler("__itex2DLayeredGrad_sparse", ptr, obj, x, y, layer, &dPdx, &dPdy, &res);
  *isResident = (res != 0);
#endif
}

template <class T>
static __device__  T tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy, bool* isResident)
{
#ifdef __CUDA_ARCH__
  T ret;
  tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy, isResident);
  return ret;
#endif  
}
#endif  /* __NV_TEX_SPARSE */


template <typename T>
static __device__ typename __nv_itex_trait<T>::type texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{
#ifdef __CUDA_ARCH__
  __nv_tex_surf_handler("__itexCubemapLayeredGrad_v2", ptr, obj, x, y, z, layer, &dPdx, &dPdy);
#endif
}

template <class T>
static __device__  T texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{
#ifdef __CUDA_ARCH__
  T ret;
  texCubemapLayeredGrad(&ret, texObject, x, y, z, layer, dPdx, dPdy);
  return ret;
#endif  
}

#undef __NV_TEX_SPARSE

#endif // __cplusplus && __CUDACC__
#endif // __TEXTURE_INDIRECT_FUNCTIONS_H__
