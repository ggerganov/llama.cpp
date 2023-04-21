/* Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
 *                   
 * NOTICE TO LICENSEE:
 *
 * The source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * The Licensed Deliverables contained herein are PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and are being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
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
 * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
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
/*
   Copyright 2010-2011, D. E. Shaw Research.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions, and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions, and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 * Neither the name of D. E. Shaw Research nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CURAND_PHILOX4X32_X__H_
#define CURAND_PHILOX4X32_X__H_

#if !defined(QUALIFIERS)
#define QUALIFIERS static __forceinline__ __device__
#endif

#define PHILOX_W32_0   (0x9E3779B9)
#define PHILOX_W32_1   (0xBB67AE85)
#define PHILOX_M4x32_0 (0xD2511F53)
#define PHILOX_M4x32_1 (0xCD9E8D57)

struct curandStatePhilox4_32_10 {
   uint4 ctr;
   uint4 output;
   uint2 key;
   unsigned int STATE;
   int boxmuller_flag;
   int boxmuller_flag_double;
   float boxmuller_extra;
   double boxmuller_extra_double;
};

typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;


QUALIFIERS void Philox_State_Incr(curandStatePhilox4_32_10_t* s, unsigned long long n)
{
   unsigned int nlo = (unsigned int)(n);
   unsigned int nhi = (unsigned int)(n>>32);

   s->ctr.x += nlo;
   if( s->ctr.x < nlo )
      nhi++;

   s->ctr.y += nhi;
   if(nhi <= s->ctr.y)
      return;
   if(++s->ctr.z) return;
   ++s->ctr.w;
}

QUALIFIERS void Philox_State_Incr_hi(curandStatePhilox4_32_10_t* s, unsigned long long n)
{
   unsigned int nlo = (unsigned int)(n);
   unsigned int nhi = (unsigned int)(n>>32);

   s->ctr.z += nlo;
   if( s->ctr.z < nlo )
      nhi++;

   s->ctr.w += nhi;
}



QUALIFIERS void Philox_State_Incr(curandStatePhilox4_32_10_t* s)
{
   if(++s->ctr.x) return;
   if(++s->ctr.y) return;
   if(++s->ctr.z) return;
   ++s->ctr.w;
}


QUALIFIERS unsigned int mulhilo32(unsigned int a, unsigned int b, unsigned int* hip)
{
#ifndef __CUDA_ARCH__
   // host code
   unsigned long long product = ((unsigned long long)a) * ((unsigned long long)b);
   *hip = product >> 32;
   return (unsigned int)product;
#else
   // device code
   *hip = __umulhi(a,b);
   return a*b;
#endif
}

QUALIFIERS uint4 _philox4x32round(uint4 ctr, uint2 key)
{
   unsigned int hi0;
   unsigned int hi1;
   unsigned int lo0 = mulhilo32(PHILOX_M4x32_0, ctr.x, &hi0);
   unsigned int lo1 = mulhilo32(PHILOX_M4x32_1, ctr.z, &hi1);

   uint4 ret  = {hi1^ctr.y^key.x, lo1, hi0^ctr.w^key.y, lo0};
   return ret;
}

QUALIFIERS uint4 curand_Philox4x32_10( uint4 c, uint2 k)
{
   c = _philox4x32round(c, k);                           // 1 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 2
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 3 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 4 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 5 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 6 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 7 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 8 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 9 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   return _philox4x32round(c, k);                        // 10
}


#endif
