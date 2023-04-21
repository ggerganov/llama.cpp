/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__MATH_CONSTANTS_H__)
#define __MATH_CONSTANTS_H__

/* single precision constants */
#define CUDART_INF_F            __int_as_float(0x7f800000U)
#define CUDART_NAN_F            __int_as_float(0x7fffffffU)
#define CUDART_MIN_DENORM_F     __int_as_float(0x00000001U)
#define CUDART_MAX_NORMAL_F     __int_as_float(0x7f7fffffU)
#define CUDART_NEG_ZERO_F       __int_as_float(0x80000000U)
#define CUDART_ZERO_F           0.0F
#define CUDART_ONE_F            1.0F
#define CUDART_SQRT_HALF_F      0.707106781F
#define CUDART_SQRT_HALF_HI_F   0.707106781F
#define CUDART_SQRT_HALF_LO_F   1.210161749e-08F
#define CUDART_SQRT_TWO_F       1.414213562F
#define CUDART_THIRD_F          0.333333333F
#define CUDART_PIO4_F           0.785398163F
#define CUDART_PIO2_F           1.570796327F
#define CUDART_3PIO4_F          2.356194490F
#define CUDART_2_OVER_PI_F      0.636619772F
#define CUDART_SQRT_2_OVER_PI_F 0.797884561F
#define CUDART_PI_F             3.141592654F
#define CUDART_L2E_F            1.442695041F
#define CUDART_L2T_F            3.321928094F
#define CUDART_LG2_F            0.301029996F
#define CUDART_LGE_F            0.434294482F
#define CUDART_LN2_F            0.693147181F
#define CUDART_LNT_F            2.302585093F
#define CUDART_LNPI_F           1.144729886F
#define CUDART_TWO_TO_M126_F    1.175494351e-38F
#define CUDART_TWO_TO_126_F     8.507059173e37F
#define CUDART_NORM_HUGE_F      3.402823466e38F
#define CUDART_TWO_TO_23_F      8388608.0F
#define CUDART_TWO_TO_24_F      16777216.0F
#define CUDART_TWO_TO_31_F      2147483648.0F
#define CUDART_TWO_TO_32_F      4294967296.0F
#define CUDART_REMQUO_BITS_F    3U
#define CUDART_REMQUO_MASK_F    (~((~0U)<<CUDART_REMQUO_BITS_F))
#define CUDART_TRIG_PLOSS_F     105615.0F

/* double precision constants */
#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_NAN              __longlong_as_double(0xfff8000000000000ULL)
#define CUDART_NEG_ZERO         __longlong_as_double(0x8000000000000000ULL)
#define CUDART_MIN_DENORM       __longlong_as_double(0x0000000000000001ULL)
#define CUDART_ZERO             0.0
#define CUDART_ONE              1.0
#define CUDART_SQRT_TWO         1.4142135623730951e+0
#define CUDART_SQRT_HALF        7.0710678118654757e-1
#define CUDART_SQRT_HALF_HI     7.0710678118654757e-1
#define CUDART_SQRT_HALF_LO   (-4.8336466567264567e-17)
#define CUDART_THIRD            3.3333333333333333e-1
#define CUDART_TWOTHIRD         6.6666666666666667e-1
#define CUDART_PIO4             7.8539816339744828e-1
#define CUDART_PIO4_HI          7.8539816339744828e-1
#define CUDART_PIO4_LO          3.0616169978683830e-17
#define CUDART_PIO2             1.5707963267948966e+0
#define CUDART_PIO2_HI          1.5707963267948966e+0
#define CUDART_PIO2_LO          6.1232339957367660e-17
#define CUDART_3PIO4            2.3561944901923448e+0
#define CUDART_2_OVER_PI        6.3661977236758138e-1
#define CUDART_PI               3.1415926535897931e+0
#define CUDART_PI_HI            3.1415926535897931e+0
#define CUDART_PI_LO            1.2246467991473532e-16
#define CUDART_SQRT_2PI         2.5066282746310007e+0
#define CUDART_SQRT_2PI_HI      2.5066282746310007e+0
#define CUDART_SQRT_2PI_LO    (-1.8328579980459167e-16)
#define CUDART_SQRT_PIO2        1.2533141373155003e+0
#define CUDART_SQRT_PIO2_HI     1.2533141373155003e+0
#define CUDART_SQRT_PIO2_LO   (-9.1642899902295834e-17)
#define CUDART_SQRT_2OPI        7.9788456080286536e-1
#define CUDART_L2E              1.4426950408889634e+0
#define CUDART_L2E_HI           1.4426950408889634e+0
#define CUDART_L2E_LO           2.0355273740931033e-17
#define CUDART_L2T              3.3219280948873622e+0
#define CUDART_LG2              3.0102999566398120e-1
#define CUDART_LG2_HI           3.0102999566398120e-1
#define CUDART_LG2_LO         (-2.8037281277851704e-18)
#define CUDART_LGE              4.3429448190325182e-1
#define CUDART_LGE_HI           4.3429448190325182e-1
#define CUDART_LGE_LO           1.09831965021676510e-17
#define CUDART_LN2              6.9314718055994529e-1
#define CUDART_LN2_HI           6.9314718055994529e-1
#define CUDART_LN2_LO           2.3190468138462996e-17
#define CUDART_LNT              2.3025850929940459e+0
#define CUDART_LNT_HI           2.3025850929940459e+0
#define CUDART_LNT_LO         (-2.1707562233822494e-16)
#define CUDART_LNPI             1.1447298858494002e+0
#define CUDART_LN2_X_1024       7.0978271289338397e+2
#define CUDART_LN2_X_1025       7.1047586007394398e+2
#define CUDART_LN2_X_1075       7.4513321910194122e+2
#define CUDART_LG2_X_1024       3.0825471555991675e+2
#define CUDART_LG2_X_1075       3.2360724533877976e+2
#define CUDART_TWO_TO_23        8388608.0
#define CUDART_TWO_TO_52        4503599627370496.0
#define CUDART_TWO_TO_53        9007199254740992.0
#define CUDART_TWO_TO_54        18014398509481984.0
#define CUDART_TWO_TO_M54       5.5511151231257827e-17
#define CUDART_TWO_TO_M1022     2.22507385850720140e-308
#define CUDART_TRIG_PLOSS       2147483648.0
#define CUDART_DBL2INT_CVT      6755399441055744.0

#endif /* !__MATH_CONSTANTS_H__ */
