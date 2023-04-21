 /* Copyright 2010-2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved. 
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
#ifndef NV_NPPS_SUPPORT_FUNCTIONS_H
#define NV_NPPS_SUPPORT_FUNCTIONS_H
 
/**
 * \file npps_support_functions.h
 * Signal Processing Support Functions.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** @defgroup signal_memory_management Memory Management
 *  @ingroup npps
 * Functions that provide memory management functionality like malloc and free.
 * @{
 */

/** @defgroup signal_malloc Malloc
 * Signal-allocator methods for allocating 1D arrays of data in device memory.
 * All allocators have size parameters to specify the size of the signal (1D array)
 * being allocated.
 *
 * The allocator methods return a pointer to the newly allocated memory of appropriate
 * type. If device-memory allocation is not possible due to resource constaints
 * the allocators return 0 (i.e. NULL pointer). 
 *
 * All signal allocators allocate memory aligned such that it is  beneficial to the 
 * performance of the majority of the signal-processing primitives. 
 * It is no mandatory however to use these allocators. Any valid
 * CUDA device-memory pointers can be passed to NPP primitives. 
 *
 * @{
 */

/**
 * 8-bit unsigned signal allocator.
 * \param nSize Number of unsigned chars in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp8u * 
nppsMalloc_8u(int nSize);

/**
 * 8-bit signed signal allocator.
 * \param nSize Number of (signed) chars in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp8s * 
nppsMalloc_8s(int nSize);

/**
 * 16-bit unsigned signal allocator.
 * \param nSize Number of unsigned shorts in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp16u * 
nppsMalloc_16u(int nSize);

/**
 * 16-bit signal allocator.
 * \param nSize Number of shorts in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp16s * 
nppsMalloc_16s(int nSize);

/**
 * 16-bit complex-value signal allocator.
 * \param nSize Number of 16-bit complex numbers in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp16sc * 
nppsMalloc_16sc(int nSize);

/**
 * 32-bit unsigned signal allocator.
 * \param nSize Number of unsigned ints in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp32u * 
nppsMalloc_32u(int nSize);

/**
 * 32-bit integer signal allocator.
 * \param nSize Number of ints in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp32s * 
nppsMalloc_32s(int nSize);

/**
 * 32-bit complex integer signal allocator.
 * \param nSize Number of complex integner values in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp32sc * 
nppsMalloc_32sc(int nSize);

/**
 * 32-bit float signal allocator.
 * \param nSize Number of floats in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp32f * 
nppsMalloc_32f(int nSize);

/**
 * 32-bit complex float signal allocator.
 * \param nSize Number of complex float values in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp32fc * 
nppsMalloc_32fc(int nSize);

/**
 * 64-bit long integer signal allocator.
 * \param nSize Number of long ints in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp64s * 
nppsMalloc_64s(int nSize);

/**
 * 64-bit complex long integer signal allocator.
 * \param nSize Number of complex long int values in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp64sc * 
nppsMalloc_64sc(int nSize);

/**
 * 64-bit float (double) signal allocator.
 * \param nSize Number of doubles in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp64f * 
nppsMalloc_64f(int nSize);

/**
 * 64-bit complex complex signal allocator.
 * \param nSize Number of complex double valuess in the new signal.
 * \return A pointer to the new signal. 0 (NULL-pointer) indicates
 *         that an error occurred during allocation.
 */
Npp64fc * 
nppsMalloc_64fc(int nSize);

/** @} signal_malloc */

/** @defgroup signal_free Free
 * Free  signal memory.
 *
 * @{
 */
 
/**
 * Free method for any signal memory.
 * \param pValues A pointer to memory allocated using nppiMalloc_<modifier>.
 */
void nppsFree(void * pValues);

/** @} signal_free */

/** end of Memory management functions
 * 
 * @}
 *
 */



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPS_SUPPORT_FUNCTIONS_H */
