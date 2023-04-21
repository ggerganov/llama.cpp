/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2010-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#ifndef fatbinary_section_INCLUDED
#define fatbinary_section_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/*
 * These defines are for the fatbin.c runtime wrapper
 */
#define FATBINC_MAGIC   0x466243B1
#define FATBINC_VERSION 1
#define FATBINC_LINK_VERSION 2

typedef struct {
  int magic;
  int version;
  const unsigned long long* data;
  void *filename_or_fatbins;  /* version 1: offline filename,
                               * version 2: array of prelinked fatbins */
} __fatBinC_Wrapper_t;

/*
 * The section that contains the fatbin control structure
 */
#ifdef STD_OS_Darwin
/* mach-o sections limited to 15 chars, and want __ prefix else strip complains, * so use a different name */
#define FATBIN_CONTROL_SECTION_NAME     "__fatbin"
#define FATBIN_DATA_SECTION_NAME        "__nv_fatbin"
/* only need segment name for mach-o */
#define FATBIN_SEGMENT_NAME             "__NV_CUDA"
#else
#define FATBIN_CONTROL_SECTION_NAME     ".nvFatBinSegment"
/*
 * The section that contains the fatbin data itself
 * (put in separate section so easy to find)
 */
#define FATBIN_DATA_SECTION_NAME        ".nv_fatbin"
#endif
/* section for pre-linked relocatable fatbin data */
#define FATBIN_PRELINK_DATA_SECTION_NAME "__nv_relfatbin"

#ifdef __cplusplus
}
#endif

#endif /* fatbinary_section_INCLUDED */
