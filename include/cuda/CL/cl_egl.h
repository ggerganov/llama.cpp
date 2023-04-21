/*******************************************************************************
 * Copyright (c) 2008-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef __OPENCL_CL_EGL_H
#define __OPENCL_CL_EGL_H

#ifdef __APPLE__
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


/* Command type for events created with clEnqueueAcquireEGLObjectsKHR */
#define CL_COMMAND_EGL_FENCE_SYNC_OBJECT_KHR  0x202F
#define CL_COMMAND_ACQUIRE_EGL_OBJECTS_KHR    0x202D
#define CL_COMMAND_RELEASE_EGL_OBJECTS_KHR    0x202E

/* Error type for clCreateFromEGLImageKHR */
#define CL_INVALID_EGL_OBJECT_KHR             -1093
#define CL_EGL_RESOURCE_NOT_ACQUIRED_KHR      -1092

/* CLeglImageKHR is an opaque handle to an EGLImage */
typedef void* CLeglImageKHR;

/* CLeglDisplayKHR is an opaque handle to an EGLDisplay */
typedef void* CLeglDisplayKHR;

/* CLeglSyncKHR is an opaque handle to an EGLSync object */
typedef void* CLeglSyncKHR;

/* properties passed to clCreateFromEGLImageKHR */
typedef intptr_t cl_egl_image_properties_khr;


#define cl_khr_egl_image 1

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromEGLImageKHR(cl_context                  context,
                        CLeglDisplayKHR             egldisplay,
                        CLeglImageKHR               eglimage,
                        cl_mem_flags                flags,
                        const cl_egl_image_properties_khr * properties,
                        cl_int *                    errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromEGLImageKHR_fn)(
    cl_context                  context,
    CLeglDisplayKHR             egldisplay,
    CLeglImageKHR               eglimage,
    cl_mem_flags                flags,
    const cl_egl_image_properties_khr * properties,
    cl_int *                    errcode_ret);


extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAcquireEGLObjectsKHR(cl_command_queue command_queue,
                              cl_uint          num_objects,
                              const cl_mem *   mem_objects,
                              cl_uint          num_events_in_wait_list,
                              const cl_event * event_wait_list,
                              cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueAcquireEGLObjectsKHR_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event);


extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReleaseEGLObjectsKHR(cl_command_queue command_queue,
                              cl_uint          num_objects,
                              const cl_mem *   mem_objects,
                              cl_uint          num_events_in_wait_list,
                              const cl_event * event_wait_list,
                              cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueReleaseEGLObjectsKHR_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event);


#define cl_khr_egl_event 1

extern CL_API_ENTRY cl_event CL_API_CALL
clCreateEventFromEGLSyncKHR(cl_context      context,
                            CLeglSyncKHR    sync,
                            CLeglDisplayKHR display,
                            cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_event (CL_API_CALL *clCreateEventFromEGLSyncKHR_fn)(
    cl_context      context,
    CLeglSyncKHR    sync,
    CLeglDisplayKHR display,
    cl_int *        errcode_ret);

#ifdef __cplusplus
}
#endif

#endif /* __OPENCL_CL_EGL_H */
