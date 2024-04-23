/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <memory>
#include <string>

struct DirectStorageCUDAFileHandle {
    virtual ~DirectStorageCUDAFileHandle() {};
};

class InteropBuffer {
public:
  virtual ~InteropBuffer() = 0;
  virtual void* get_device_ptr() const = 0;
  virtual void* get_host_ptr() const = 0;
};

class DirectStorageCUDA
{
public:
    virtual ~DirectStorageCUDA();

    using File = std::unique_ptr<DirectStorageCUDAFileHandle>;

    virtual std::unique_ptr<InteropBuffer> create_interop_buffer(size_t size) = 0;

    virtual File openFile(std::string const& filename) = 0;
    virtual void loadFile(File const& file, size_t read_start, size_t read_len, void* cuda_dst_ptr) = 0;
    virtual void loadFile(File const& file, size_t read_start, size_t read_len, InteropBuffer *interop_buffer, size_t interop_buffer_offset) = 0;
    virtual void flush(bool last = false) = 0;

    static std::unique_ptr<DirectStorageCUDA> create(int scratch_size, int number_of_scratch_spaces);
};

