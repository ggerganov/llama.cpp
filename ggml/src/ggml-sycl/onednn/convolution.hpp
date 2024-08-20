//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_ONEDNN_CONV_HPP
#define GGML_SYCL_ONEDNN_CONV_HPP

#include <fstream>
#include <iostream>

#include "ggml-sycl.h"

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

class DnnlConvWrapper {
public:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    struct conv_params {
        int s0;
        int s1;
        int p0;
        int p1;
        int d0;
        int d1;
        bool is_2d;
    };

    template<typename T>
    static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) return dt::f32;
        else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
        else static_assert(0);
    }

    static inline void forward(const dnnl::stream& stream,
        int n, int h, int w, int ic, int oc, int kh, int kw,
        conv_params& params,
        const void* a, dt at, const void* b, dt bt, void* c, dt ct)
    {
        auto const eng = stream.get_engine();
        dnnl::memory::dims a_dims, b_dims, c_dims;
        dnnl::memory::desc a_md, b_md, c_md, bias_md;
        dnnl::primitive_attr pattr;

        if(params.is_2d) {
            a_dims = { n, ic, h, w };
            b_dims = { oc, ic, kh, kw };
            c_dims = { n, oc, h, w };
            a_md = dnnl::memory::desc(a_dims, at, tag::nchw);
            b_md = dnnl::memory::desc(b_dims, bt, tag::oihw);
            c_md = dnnl::memory::desc(c_dims, ct, tag::nchw);
        } else {
            a_dims = { n, ic, h };
            b_dims = { oc, ic, kh };
            c_dims = { n, oc, h };
            a_md = dnnl::memory::desc(a_dims, at, tag::ncw);
            b_md = dnnl::memory::desc(b_dims, bt, tag::oiw);
            c_md = dnnl::memory::desc(c_dims, ct, tag::ncw);
        }

        auto a_mem = dnnl::memory(a_md, eng, (void*)a);
        auto b_mem = dnnl::memory(b_md, eng, (void*)b);

        // Create the primitive.
        auto   conv_fwd_pd = dnnl::convolution_forward::primitive_desc(
            eng,
            dnnl::prop_kind::forward,
            dnnl::algorithm::convolution_direct,
            a_md,
            b_md,
            bias_md,
            c_md,
            {params.s0, params.s1},
            {params.d0, params.d1},
            {params.p0, params.p1},
            {params.p0, params.p1},
            pattr);
        auto conv_fwd = dnnl::convolution_forward(conv_fwd_pd);
        auto c_mem = dnnl::memory(conv_fwd_pd.dst_desc(), eng, c);
        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> conv_args;
        conv_args.insert({ DNNL_ARG_SRC, a_mem });
        conv_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
        conv_args.insert({ DNNL_ARG_DST, c_mem });

        conv_fwd.execute(stream, conv_args);
    }
};

#endif

#endif // GGML_SYCL_ONEDNN_CONV_HPP
