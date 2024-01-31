//
//  MIT license
//  Copyright (C) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT
//


#include "ggml-sycl.h"

int main(int argc, char ** argv) {
    ggml_backend_sycl_print_sycl_devices();
    return 0;
}
