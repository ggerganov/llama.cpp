echo "modify the ggml-sycl.cpp to fix dpct result error"
TARGET_FILE=ggml-sycl.cpp
sed -i "s/CUDA_CHECK(id = dpct::dev_mgr::instance().current_device_id());/id = dpct::dev_mgr::instance().current_device_id();/g" ${TARGET_FILE}
sed -i "s/CUDA_CHECK(current_device = dpct::dev_mgr::instance().current_device_id());/current_device = dpct::dev_mgr::instance().current_device_id();/g" ${TARGET_FILE}
sed -i "s/g_cublas_handles, oneapi::mkl::transpose::trans,/*g_cublas_handles[id], oneapi::mkl::transpose::trans,/g" ${TARGET_FILE}
sed -i "s/cu_compute_type = CUBLAS_COMPUTE_16F;/cu_compute_type = dpct::library_data_t::real_half;/g" ${TARGET_FILE}
sed -i "s/cu_compute_type = CUBLAS_COMPUTE_32F;/cu_compute_type = dpct::library_data_t::real_float;/g" ${TARGET_FILE}
sed -i "s/tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs, stream_ct1);/tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);/g" ${TARGET_FILE}
sed -i "s/cuGetErrorString(err, &err_str);/\/\/cuGetErrorString(err, &err_str);/g" ${TARGET_FILE}

#set empty function
#ggml_cuda_set_peer_access
#ggml_cuda_pool_malloc_vmm
#ggml_cuda_pool_free_vmm

#replace tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs, stream_ct1);/tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);

echo "done"