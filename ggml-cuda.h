#ifdef  __cplusplus
extern "C" {
#endif

void dequantize_row_q4_0_cuda(const void * vx, float * y, int k);
void dequantize_row_q4_1_cuda(const void * vx, float * y, int k);
void dequantize_row_q4_2_cuda(const void * vx, float * y, int k);

#ifdef  __cplusplus
}
#endif
