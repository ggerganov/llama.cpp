/* Shared by both pinned engines. Byte-based parsing is intentional: GGUF blocks
 * can start at any byte alignment. No vector loads from original packed data. */
#include "quant.h"
#if !defined(__k1om__) && !defined(GGML_KNC_TEST_EMULATION)
#error Production glue requires native K1OM
#endif
#include <stdint.h>
#include <string.h>
#if (defined(TARGET_KNC_3120A) + defined(TARGET_KNC_5110P) + defined(TARGET_KNC_7120P)) != 1
#error Explicit card profile required
#endif
#define DECL(q) extern void ggml_knc_q##q##_block_native(const uint8_t*,const uint8_t*,const int8_t*,const int32_t*,const uint8_t*,const uint8_t*,size_t,int32_t*);
DECL(2) DECL(3) DECL(4) DECL(6)
struct scratch {
 uint8_t low[128] __attribute__((aligned(64)));
 uint8_t high[64] __attribute__((aligned(64)));
 int8_t y[256] __attribute__((aligned(64)));
 int32_t scales[16] __attribute__((aligned(64)));
 int32_t lanes[16] __attribute__((aligned(64)));
};
static __thread struct scratch w;
static float half(const uint8_t *p){
 uint32_t h=p[0]|((uint32_t)p[1]<<8),s=(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,b;
 if(e==0){if(!m)b=s;else{int shift=0;while(!(m&1024)){m<<=1;++shift;}b=s|((127-14-shift)<<23)|((m&1023)<<13);}}
 else b=s|((e==31?255:e+112)<<23)|(m<<13);
 float f;memcpy(&f,&b,4);return f;
}
static float packed_float(const uint8_t *p){
 uint32_t bits;float f;
#ifdef __k1om__
 /* Preserve the v0.2.1 unaligned scalar-load fix; memcpy alone is unsafe in GCC5. */
 __asm__("movl %1, %0":"=r"(bits):"m"(*(const unsigned char (*)[4])p));
#else
 memcpy(&bits,p,4);
#endif
 memcpy(&f,&bits,4);return f;
}
static int sum16(const uint8_t *p){uint16_t u=p[0]|((uint16_t)p[1]<<8);return (int16_t)u;}
int ggml_knc_dot_k(int q,int n,float *out,const void *vx,const void *vy){
 if(n<0||n%256||!out||(n&&(!vx||!vy)))return -1;
 int stride=q==2?84:q==3?110:q==4?144:q==6?210:0;if(!stride)return -1;
 const uint8_t *x=vx,*y=vy;float sum=0;
 for(int b=0;b<n/256;++b,x+=stride,y+=292){
  float d,dm=0;int correction=0;size_t remaining=n/256-b;
  memcpy(w.y,y+4,256);
  if(q==6){
   memcpy(w.low,x,128);memcpy(w.high,x+128,64);
   for(int j=0;j<16;++j)w.scales[j]=(int8_t)x[192+j];d=half(x+208);
   ggml_knc_q6_block_native(w.low,w.high,w.y,w.scales,x,y,remaining,w.lanes);
  }else if(q==4){
   memcpy(w.low,x+16,128);d=half(x);dm=half(x+2);
   for(int j=0;j<8;++j){int sc,m;if(j<4){sc=x[4+j]&63;m=x[8+j]&63;}else{sc=(x[8+j]&15)|((x[j]>>6)<<4);m=(x[8+j]>>4)|((x[4+j]>>6)<<4);}
    w.scales[2*j]=w.scales[2*j+1]=sc;correction+=m*(sum16(y+260+4*j)+sum16(y+262+4*j));}
   ggml_knc_q4_block_native(w.low,w.high,w.y,w.scales,x,y,remaining,w.lanes);
  }else if(q==3){
   memcpy(w.low,x+32,64);memcpy(w.high,x,32);d=half(x+108);
   for(int j=0;j<16;++j)w.scales[j]=((x[96+j%8]>>(j/8*4))&15)|(((x[104+j%4]>>(j/4*2))&3)<<4);
   for(int j=0;j<16;++j)w.scales[j]-=32;
   ggml_knc_q3_block_native(w.low,w.high,w.y,w.scales,x,y,remaining,w.lanes);
  }else{
   memcpy(w.low,x+16,64);d=half(x+80);dm=half(x+82);
   for(int j=0;j<16;++j){w.scales[j]=x[j]&15;correction+=(x[j]>>4)*sum16(y+260+2*j);}
   ggml_knc_q2_block_native(w.low,w.high,w.y,w.scales,x,y,remaining,w.lanes);
  }
  int32_t dot=0;for(int j=0;j<16;++j)dot+=w.lanes[j];
  float yd=packed_float(y);sum+=(d*yd)*dot-(dm*yd)*correction;
 }
 *out=sum;return 0;
}
