#import "mtl.h"

#import "ggml.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

struct ggml_mtl_context {
    struct ggml_context * ctx_data;
    struct ggml_context * ctx_eval;
    struct ggml_context * ctx_work;

    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    id<MTLBuffer> buffer_data;
    id<MTLBuffer> buffer_eval;

    id<MTLBuffer> out;

    // custom kernels
    id<MTLFunction>             function_add;
    id<MTLComputePipelineState> pipeline_add;

    id<MTLFunction>             function_mul;
    id<MTLComputePipelineState> pipeline_mul;

    id<MTLFunction>             function_relu;
    id<MTLComputePipelineState> pipeline_relu;

    id<MTLFunction>             function_soft_max;
    id<MTLComputePipelineState> pipeline_soft_max;

    id<MTLFunction>             function_get_rows_q4_0;
    id<MTLComputePipelineState> pipeline_get_rows_q4_0;

    id<MTLFunction>             function_rms_norm;
    id<MTLComputePipelineState> pipeline_rms_norm;

    id<MTLFunction>             function_mul_mat_q4_0;
    id<MTLComputePipelineState> pipeline_mul_mat_q4_0;
};

// MSL code
// TODO: move the contents here when ready
//       for now it is easier to work in a separate file
NSString * const msl_library_llama = @"see mtl.metal";

struct ggml_mtl_context * llama_mtl_init(
    struct ggml_context * ctx_data,
    struct ggml_context * ctx_eval,
    struct ggml_context * ctx_work,
    struct ggml_cgraph  * gf) {
    fprintf(stderr, "%s: allocating\n", __func__);

    struct ggml_mtl_context * ctx = malloc(sizeof(struct ggml_mtl_context));

    ctx->ctx_data = ctx_data;
    ctx->ctx_eval = ctx_eval;
    ctx->ctx_work = ctx_work;

    ctx->device = MTLCreateSystemDefaultDevice();
    ctx->queue  = [ctx->device newCommandQueue];

    // determine if we can use MPS
    if (MPSSupportsMTLDevice(ctx->device)) {
        fprintf(stderr, "%s: using MPS\n", __func__);
    } else {
        fprintf(stderr, "%s: not using MPS\n", __func__);
        GGML_ASSERT(false && "MPS not supported");
    }

#if 0
    // compile from source string and show compile log
    {
        NSError * error = nil;

        ctx->library = [ctx->device newLibraryWithSource:msl_library_llama options:nil error:&error];
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }
    }
#elif 0
    // this does not work !?!?!

    // load library from "mtl.metallib"
    {
        NSError * error = nil;

        NSString * path = [[NSBundle mainBundle] pathForResource:@"./mtl" ofType:@"metallib"];
        ctx->library = [ctx->device newLibraryWithFile:path error:&error];
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }
    }
#else
    // read the source from "../examples/mtl/mtl.metal" into a string and use newLibraryWithSource
    {
        NSError * error = nil;

        NSString * path = [[NSBundle mainBundle] pathForResource:@"../../examples/mtl/mtl" ofType:@"metal"];
        NSString * src  = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }

        ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&error];
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }
    }
#endif

    // load kernels
    {
        MTLFunctionConstantValues * constants = [MTLFunctionConstantValues new];

        ctx->function_add = [ctx->library newFunctionWithName:@"kernel_add"];
        ctx->pipeline_add = [ctx->device newComputePipelineStateWithFunction:ctx->function_add error:nil];
        fprintf(stderr, "%s: loaded kernel_add: %p\n", __func__, (void *) ctx->pipeline_add);

        ctx->function_mul = [ctx->library newFunctionWithName:@"kernel_mul"];
        ctx->pipeline_mul = [ctx->device newComputePipelineStateWithFunction:ctx->function_mul error:nil];
        fprintf(stderr, "%s: loaded kernel_mul: %p\n", __func__, (void *) ctx->pipeline_mul);

        ctx->function_relu = [ctx->library newFunctionWithName:@"kernel_relu"];
        ctx->pipeline_relu = [ctx->device newComputePipelineStateWithFunction:ctx->function_relu error:nil];
        fprintf(stderr, "%s: loaded kernel_relu: %p\n", __func__, (void *) ctx->pipeline_relu);

        ctx->function_soft_max = [ctx->library newFunctionWithName:@"kernel_soft_max" constantValues:constants error:nil];
        ctx->pipeline_soft_max = [ctx->device newComputePipelineStateWithFunction:ctx->function_soft_max error:nil];
        fprintf(stderr, "%s: loaded kernel_soft_max: %p\n", __func__, (void *) ctx->pipeline_soft_max);

        ctx->function_get_rows_q4_0 = [ctx->library newFunctionWithName:@"kernel_get_rows_q4_0"];
        ctx->pipeline_get_rows_q4_0 = [ctx->device newComputePipelineStateWithFunction:ctx->function_get_rows_q4_0 error:nil];
        fprintf(stderr, "%s: loaded kernel_get_rows_q4_0: %p\n", __func__, (void *) ctx->pipeline_get_rows_q4_0);

        ctx->function_rms_norm = [ctx->library newFunctionWithName:@"kernel_rms_norm"];
        ctx->pipeline_rms_norm = [ctx->device newComputePipelineStateWithFunction:ctx->function_rms_norm error:nil];
        fprintf(stderr, "%s: loaded kernel_rms_norm: %p\n", __func__, (void *) ctx->pipeline_rms_norm);

        ctx->function_mul_mat_q4_0 = [ctx->library newFunctionWithName:@"kernel_mul_mat_q4_0"];
        ctx->pipeline_mul_mat_q4_0 = [ctx->device newComputePipelineStateWithFunction:ctx->function_mul_mat_q4_0 error:nil];
        fprintf(stderr, "%s: loaded kernel_mul_mat_q4_0: %p\n", __func__, (void *) ctx->pipeline_mul_mat_q4_0);
    }

    // MTLBuffer approach

    // pin ctx_data memory to GPU
    // use MTLStorageModeShared to allow us to initialize the weights from the CPU
    // TODO: how to use MTLStorageModeManaged?
    // TODO: see if we can avoid this copy somehow
    {
        const void * mem_buffer = ggml_get_mem_buffer(ctx_data);
        const size_t mem_size   = ggml_get_mem_size(ctx_data);

        ctx->buffer_data = [ctx->device newBufferWithBytes:mem_buffer length:mem_size options:MTLResourceStorageModeShared];

        fprintf(stderr, "%s: allocated data buffer, size = %8.2f MB\n", __func__, mem_size / 1024.0 / 1024.0);
    }

    // pin ctx_eval memory to GPU
    // this buffer will be used for the intermediate results of the evaluation
    {
        const size_t mem_size = ggml_get_mem_size(ctx_eval);

        ctx->buffer_eval = [ctx->device newBufferWithLength:mem_size options:MTLResourceStorageModePrivate];

        fprintf(stderr, "%s: allocated eval buffer, size = %8.2f MB\n", __func__, mem_size / 1024.0 / 1024.0);
    }

    // allocate buffer for result extraction
    {
        const size_t mem_size = ggml_nbytes(gf->nodes[gf->n_nodes - 1]);

        ctx->out = [ctx->device newBufferWithLength:mem_size options:MTLResourceStorageModeShared];

        fprintf(stderr, "%s: allocated  out buffer, size = %8.2f MB\n", __func__, mem_size / 1024.0 / 1024.0);
    }

    return ctx;
}

void llama_mtl_free(struct ggml_mtl_context * ctx) {
    fprintf(stderr, "%s: deallocating\n", __func__);

    free(ctx);
}

// get data / eval buffer + offset
id<MTLBuffer> llama_mtl_get_buffer(struct ggml_mtl_context * ctx, struct ggml_tensor * t, size_t * offs) {
    const int64_t offs_data = (int64_t) t->data - (int64_t) ggml_get_mem_buffer(ctx->ctx_data);
    const int64_t offs_eval = (int64_t) t->data - (int64_t) ggml_get_mem_buffer(ctx->ctx_eval);

    const bool is_data = (offs_eval < 0) || (offs_data >= 0 && offs_data < offs_eval);

    const size_t t_size = ggml_nbytes(t);
    const size_t t_offs = is_data ? offs_data : offs_eval;

    id<MTLBuffer> result;

    if (is_data) {
        fprintf(stderr, "%s: data tensor '%16s', offs = %8ld, size = %8ld\n", __func__, t->name, t_offs, t_size);
        result = ctx->buffer_data;
    } else {
        fprintf(stderr, "%s: eval tensor '%16s', offs = %8ld, size = %8ld\n", __func__, t->name, t_offs, t_size);
        result = ctx->buffer_eval;
    }

    if (result == nil) {
        fprintf(stderr, "%s: error: buffer is nil\n", __func__);
        GGML_ASSERT(false);
    }

    if (offs != nil) {
        *offs = t_offs;
    }

    return result;
}

int llama_mtl_eval(
        struct ggml_mtl_context * ctx,
        struct ggml_cgraph      * gf) {
    fprintf(stderr, "%s: evaluating\n", __func__);

    id<MTLCommandBuffer> command_buffer  = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = nil;

    size_t offs_src0;
    size_t offs_src1;
    size_t offs_dst;

    // copy the input data to the GPU
    {
        struct ggml_tensor * embd = ggml_graph_get_tensor(gf, "embd");

        id<MTLBuffer> id_dst = llama_mtl_get_buffer(ctx, embd, &offs_src0);

        memcpy((char *) id_dst.contents + offs_src0, embd->data, ggml_nbytes(embd));
    }

    for (int i = 0; i < gf->n_nodes; ++i) {
        fprintf(stderr, "%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

        switch (gf->nodes[i]->op) {
            case GGML_OP_ADD:
                {
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoder];
                    }

                    id<MTLBuffer> id_src0 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src0, &offs_src0);
                    id<MTLBuffer> id_src1 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src1, &offs_src1);
                    id<MTLBuffer> id_dst  = llama_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    [encoder setComputePipelineState:ctx->pipeline_add];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                    const int64_t n = ggml_nelements(gf->nodes[i]);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
            case GGML_OP_MUL:
                {
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoder];
                    }

                    id<MTLBuffer> id_src0 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src0, &offs_src0);
                    id<MTLBuffer> id_src1 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src1, &offs_src1);
                    id<MTLBuffer> id_dst  = llama_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    const int64_t ne00 = gf->nodes[i]->src0->ne[0];

                    [encoder setComputePipelineState:ctx->pipeline_mul];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                    [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];

                    const int64_t n = ggml_nelements(gf->nodes[i]);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
            case GGML_OP_RELU:
                {
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoder];
                    }

                    id<MTLBuffer> id_src = llama_mtl_get_buffer(ctx, gf->nodes[i]->src0, &offs_src0);
                    id<MTLBuffer> id_dst = llama_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    [encoder setComputePipelineState:ctx->pipeline_relu];
                    [encoder setBuffer:id_src offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(gf->nodes[i]);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
            case GGML_OP_SOFT_MAX:
                {
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoder];
                    }

                    id<MTLBuffer> id_src = llama_mtl_get_buffer(ctx, gf->nodes[i]->src0, &offs_src0);
                    id<MTLBuffer> id_dst = llama_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    [encoder setComputePipelineState:ctx->pipeline_soft_max];
                    [encoder setBuffer:id_src offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst offset:offs_dst  atIndex:1];

                    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
            case GGML_OP_MUL_MAT:
                if (gf->nodes[i]->src0->type == GGML_TYPE_F32) {
                    // for F32 x F32 we use MPS

                    if (encoder != nil) {
                        [encoder endEncoding];
                        encoder = nil;
                    }

                    // use MPSMatrixMultiplication
                    id<MTLBuffer> id_src0 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src0, &offs_src0);
                    id<MTLBuffer> id_src1 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src1, &offs_src1);
                    id<MTLBuffer> id_dst  = llama_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    const int64_t ncols0 = gf->nodes[i]->src0->ne[0];
                    const int64_t nrows0 = gf->nodes[i]->src0->ne[1];

                    const int64_t ncols1 = gf->nodes[i]->src1->ne[0];
                    const int64_t nrows1 = gf->nodes[i]->src1->ne[1];

                    const int64_t ncols2 = gf->nodes[i]->ne[0];
                    const int64_t nrows2 = gf->nodes[i]->ne[1];

                    GGML_ASSERT(ncols0 == ncols1);

                    MPSMatrixDescriptor * desc0 = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:nrows0 columns:ncols0 rowBytes:gf->nodes[i]->src0->nb[1] dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor * desc1 = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:nrows1 columns:ncols1 rowBytes:gf->nodes[i]->src1->nb[1] dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor * desc2 = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:nrows2 columns:ncols2 rowBytes:gf->nodes[i]->nb[1] dataType:MPSDataTypeFloat32];

                    MPSMatrix * mat_src0 = [[MPSMatrix alloc] initWithBuffer:id_src0 offset:offs_src0 descriptor:desc0];
                    MPSMatrix * mat_src1 = [[MPSMatrix alloc] initWithBuffer:id_src1 offset:offs_src1 descriptor:desc1];
                    MPSMatrix * mat_dst  = [[MPSMatrix alloc] initWithBuffer:id_dst  offset:offs_dst  descriptor:desc2];

                    MPSMatrixMultiplication * mul = [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
                        transposeLeft:false transposeRight:true resultRows:nrows1 resultColumns:nrows0 interiorColumns:ncols0 alpha:1.0 beta:0.0];

                    [mul encodeToCommandBuffer:command_buffer leftMatrix:mat_src1 rightMatrix:mat_src0 resultMatrix:mat_dst];
                } else {
                    // for Q4 x F32 we use custom kernel

                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoder];
                    }

                    GGML_ASSERT(gf->nodes[i]->src0->ne[2] == 1);
                    GGML_ASSERT(gf->nodes[i]->src1->ne[2] == 1);

                    id<MTLBuffer> id_src0 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src0, &offs_src0);
                    id<MTLBuffer> id_src1 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src1, &offs_src1);
                    id<MTLBuffer> id_dst  = llama_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    const int64_t ne00 = gf->nodes[i]->src0->ne[0];
                    const int64_t ne01 = gf->nodes[i]->src0->ne[1];
                    const int64_t ne10 = gf->nodes[i]->src1->ne[0];
                    const int64_t ne11 = gf->nodes[i]->src1->ne[1];
                    const int64_t ne0  = gf->nodes[i]->ne[0];
                    const int64_t ne1  = gf->nodes[i]->ne[1];

                    [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_0];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                    [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];
                    [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:4];
                    [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:5];
                    [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:6];
                    [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:7];
                    [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:8];

                    printf("mul_mat: %lldx%lld * %lldx%lld -> %lldx%lld\n", ne00, ne01, ne10, ne11, ne0, ne1);

                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne11, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                } break;
            case GGML_OP_GET_ROWS:
                {
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoder];
                    }

                    id<MTLBuffer> id_src0 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src0, &offs_src0);
                    id<MTLBuffer> id_src1 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src1, &offs_src1);
                    id<MTLBuffer> id_dst  = llama_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    switch (gf->nodes[i]->src0->type) {
                        case GGML_TYPE_Q4_0: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_0]; break;
                        default: {
                                     // not implemented
                                     fprintf(stderr, "%s: node %3d, op = %8s, type = %8s not implemented\n", __func__, i, ggml_op_name(gf->nodes[i]->op), ggml_type_name(gf->nodes[i]->src0->type));
                                 }
                    }

                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                    [encoder setBytes:&(gf->nodes[i]->src0->ne[0]) length:sizeof( int64_t) atIndex:3];
                    [encoder setBytes:&(gf->nodes[i]->src0->nb[1]) length:sizeof(uint64_t) atIndex:4];
                    [encoder setBytes:&(gf->nodes[i]->nb[1])       length:sizeof(uint64_t) atIndex:5];

                    const int64_t n = ggml_nelements(gf->nodes[i]->src1);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
            case GGML_OP_RMS_NORM:
                {
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoder];
                    }

                    id<MTLBuffer> id_src0 = llama_mtl_get_buffer(ctx, gf->nodes[i]->src0, &offs_src0);
                    id<MTLBuffer> id_dst  = llama_mtl_get_buffer(ctx, gf->nodes[i],       &offs_dst);

                    const  int64_t ne00 = gf->nodes[i]->src0->ne[0];
                    const uint64_t nb01 = gf->nodes[i]->src0->nb[1];
                    const    float eps  = 1e-6f;

                    [encoder setComputePipelineState:ctx->pipeline_rms_norm];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                    [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                    [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:3];
                    [encoder setBytes:&eps  length:sizeof(  float)  atIndex:4];

                    const int64_t nrows = ggml_nrows(gf->nodes[i]->src0);

                    [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
            default:
                fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(gf->nodes[i]->op));
                GGML_ASSERT(false);
                return -1;
        }
    }

    // extract results from the GPU
    {
        fprintf(stderr, "%s: extract results from the GPU\n", __func__);

        if (encoder != nil) {
            [encoder endEncoding];
            encoder = nil;
        }

        struct ggml_tensor * out = gf->nodes[gf->n_nodes - 1];

        id<MTLBuffer> id_src = llama_mtl_get_buffer(ctx, out, &offs_src0);
        id<MTLBuffer> id_dst = ctx->out;

        printf("XXXXX n = %d\n", ggml_nelements(out));

        id<MTLBlitCommandEncoder> encoder_blit = [command_buffer blitCommandEncoder];
        [encoder_blit copyFromBuffer:id_src sourceOffset:offs_src0 toBuffer:id_dst destinationOffset:0 size:ggml_nbytes(out)];
        [encoder_blit endEncoding];
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    {
        const double time_elapsed = [command_buffer GPUEndTime] - [command_buffer GPUStartTime];
        fprintf(stderr, "%s: time elapsed = %f\n", __func__, time_elapsed);
    }

    // TODO
    const float * logits = ctx->out.contents;

    {
        struct ggml_tensor * t = ggml_get_tensor(ctx->ctx_eval, "mtl-check");
        float * data = (float *) ctx->out.contents;
        printf("data: ");
        int n = t->ne[0];
        if (n > 10) {
            n = 10;
        }
        for (int i = 0; i < n; i++) {
            printf("%f ", data[i]);
        }
        printf("\n");
        double sum = 0.0;
        for (int i = 0; i < ggml_nelements(t); i++) {
            sum += data[i];
        }
        printf("sum:  %f\n", sum);
    }

    return 0;
}
