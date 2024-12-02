import os

import kp

def compile_source(source):
    os.system("glslangValidator --stdin -S comp -V -o tmp_kp_shader.comp.spv << END\n" + source + "\nEND")
    return open("tmp_kp_shader.comp.spv", "rb").read()


# This is the convolution & leakyrelu shader.
global conv_shader
conv_shader = compile_source("""
#version 450

layout (local_size_x = 8, local_size_y = 2) in;

// [y][x][group] (vec4: channels)
layout (set = 0, binding = 0) buffer buf_in_image { readonly restrict vec4 in_image[]; };
// [outputCGroups] (vec4: output channels)
layout (set = 0, binding = 1) buffer buf_in_bias { readonly restrict vec4 in_bias[]; };
// [outputCGroups][kernelH][kernelW][inputCGroups] (mat4: input & output channels)
layout (set = 0, binding = 2) buffer buf_in_weight { readonly restrict mat4 in_weight[]; };
// [y][x][group] (vec4: channels)
layout (set = 0, binding = 3) buffer buf_out_image { writeonly restrict vec4 out_image[]; };

// The 'c' measures in cgroups.
// Some maths changes as a result.
layout (constant_id = 0) const float in_w = 0;
layout (constant_id = 1) const float in_h = 0;
layout (constant_id = 2) const float in_cg = 0;
layout (constant_id = 3) const float out_w = 0;
layout (constant_id = 4) const float out_h = 0;
layout (constant_id = 5) const float out_cg = 0;

uint index_in_no_ic(uvec2 pos) {
    return (pos.x + (pos.y * uint(in_w))) * uint(in_cg);
}

uint index_out(uvec2 pos) {
    return ((pos.x + (pos.y * uint(out_w))) * uint(out_cg)) + gl_GlobalInvocationID.z;
}

void main() {
    // out x/y is gl_GlobalInvocationID.xy
    // we need to account for workgroupy padding *here*
    // so long as we aren't trying to output to a pixel that doesn't exist,
    //  we won't read from any pixels that don't exist
    if (
        (gl_GlobalInvocationID.x < (uint(in_w) - 2)) &&
        (gl_GlobalInvocationID.y < (uint(in_h) - 2))
    ) {
        vec4 value = in_bias[gl_GlobalInvocationID.z];
        for (uint x = 0; x < 3; x++) {
            for (uint y = 0; y < 3; y++) {
                uint weight_ptr = ((gl_GlobalInvocationID.z * 9) + (x + (y * 3))) * uint(in_cg);
                // specific pixel
                // important to note is that since in position has a border around it,
                // no further transformation is necessary (the - is implied)
                uvec2 in_pos = gl_GlobalInvocationID.xy + uvec2(x, y);
                uint in_ptr = index_in_no_ic(in_pos);
                for (uint icg = 0; icg < uint(in_cg); icg++) {
                    // input channel group
                    vec4 iCG = in_image[in_ptr];
                    // handle all 4 input components
                    value += iCG * in_weight[weight_ptr];
                    weight_ptr += 1;
                    in_ptr += 1;
                }
            }
        }
        // leakyrelu slope 0.1
        value = (max(value, 0.0) * 0.9) + (value * 0.1);
        out_image[index_out(gl_GlobalInvocationID.xy)] = value;
    }
}
""")

