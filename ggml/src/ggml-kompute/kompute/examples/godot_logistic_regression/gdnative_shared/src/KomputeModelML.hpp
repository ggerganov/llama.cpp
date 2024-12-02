#pragma once

#include <Array.hpp>
#include <Godot.hpp>
#include <Node2D.hpp>

#include <memory>

#include "kompute/Kompute.hpp"

namespace godot {
class KomputeModelML : public Node2D
{
  private:
    GODOT_CLASS(KomputeModelML, Node2D);

  public:
    KomputeModelML();

    void train(Array y, Array xI, Array xJ);

    Array predict(Array xI, Array xJ);

    Array get_params();

    void _process(float delta);
    void _init();

    static void _register_methods();

  private:
    std::shared_ptr<kp::Tensor> mWeights;
    std::shared_ptr<kp::Tensor> mBias;
};

static std::string LR_SHADER = R"(
#version 450

layout (constant_id = 0) const uint M = 0;

layout (local_size_x = 1) in;

layout(set = 0, binding = 0) buffer bxi { float xi[]; };
layout(set = 0, binding = 1) buffer bxj { float xj[]; };
layout(set = 0, binding = 2) buffer by { float y[]; };
layout(set = 0, binding = 3) buffer bwin { float win[]; };
layout(set = 0, binding = 4) buffer bwouti { float wouti[]; };
layout(set = 0, binding = 5) buffer bwoutj { float woutj[]; };
layout(set = 0, binding = 6) buffer bbin { float bin[]; };
layout(set = 0, binding = 7) buffer bbout { float bout[]; };
layout(set = 0, binding = 8) buffer blout { float lout[]; };

float m = float(M);

float sigmoid(float z) {
    return 1.0 / (1.0 + exp(-z));
}

float inference(vec2 x, vec2 w, float b) {
    // Compute the linear mapping function
    float z = dot(w, x) + b;
    // Calculate the y-hat with sigmoid 
    float yHat = sigmoid(z);
    return yHat;
}

float calculateLoss(float yHat, float y) {
    return -(y * log(yHat)  +  (1.0 - y) * log(1.0 - yHat));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;

    vec2 wCurr = vec2(win[0], win[1]);
    float bCurr = bin[0];

    vec2 xCurr = vec2(xi[idx], xj[idx]);
    float yCurr = y[idx];

    float yHat = inference(xCurr, wCurr, bCurr);

    float dZ = yHat - yCurr;
    vec2 dW = (1. / m) * xCurr * dZ;
    float dB = (1. / m) * dZ;
    wouti[idx] = dW.x;
    woutj[idx] = dW.y;
    bout[idx] = dB;

    lout[idx] = calculateLoss(yHat, yCurr);
}
)";

}
