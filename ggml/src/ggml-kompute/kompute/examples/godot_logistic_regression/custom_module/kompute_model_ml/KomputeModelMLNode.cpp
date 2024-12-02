/* summator.cpp */

#include <vector>

#include "KomputeModelMLNode.h"

KomputeModelMLNode::KomputeModelMLNode()
{
    std::cout << "CALLING CONSTRUCTOR" << std::endl;
    this->_init();
}

void
KomputeModelMLNode::train(Array yArr, Array xIArr, Array xJArr)
{

    assert(yArr.size() == xIArr.size());
    assert(xIArr.size() == xJArr.size());

    std::vector<float> yData;
    std::vector<float> xIData;
    std::vector<float> xJData;
    std::vector<float> zerosData;

    for (size_t i = 0; i < yArr.size(); i++) {
        yData.push_back(yArr[i]);
        xIData.push_back(xIArr[i]);
        xJData.push_back(xJArr[i]);
        zerosData.push_back(0);
    }

    uint32_t ITERATIONS = 100;
    float learningRate = 0.1;

    {
        kp::Manager mgr;

        std::shared_ptr<kp::Tensor> xI = mgr.tensor(xIData);
        std::shared_ptr<kp::Tensor> xJ = mgr.tensor(xJData);

        std::shared_ptr<kp::Tensor> y = mgr.tensor(yData);

        std::shared_ptr<kp::Tensor> wIn = mgr.tensor({ 0.001, 0.001 });
        std::shared_ptr<kp::Tensor> wOutI = mgr.tensor(zerosData);
        std::shared_ptr<kp::Tensor> wOutJ = mgr.tensor(zerosData);

        std::shared_ptr<kp::Tensor> bIn = mgr.tensor({ 0 });
        std::shared_ptr<kp::Tensor> bOut = mgr.tensor(zerosData);

        std::shared_ptr<kp::Tensor> lOut = mgr.tensor(zerosData);

        std::vector<std::shared_ptr<kp::Tensor>> params = { xI,  xJ,    y,
                                                            wIn, wOutI, wOutJ,
                                                            bIn, bOut,  lOut };

        {
            std::vector<uint32_t> spirv(
              (uint32_t*)
                kp::shader_data::shaders_glsl_logisticregression_comp_spv,
              (uint32_t*)(kp::shader_data::
                            shaders_glsl_logisticregression_comp_spv +
                          kp::shader_data::
                            shaders_glsl_logisticregression_comp_spv_len));

            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv);

            mgr.sequence()->eval<kp::OpTensorSyncDevice>(params);

            std::shared_ptr<kp::Sequence> sq =
              mgr.sequence()
                ->record<kp::OpTensorSyncDevice>({ wIn, bIn })
                ->record<kp::OpAlgoDispatch>(algo)
                ->record<kp::OpTensorSyncLocal>({ wOutI, wOutJ, bOut, lOut });

            // Iterate across all expected iterations
            for (size_t i = 0; i < ITERATIONS; i++) {

                sq->eval();

                for (size_t j = 0; j < bOut->size(); j++) {
                    wIn->data()[0] -= learningRate * wOutI->data()[j];
                    wIn->data()[1] -= learningRate * wOutJ->data()[j];
                    bIn->data()[0] -= learningRate * bOut->data()[j];
                }
            }
        }

        KP_LOG_INFO("RESULT: <<<<<<<<<<<<<<<<<<<");
        KP_LOG_INFO(wIn->data()[0]);
        KP_LOG_INFO(wIn->data()[1]);
        KP_LOG_INFO(bIn->data()[0]);

        this->mWeights = kp::Tensor(wIn->data());
        this->mBias = kp::Tensor(bIn->data());
    }
}

Array
KomputeModelMLNode::predict(Array xI, Array xJ)
{
    assert(xI.size() == xJ.size());

    Array retArray;

    // We run the inference in the CPU for simplicity
    // BUt you can also implement the inference on GPU
    // GPU implementation would speed up minibatching
    for (size_t i = 0; i < xI.size(); i++) {
        float xIVal = xI[i];
        float xJVal = xJ[i];
        float result =
          (xIVal * this->mWeights.data()[0] + xJVal * this->mWeights.data()[1] +
           this->mBias.data()[0]);

        // Instead of using sigmoid we'll just return full numbers
        Variant var = result > 0 ? 1 : 0;
        retArray.push_back(var);
    }

    return retArray;
}

Array
KomputeModelMLNode::get_params()
{
    Array retArray;

    KP_LOG_INFO(this->mWeights.size() + this->mBias.size());

    if (this->mWeights.size() + this->mBias.size() == 0) {
        return retArray;
    }

    retArray.push_back(this->mWeights.data()[0]);
    retArray.push_back(this->mWeights.data()[1]);
    retArray.push_back(this->mBias.data()[0]);
    retArray.push_back(99.0);

    return retArray;
}

void
KomputeModelMLNode::_init()
{
    std::cout << "CALLING INIT" << std::endl;
}

void
KomputeModelMLNode::_process(float delta)
{}

void
KomputeModelMLNode::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("_process", "delta"),
                         &KomputeModelMLNode::_process);
    ClassDB::bind_method(D_METHOD("_init"), &KomputeModelMLNode::_init);

    ClassDB::bind_method(D_METHOD("train", "yArr", "xIArr", "xJArr"),
                         &KomputeModelMLNode::train);
    ClassDB::bind_method(D_METHOD("predict", "xI", "xJ"),
                         &KomputeModelMLNode::predict);
    ClassDB::bind_method(D_METHOD("get_params"),
                         &KomputeModelMLNode::get_params);
}
