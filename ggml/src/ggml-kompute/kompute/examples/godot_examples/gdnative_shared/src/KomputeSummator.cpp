/* summator.cpp */

#include <iostream>
#include <vector>

#include "KomputeSummator.hpp"

static std::vector<uint32_t>
compileSource(const std::string& source)
{
    std::ofstream fileOut("tmp_kp_shader.comp");
    fileOut << source;
    fileOut.close();
    if (system(
          std::string(
            "glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv")
            .c_str()))
        throw std::runtime_error("Error running glslangValidator command");
    std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(
      buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return { (uint32_t*)buffer.data(),
             (uint32_t*)(buffer.data() + buffer.size()) };
}

namespace godot {

KomputeSummator::KomputeSummator()
{
    std::cout << "CALLING CONSTRUCTOR" << std::endl;
    this->_init();
}

void
KomputeSummator::add(float value)
{
    // Set the new data in the local device
    this->mSecondaryTensor->setData({ value });
    // Execute recorded sequence
    this->mSequence->eval();
}

void
KomputeSummator::reset()
{}

float
KomputeSummator::get_total() const
{
    return this->mPrimaryTensor->data()[0];
}

void
KomputeSummator::_init()
{
    std::cout << "CALLING INIT" << std::endl;
    this->mPrimaryTensor = this->mManager.tensor({ 0.0 });
    this->mSecondaryTensor = this->mManager.tensor({ 0.0 });
    this->mSequence = this->mManager.sequence("AdditionSeq");

    // We now record the steps in the sequence
    {
        std::string shader(R"(
            #version 450

            layout (local_size_x = 1) in;

            layout(set = 0, binding = 0) buffer a { float pa[]; };
            layout(set = 0, binding = 1) buffer b { float pb[]; };

            void main() {
                uint index = gl_GlobalInvocationID.x;
                pa[index] = pb[index] + pa[index];
            }
        )");

        this->mSequence->begin();

        // First we ensure secondary tensor loads to GPU
        // No need to sync the primary tensor as it should not be changed
        this->mSequence->record<kp::OpTensorSyncDevice>(
          { this->mSecondaryTensor });

        // Then we run the operation with both tensors
        this->mSequence->record<kp::OpAlgoCreate>(
          { this->mPrimaryTensor, this->mSecondaryTensor },
          compileSource(shader));

        // We map the result back to local
        this->mSequence->record<kp::OpTensorSyncLocal>(
          { this->mPrimaryTensor });

        this->mSequence->end();
    }
}

void
KomputeSummator::_process(float delta)
{}

void
KomputeSummator::_register_methods()
{
    register_method((char*)"_process", &KomputeSummator::_process);
    register_method((char*)"_init", &KomputeSummator::_init);

    register_method((char*)"add", &KomputeSummator::add);
    register_method((char*)"reset", &KomputeSummator::reset);
    register_method((char*)"get_total", &KomputeSummator::get_total);
}

}
