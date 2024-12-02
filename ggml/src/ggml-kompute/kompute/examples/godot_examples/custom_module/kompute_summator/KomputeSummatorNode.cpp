/* summator.cpp */

#include <vector>

#include "KomputeSummatorNode.h"

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

KomputeSummatorNode::KomputeSummatorNode()
{
    this->_init();
}

void
KomputeSummatorNode::add(float value)
{
    // Set the new data in the local device
    this->mSecondaryTensor->setData({ value });
    // Execute recorded sequence
    if (std::shared_ptr<kp::Sequence> sq = this->mSequence) {
        sq->eval();
    } else {
        throw std::runtime_error("Sequence pointer no longer available");
    }
}

void
KomputeSummatorNode::reset()
{}

float
KomputeSummatorNode::get_total() const
{
    return this->mPrimaryTensor->data()[0];
}

void
KomputeSummatorNode::_init()
{
    std::cout << "CALLING INIT" << std::endl;
    this->mPrimaryTensor = this->mManager.tensor({ 0.0 });
    this->mSecondaryTensor = this->mManager.tensor({ 0.0 });
    this->mSequence = this->mManager.sequence();

    // We now record the steps in the sequence
    if (std::shared_ptr<kp::Sequence> sq = this->mSequence) {

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

        std::shared_ptr<kp::Algorithm> algo = this->mManager.algorithm(
          { this->mPrimaryTensor, this->mSecondaryTensor },
          compileSource(shader));

        // First we ensure secondary tensor loads to GPU
        // No need to sync the primary tensor as it should not be changed
        sq->record<kp::OpTensorSyncDevice>({ this->mSecondaryTensor });

        // Then we run the operation with both tensors
        sq->record<kp::OpAlgoDispatch>(algo);

        // We map the result back to local
        sq->record<kp::OpTensorSyncLocal>({ this->mPrimaryTensor });

    } else {
        throw std::runtime_error("Sequence pointer no longer available");
    }
}

void
KomputeSummatorNode::_process(float delta)
{}

void
KomputeSummatorNode::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("_process", "delta"),
                         &KomputeSummatorNode::_process);
    ClassDB::bind_method(D_METHOD("_init"), &KomputeSummatorNode::_init);

    ClassDB::bind_method(D_METHOD("add", "value"), &KomputeSummatorNode::add);
    ClassDB::bind_method(D_METHOD("reset"), &KomputeSummatorNode::reset);
    ClassDB::bind_method(D_METHOD("get_total"),
                         &KomputeSummatorNode::get_total);
}
