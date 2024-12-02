#pragma once

#include <memory>

#include "kompute/Kompute.hpp"

#include "scene/main/node.h"

class KomputeSummatorNode : public Node
{
    GDCLASS(KomputeSummatorNode, Node);

  public:
    KomputeSummatorNode();

    void add(float value);
    void reset();
    float get_total() const;

    void _process(float delta);
    void _init();

  protected:
    static void _bind_methods();

  private:
    kp::Manager mManager;
    std::shared_ptr<kp::Sequence> mSequence;
    std::shared_ptr<kp::Tensor> mPrimaryTensor;
    std::shared_ptr<kp::Tensor> mSecondaryTensor;
};
