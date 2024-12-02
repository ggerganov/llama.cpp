#pragma once

#include <Godot.hpp>
#include <Node2D.hpp>

#include <memory>

#include "kompute/Kompute.hpp"

namespace godot {
class KomputeSummator : public Node2D
{
  private:
    GODOT_CLASS(KomputeSummator, Node2D);

  public:
    KomputeSummator();

    void add(float value);
    void reset();
    float get_total() const;

    void _process(float delta);
    void _init();

    static void _register_methods();

  private:
    kp::Manager mManager;
    std::shared_ptr<kp::Sequence> mSequence;
    std::shared_ptr<kp::Tensor> mPrimaryTensor;
    std::shared_ptr<kp::Tensor> mSecondaryTensor;
};

}
