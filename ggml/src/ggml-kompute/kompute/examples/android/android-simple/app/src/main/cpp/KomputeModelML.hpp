
#pragma once

#include <vector>

class KomputeModelML
{

  public:
    KomputeModelML();
    virtual ~KomputeModelML();

    void train(std::vector<float> yData,
               std::vector<float> xIData,
               std::vector<float> xJData);

    std::vector<float> predict(std::vector<float> xI, std::vector<float> xJ);

    std::vector<float> get_params();

  private:
    std::vector<float> mWeights;
    std::vector<float> mBias;
};
