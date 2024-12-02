// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "kompute/Kompute.hpp"
#include "kompute/logger/Logger.hpp"

TEST(TestTensor, ConstructorData)
{
    kp::Manager mgr;
    std::vector<float> vec{ 0, 1, 2 };
    std::shared_ptr<kp::TensorT<float>> tensor = mgr.tensor(vec);
    EXPECT_EQ(tensor->size(), vec.size());
    EXPECT_EQ(tensor->dataTypeMemorySize(), sizeof(float));
    EXPECT_EQ(tensor->vector(), vec);
}

TEST(TestTensor, DataTypes)
{
    kp::Manager mgr;

    {
        std::vector<float> vec{ 0, 1, 2 };
        std::shared_ptr<kp::TensorT<float>> tensor = mgr.tensor(vec);
        EXPECT_EQ(tensor->dataType(), kp::Tensor::TensorDataTypes::eFloat);
    }

    {
        std::vector<int32_t> vec{ 0, 1, 2 };
        std::shared_ptr<kp::TensorT<int32_t>> tensor = mgr.tensorT(vec);
        EXPECT_EQ(tensor->dataType(), kp::Tensor::TensorDataTypes::eInt);
    }

    {
        std::vector<uint32_t> vec{ 0, 1, 2 };
        std::shared_ptr<kp::TensorT<uint32_t>> tensor = mgr.tensorT(vec);
        EXPECT_EQ(tensor->dataType(),
                  kp::Tensor::TensorDataTypes::eUnsignedInt);
    }

    {
        std::vector<double> vec{ 0, 1, 2 };
        std::shared_ptr<kp::TensorT<double>> tensor = mgr.tensorT(vec);
        EXPECT_EQ(tensor->dataType(), kp::Tensor::TensorDataTypes::eDouble);
    }
}
