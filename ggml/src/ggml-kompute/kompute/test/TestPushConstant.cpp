// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "kompute/Kompute.hpp"
#include "kompute/logger/Logger.hpp"

#include "shaders/Utils.hpp"

TEST(TestPushConstants, TestConstantsAlgoDispatchOverride)
{
    {
        std::string shader(R"(
          #version 450
          layout(push_constant) uniform PushConstants {
            float x;
            float y;
            float z;
          } pcs;
          layout (local_size_x = 1) in;
          layout(set = 0, binding = 0) buffer a { float pa[]; };
          void main() {
              pa[0] += pcs.x;
              pa[1] += pcs.y;
              pa[2] += pcs.z;
          })");

        std::vector<uint32_t> spirv = compileSource(shader);

        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            std::shared_ptr<kp::TensorT<float>> tensor =
              mgr.tensor({ 0, 0, 0 });

            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
              { tensor }, spirv, kp::Workgroup({ 1 }), {}, { 0.0, 0.0, 0.0 });

            sq = mgr.sequence()->eval<kp::OpTensorSyncDevice>({ tensor });

            // We need to run this in sequence to avoid race condition
            // We can't use atomicAdd as swiftshader doesn't support it for
            // float
            sq->eval<kp::OpAlgoDispatch>(algo,
                                         std::vector<float>{ 0.1, 0.2, 0.3 });
            sq->eval<kp::OpAlgoDispatch>(algo,
                                         std::vector<float>{ 0.3, 0.2, 0.1 });
            sq->eval<kp::OpTensorSyncLocal>({ tensor });

            EXPECT_EQ(tensor->vector(), std::vector<float>({ 0.4, 0.4, 0.4 }));
        }
    }
}

TEST(TestPushConstants, TestConstantsAlgoDispatchNoOverride)
{
    {
        std::string shader(R"(
          #version 450
          layout(push_constant) uniform PushConstants {
            float x;
            float y;
            float z;
          } pcs;
          layout (local_size_x = 1) in;
          layout(set = 0, binding = 0) buffer a { float pa[]; };
          void main() {
              pa[0] += pcs.x;
              pa[1] += pcs.y;
              pa[2] += pcs.z;
          })");

        std::vector<uint32_t> spirv = compileSource(shader);

        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            std::shared_ptr<kp::TensorT<float>> tensor =
              mgr.tensor({ 0, 0, 0 });

            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
              { tensor }, spirv, kp::Workgroup({ 1 }), {}, { 0.1, 0.2, 0.3 });

            sq = mgr.sequence()->eval<kp::OpTensorSyncDevice>({ tensor });

            // We need to run this in sequence to avoid race condition
            // We can't use atomicAdd as swiftshader doesn't support it for
            // float
            sq->eval<kp::OpAlgoDispatch>(algo);
            sq->eval<kp::OpAlgoDispatch>(algo,
                                         std::vector<float>{ 0.3, 0.2, 0.1 });
            sq->eval<kp::OpTensorSyncLocal>({ tensor });

            EXPECT_EQ(tensor->vector(), std::vector<float>({ 0.4, 0.4, 0.4 }));
        }
    }
}

TEST(TestPushConstants, TestConstantsWrongSize)
{
    {
        std::string shader(R"(
          #version 450
          layout(push_constant) uniform PushConstants {
            float x;
            float y;
            float z;
          } pcs;
          layout (local_size_x = 1) in;
          layout(set = 0, binding = 0) buffer a { float pa[]; };
          void main() {
              pa[0] += pcs.x;
              pa[1] += pcs.y;
              pa[2] += pcs.z;
          })");

        std::vector<uint32_t> spirv = compileSource(shader);

        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            std::shared_ptr<kp::TensorT<float>> tensor =
              mgr.tensor({ 0, 0, 0 });

            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
              { tensor }, spirv, kp::Workgroup({ 1 }), {}, { 0.0 });

            sq = mgr.sequence()->record<kp::OpTensorSyncDevice>({ tensor });

            EXPECT_THROW(sq->record<kp::OpAlgoDispatch>(
                           algo, std::vector<float>{ 0.1, 0.2, 0.3 }),
                         std::runtime_error);
        }
    }
}

// TODO: Ensure different types are considered for push constants
// TEST(TestPushConstants, TestConstantsWrongType)
// {
//     {
//         std::string shader(R"(
//           #version 450
//           layout(push_constant) uniform PushConstants {
//             float x;
//             float y;
//             float z;
//           } pcs;
//           layout (local_size_x = 1) in;
//           layout(set = 0, binding = 0) buffer a { float pa[]; };
//           void main() {
//               pa[0] += pcs.x;
//               pa[1] += pcs.y;
//               pa[2] += pcs.z;
//           })");
//
//         std::vector<uint32_t> spirv = compileSource(shader);
//
//         std::shared_ptr<kp::Sequence> sq = nullptr;
//
//         {
//             kp::Manager mgr;
//
//             std::shared_ptr<kp::TensorT<float>> tensor =
//               mgr.tensor({ 0, 0, 0 });
//
//             std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
//               { tensor }, spirv, kp::Workgroup({ 1 }), {}, { 0.0 });
//
//             sq = mgr.sequence()->record<kp::OpTensorSyncDevice>({ tensor });
//
//             EXPECT_THROW(sq->record<kp::OpAlgoDispatch>(
//                            algo, std::vector<uint32_t>{ 1, 2, 3 }),
//                          std::runtime_error);
//         }
//     }
// }

TEST(TestPushConstants, TestConstantsMixedTypes)
{
    {
        std::string shader(R"(
          #version 450
          layout(push_constant) uniform PushConstants {
            float x;
            uint y;
            int z;
          } pcs;
          layout (local_size_x = 1) in;
          layout(set = 0, binding = 0) buffer a { float pa[]; };
          void main() {
              pa[0] += pcs.x;
              pa[1] += pcs.y - 2147483000;
              pa[2] += pcs.z;
          })");

        struct TestConsts
        {
            float x;
            uint32_t y;
            int32_t z;
        };

        std::vector<uint32_t> spirv = compileSource(shader);

        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            std::shared_ptr<kp::TensorT<float>> tensor =
              mgr.tensorT<float>({ 0, 0, 0 });

            std::shared_ptr<kp::Algorithm> algo =
              mgr.algorithm<float, TestConsts>(
                { tensor }, spirv, kp::Workgroup({ 1 }), {}, { { 0, 0, 0 } });

            sq = mgr.sequence()->eval<kp::OpTensorSyncDevice>({ tensor });

            // We need to run this in sequence to avoid race condition
            // We can't use atomicAdd as swiftshader doesn't support it for
            // float
            sq->eval<kp::OpAlgoDispatch>(
              algo, std::vector<TestConsts>{ { 15.32, 2147483650, 10 } });
            sq->eval<kp::OpAlgoDispatch>(
              algo, std::vector<TestConsts>{ { 30.32, 2147483650, -3 } });
            sq->eval<kp::OpTensorSyncLocal>({ tensor });

            EXPECT_EQ(tensor->vector(), std::vector<float>({ 45.64, 1300, 7 }));
        }
    }
}

TEST(TestPushConstants, TestConstantsInt)
{
    {
        std::string shader(R"(
          #version 450
          layout(push_constant) uniform PushConstants {
            int x;
            int y;
            int z;
          } pcs;
          layout (local_size_x = 1) in;
          layout(set = 0, binding = 0) buffer a { int pa[]; };
          void main() {
              pa[0] += pcs.x;
              pa[1] += pcs.y;
              pa[2] += pcs.z;
          })");

        std::vector<uint32_t> spirv = compileSource(shader);

        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            std::shared_ptr<kp::TensorT<int32_t>> tensor =
              mgr.tensorT<int32_t>({ -1, -1, -1 });

            std::shared_ptr<kp::Algorithm> algo =
              mgr.algorithm<int32_t, int32_t>(
                { tensor }, spirv, kp::Workgroup({ 1 }), {}, { { 0, 0, 0 } });

            sq = mgr.sequence()->eval<kp::OpTensorSyncDevice>({ tensor });

            // We need to run this in sequence to avoid race condition
            // We can't use atomicAdd as swiftshader doesn't support it for
            // float
            sq->eval<kp::OpAlgoDispatch>(
              algo, std::vector<int32_t>{ { -1, -1, -1 } });
            sq->eval<kp::OpAlgoDispatch>(
              algo, std::vector<int32_t>{ { -1, -1, -1 } });
            sq->eval<kp::OpTensorSyncLocal>({ tensor });

            EXPECT_EQ(tensor->vector(), std::vector<int32_t>({ -3, -3, -3 }));
        }
    }
}

TEST(TestPushConstants, TestConstantsUnsignedInt)
{
    {
        std::string shader(R"(
          #version 450
          layout(push_constant) uniform PushConstants {
            uint x;
            uint y;
            uint z;
          } pcs;
          layout (local_size_x = 1) in;
          layout(set = 0, binding = 0) buffer a { uint pa[]; };
          void main() {
              pa[0] += pcs.x;
              pa[1] += pcs.y;
              pa[2] += pcs.z;
          })");

        std::vector<uint32_t> spirv = compileSource(shader);

        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            std::shared_ptr<kp::TensorT<uint32_t>> tensor =
              mgr.tensorT<uint32_t>({ 0, 0, 0 });

            std::shared_ptr<kp::Algorithm> algo =
              mgr.algorithm<uint32_t, uint32_t>(
                { tensor }, spirv, kp::Workgroup({ 1 }), {}, { { 0, 0, 0 } });

            sq = mgr.sequence()->eval<kp::OpTensorSyncDevice>({ tensor });

            // We need to run this in sequence to avoid race condition
            // We can't use atomicAdd as swiftshader doesn't support it for
            // float
            sq->eval<kp::OpAlgoDispatch>(
              algo,
              std::vector<uint32_t>{ { 2147483650, 2147483650, 2147483650 } });
            sq->eval<kp::OpAlgoDispatch>(algo,
                                         std::vector<uint32_t>{ { 5, 5, 5 } });
            sq->eval<kp::OpTensorSyncLocal>({ tensor });

            EXPECT_EQ(
              tensor->vector(),
              std::vector<uint32_t>({ 2147483655, 2147483655, 2147483655 }));
        }
    }
}

TEST(TestPushConstants, TestConstantsDouble)
{
    {
        std::string shader(R"(
          #version 450
          layout(push_constant) uniform PushConstants {
            double x;
            double y;
            double z;
          } pcs;
          layout (local_size_x = 1) in;
          layout(set = 0, binding = 0) buffer a { double pa[]; };
          void main() {
              pa[0] += pcs.x;
              pa[1] += pcs.y;
              pa[2] += pcs.z;
          })");

        std::vector<uint32_t> spirv = compileSource(shader);

        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            std::shared_ptr<kp::TensorT<double>> tensor =
              mgr.tensorT<double>({ 0, 0, 0 });

            std::shared_ptr<kp::Algorithm> algo = mgr.algorithm<double, double>(
              { tensor }, spirv, kp::Workgroup({ 1 }), {}, { { 0, 0, 0 } });

            sq = mgr.sequence()->eval<kp::OpTensorSyncDevice>({ tensor });

            // We need to run this in sequence to avoid race condition
            // We can't use atomicAdd as swiftshader doesn't support it for
            // float
            sq->eval<kp::OpAlgoDispatch>(
              algo,
              std::vector<double>{ { 1.1111222233334444,
                                     2.1111222233334444,
                                     3.1111222233334444 } });
            sq->eval<kp::OpAlgoDispatch>(
              algo,
              std::vector<double>{ { 1.1111222233334444,
                                     2.1111222233334444,
                                     3.1111222233334444 } });
            sq->eval<kp::OpTensorSyncLocal>({ tensor });

            EXPECT_EQ(tensor->vector(),
                      std::vector<double>({ 2.2222444466668888,
                                            4.2222444466668888,
                                            6.2222444466668888 }));
        }
    }
}
