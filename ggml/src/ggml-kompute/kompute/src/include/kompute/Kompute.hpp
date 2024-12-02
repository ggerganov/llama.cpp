#pragma once

#include "Algorithm.hpp"
#include "Core.hpp"
#include "Manager.hpp"
#include "Sequence.hpp"
#include "Tensor.hpp"

#include "operations/OpAlgoDispatch.hpp"
#include "operations/OpBase.hpp"
#include "operations/OpMemoryBarrier.hpp"
#include "operations/OpMult.hpp"
#include "operations/OpTensorCopy.hpp"
#include "operations/OpTensorSyncDevice.hpp"
#include "operations/OpTensorSyncLocal.hpp"
#include "operations/OpBufferSyncDevice.hpp"
#include "operations/OpBufferSyncLocal.hpp"
#include "operations/OpTensorFill.hpp"

// Will be build by CMake and placed inside the build directory
#include "ShaderLogisticRegression.hpp"
#include "ShaderOpMult.hpp"
