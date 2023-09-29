cmake_minimum_required(VERSION 3.20)

# ####################################################
# Kompute
# ####################################################
target_include_directories(kompute PUBLIC $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_sources(kompute PRIVATE

    # Header files (useful in IDEs)
    kompute/Algorithm.hpp
    kompute/Core.hpp
    kompute/Kompute.hpp
    kompute/Manager.hpp
    kompute/Sequence.hpp
    kompute/Tensor.hpp

    kompute/operations/OpAlgoDispatch.hpp
    kompute/operations/OpBase.hpp
    kompute/operations/OpMemoryBarrier.hpp
    kompute/operations/OpMult.hpp
    kompute/operations/OpTensorCopy.hpp
    kompute/operations/OpTensorFill.hpp
    kompute/operations/OpTensorSyncDevice.hpp
    kompute/operations/OpTensorSyncLocal.hpp
    kompute/operations/OpBufferSyncDevice.hpp
    kompute/operations/OpBufferSyncLocal.hpp

    kompute/logger/Logger.hpp
)

#install(DIRECTORY kompute DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ####################################################
# Logger
# ####################################################
target_include_directories(kp_logger PUBLIC $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_sources(kp_logger PRIVATE

    # Header files (useful in IDEs)
    kompute/logger/Logger.hpp
)

#install(DIRECTORY logger DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})