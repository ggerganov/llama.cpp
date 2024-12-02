/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_kp_Algorithm =
R"doc(Abstraction for compute shaders that are run on top of tensors grouped
via ParameterGroups (which group descriptorsets))doc";

static const char *__doc_kp_Algorithm_Algorithm =
R"doc(Main constructor for algorithm with configuration parameters to create
the underlying resources.

@param device The Vulkan device to use for creating resources @param
tensors (optional) The tensors to use to create the descriptor
resources @param spirv (optional) The spirv code to use to create the
algorithm @param workgroup (optional) The kp::Workgroup to use for the
dispatch which defaults to kp::Workgroup(tensor[0].size(), 1, 1) if
not set. @param specializationConstants (optional) The std::vector<float>
to use to initialize the specialization constants which cannot be
changed once set. @param pushConstants (optional) The std::vector<float> to
use when initializing the pipeline, which set the size of the push
constants - these can be modified but all new values must have the
same vector size as this initial value.)doc";

static const char *__doc_kp_Algorithm_createParameters = R"doc()doc";

static const char *__doc_kp_Algorithm_createPipeline = R"doc()doc";

static const char *__doc_kp_Algorithm_createShaderModule = R"doc()doc";

static const char *__doc_kp_Algorithm_destroy = R"doc()doc";

static const char *__doc_kp_Algorithm_getPush =
R"doc(Gets the specialization constants of the current algorithm.

@returns The std::vector<float> currently set for push constants)doc";

static const char *__doc_kp_Algorithm_getSpecializationConstants =
R"doc(Gets the specialization constants of the current algorithm.

@returns The std::vector<float> currently set for specialization constants)doc";

static const char *__doc_kp_Algorithm_getTensors =
R"doc(Gets the current tensors that are used in the algorithm.

@returns The list of tensors used in the algorithm.)doc";

static const char *__doc_kp_Algorithm_getWorkgroup =
R"doc(Gets the current workgroup from the algorithm.

@param The kp::Constant to use to set the push constants to use in the
next bindPush(...) calls. The constants provided must be of the same
size as the ones created during initialization.)doc";

static const char *__doc_kp_Algorithm_isInit =
R"doc(function that checks all the gpu resource components to verify if
these have been created and returns true if all are valid.

@returns returns true if the algorithm is currently initialized.)doc";

static const char *__doc_kp_Algorithm_mDescriptorPool = R"doc()doc";

static const char *__doc_kp_Algorithm_mDescriptorSet = R"doc()doc";

static const char *__doc_kp_Algorithm_mDescriptorSetLayout = R"doc()doc";

static const char *__doc_kp_Algorithm_mDevice = R"doc()doc";

static const char *__doc_kp_Algorithm_mFreeDescriptorPool = R"doc()doc";

static const char *__doc_kp_Algorithm_mFreeDescriptorSet = R"doc()doc";

static const char *__doc_kp_Algorithm_mFreeDescriptorSetLayout = R"doc()doc";

static const char *__doc_kp_Algorithm_mFreePipeline = R"doc()doc";

static const char *__doc_kp_Algorithm_mFreePipelineCache = R"doc()doc";

static const char *__doc_kp_Algorithm_mFreePipelineLayout = R"doc()doc";

static const char *__doc_kp_Algorithm_mFreeShaderModule = R"doc()doc";

static const char *__doc_kp_Algorithm_mPipeline = R"doc()doc";

static const char *__doc_kp_Algorithm_mPipelineCache = R"doc()doc";

static const char *__doc_kp_Algorithm_mPipelineLayout = R"doc()doc";

static const char *__doc_kp_Algorithm_mPushConstants = R"doc()doc";

static const char *__doc_kp_Algorithm_mShaderModule = R"doc()doc";

static const char *__doc_kp_Algorithm_mSpecializationConstants = R"doc()doc";

static const char *__doc_kp_Algorithm_mSpirv = R"doc()doc";

static const char *__doc_kp_Algorithm_mTensors = R"doc()doc";

static const char *__doc_kp_Algorithm_mWorkgroup = R"doc()doc";

static const char *__doc_kp_Algorithm_rebuild =
R"doc(Rebuild function to reconstruct algorithm with configuration
parameters to create the underlying resources.

@param tensors The tensors to use to create the descriptor resources
@param spirv The spirv code to use to create the algorithm @param
workgroup (optional) The kp::Workgroup to use for the dispatch which
defaults to kp::Workgroup(tensor[0].size(), 1, 1) if not set. @param
specializationConstants (optional) The std::vector<float> to use to
initialize the specialization constants which cannot be changed once
set. @param pushConstants (optional) The std::vector<float> to use when
initializing the pipeline, which set the size of the push constants -
these can be modified but all new values must have the same vector
size as this initial value.)doc";

static const char *__doc_kp_Algorithm_recordBindCore =
R"doc(Records command that binds the "core" algorithm components which
consist of binding the pipeline and binding the descriptorsets.

@param commandBuffer Command buffer to record the algorithm resources
to)doc";

static const char *__doc_kp_Algorithm_recordBindPush =
R"doc(Records command that binds the push constants to the command buffer
provided - it is required that the pushConstants provided are of the
same size as the ones provided during initialization.

@param commandBuffer Command buffer to record the algorithm resources
to)doc";

static const char *__doc_kp_Algorithm_recordDispatch =
R"doc(Records the dispatch function with the provided template parameters or
alternatively using the size of the tensor by default.

@param commandBuffer Command buffer to record the algorithm resources
to)doc";

static const char *__doc_kp_Algorithm_setPush =
R"doc(Sets the push constants to the new value provided to use in the next
bindPush()

@param The kp::Constant to use to set the push constants to use in the
next bindPush(...) calls. The constants provided must be of the same
size as the ones created during initialization.)doc";

static const char *__doc_kp_Algorithm_setWorkgroup =
R"doc(Sets the work group to use in the recordDispatch

@param workgroup The kp::Workgroup value to use to update the
algorithm. It must have a value greater than 1 on the x value (index
1) otherwise it will be initialized on the size of the first tensor
(ie. this->mTensor[0]->size()))doc";

static const char *__doc_kp_Manager =
R"doc(Base orchestrator which creates and manages device and child
components)doc";

static const char *__doc_kp_Manager_Manager =
R"doc(Base constructor and default used which creates the base resources
including choosing the device 0 by default.)doc";

static const char *__doc_kp_Manager_Manager_2 =
R"doc(Similar to base constructor but allows for further configuration to
use when creating the Vulkan resources.

@param physicalDeviceIndex The index of the physical device to use
@param familyQueueIndices (Optional) List of queue indices to add for
explicit allocation @param desiredExtensions The desired extensions to
load from physicalDevice)doc";

static const char *__doc_kp_Manager_Manager_3 =
R"doc(Manager constructor which allows your own vulkan application to
integrate with the kompute use.

@param instance Vulkan compute instance to base this application
@param physicalDevice Vulkan physical device to use for application
@param device Vulkan logical device to use for all base resources
@param physicalDeviceIndex Index for vulkan physical device used)doc";

static const char *__doc_kp_Manager_algorithm =
R"doc(Create a managed algorithm that will be destroyed by this manager if
it hasn't been destroyed by its reference count going to zero.

@param tensors (optional) The tensors to initialise the algorithm with
@param spirv (optional) The SPIRV bytes for the algorithm to dispatch
@param workgroup (optional) kp::Workgroup for algorithm to use, and
defaults to (tensor[0].size(), 1, 1) @param specializationConstants
(optional) kp::Constant to use for specialization constants, and
defaults to an empty constant @param pushConstants (optional)
kp::Constant to use for push constants, and defaults to an empty
constant @returns Shared pointer with initialised algorithm)doc";

static const char *__doc_kp_Manager_clear =
R"doc(Run a pseudo-garbage collection to release all the managed resources
that have been already freed due to these reaching to zero ref count.)doc";

static const char *__doc_kp_Manager_createDevice = R"doc()doc";

static const char *__doc_kp_Manager_createInstance = R"doc()doc";

static const char *__doc_kp_Manager_destroy = R"doc(Destroy the GPU resources and all managed resources by manager.)doc";

static const char *__doc_kp_Manager_mComputeQueueFamilyIndices = R"doc()doc";

static const char *__doc_kp_Manager_mComputeQueues = R"doc()doc";

static const char *__doc_kp_Manager_mDevice = R"doc()doc";

static const char *__doc_kp_Manager_mFreeDevice = R"doc()doc";

static const char *__doc_kp_Manager_mFreeInstance = R"doc()doc";

static const char *__doc_kp_Manager_mInstance = R"doc()doc";

static const char *__doc_kp_Manager_mManageResources = R"doc()doc";

static const char *__doc_kp_Manager_mManagedAlgorithms = R"doc()doc";

static const char *__doc_kp_Manager_mManagedSequences = R"doc()doc";

static const char *__doc_kp_Manager_mManagedTensors = R"doc()doc";

static const char *__doc_kp_Manager_mPhysicalDevice = R"doc()doc";

static const char *__doc_kp_Manager_sequence =
R"doc(Create a managed sequence that will be destroyed by this manager if it
hasn't been destroyed by its reference count going to zero.

@param queueIndex The queue to use from the available queues @param
nrOfTimestamps The maximum number of timestamps to allocate. If zero
(default), disables latching of timestamps. @returns Shared pointer
with initialised sequence)doc";

static const char *__doc_kp_Manager_tensor = R"doc()doc";

static const char *__doc_kp_Manager_tensor_2 = R"doc()doc";

static const char *__doc_kp_Manager_tensorT =
R"doc(Create a managed tensor that will be destroyed by this manager if it
hasn't been destroyed by its reference count going to zero.

@param data The data to initialize the tensor with @param tensorType
The type of tensor to initialize @returns Shared pointer with
initialised tensor)doc";

static const char *__doc_kp_OpAlgoDispatch =
R"doc(Operation that provides a general abstraction that simplifies the use
of algorithm and parameter components which can be used with shaders.
By default it enables the user to provide a dynamic number of tensors
which are then passed as inputs.)doc";

static const char *__doc_kp_OpAlgoDispatch_OpAlgoDispatch =
R"doc(Constructor that stores the algorithm to use as well as the relevant
push constants to override when recording.

@param algorithm The algorithm object to use for dispatch @param
pushConstants The push constants to use for override)doc";

static const char *__doc_kp_OpAlgoDispatch_mAlgorithm = R"doc()doc";

static const char *__doc_kp_OpAlgoDispatch_mPushConstants = R"doc()doc";

static const char *__doc_kp_OpAlgoDispatch_postEval =
R"doc(Does not perform any postEval commands.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpAlgoDispatch_preEval =
R"doc(Does not perform any preEval commands.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpAlgoDispatch_record =
R"doc(This records the commands that are to be sent to the GPU. This
includes the barriers that ensure the memory has been copied before
going in and out of the shader, as well as the dispatch operation that
sends the shader processing to the gpu. This function also records the
GPU memory copy of the output data for the staging buffer so it can be
read by the host.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpBase =
R"doc(Base Operation which provides the high level interface that Kompute
operations implement in order to perform a set of actions in the GPU.

Operations can perform actions on tensors, and optionally can also own
an Algorithm with respective parameters. kp::Operations with
kp::Algorithms would inherit from kp::OpBaseAlgo.)doc";

static const char *__doc_kp_OpBase_postEval =
R"doc(Post eval is called after the Sequence has called eval and submitted
the commands to the GPU for processing, and can be used to perform any
tear-down steps required as the computation iteration finishes. It's
worth noting that there are situations where eval can be called
multiple times, so the resources that are destroyed should not require
a re-init unless explicitly provided by the user.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpBase_preEval =
R"doc(Pre eval is called before the Sequence has called eval and submitted
the commands to the GPU for processing, and can be used to perform any
per-eval setup steps required as the computation iteration begins.
It's worth noting that there are situations where eval can be called
multiple times, so the resources that are created should be idempotent
in case it's called multiple times in a row.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpBase_record =
R"doc(The record function is intended to only send a record command or run
commands that are expected to record operations that are to be
submitted as a batch into the GPU.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpMult =
R"doc(Operation that performs multiplication on two tensors and outpus on
third tensor.)doc";

static const char *__doc_kp_OpMult_OpMult =
R"doc(Default constructor with parameters that provides the bare minimum
requirements for the operations to be able to create and manage their
sub-components.

@param tensors Tensors that are to be used in this operation @param
algorithm An algorithm that will be overridden with the OpMult shader
data and the tensors provided which are expected to be 3)doc";

static const char *__doc_kp_OpTensorCopy =
R"doc(Operation that copies the data from the first tensor to the rest of
the tensors provided, using a record command for all the vectors. This
operation does not own/manage the memory of the tensors passed to it.
The operation must only receive tensors of type)doc";

static const char *__doc_kp_OpTensorCopy_OpTensorCopy =
R"doc(Default constructor with parameters that provides the core vulkan
resources and the tensors that will be used in the operation.

@param tensors Tensors that will be used to create in operation.)doc";

static const char *__doc_kp_OpTensorCopy_mTensors = R"doc()doc";

static const char *__doc_kp_OpTensorCopy_postEval =
R"doc(Copies the local vectors for all the tensors to sync the data with the
gpu.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpTensorCopy_preEval =
R"doc(Does not perform any preEval commands.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpTensorCopy_record =
R"doc(Records the copy commands from the first tensor into all the other
tensors provided. Also optionally records a barrier.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpTensorSyncDevice =
R"doc(Operation that syncs tensor's device by mapping local data into the
device memory. For TensorTypes::eDevice it will use a record operation
for the memory to be syncd into GPU memory which means that the
operation will be done in sync with GPU commands. For
TensorTypes::eHost it will only map the data into host memory which
will happen during preEval before the recorded commands are
dispatched.)doc";

static const char *__doc_kp_OpTensorSyncDevice_OpTensorSyncDevice =
R"doc(Default constructor with parameters that provides the core vulkan
resources and the tensors that will be used in the operation. The
tensos provided cannot be of type TensorTypes::eStorage.

@param tensors Tensors that will be used to create in operation.)doc";

static const char *__doc_kp_OpTensorSyncDevice_mTensors = R"doc()doc";

static const char *__doc_kp_OpTensorSyncDevice_postEval =
R"doc(Does not perform any postEval commands.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpTensorSyncDevice_preEval =
R"doc(Does not perform any preEval commands.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpTensorSyncDevice_record =
R"doc(For device tensors, it records the copy command for the tensor to copy
the data from its staging to device memory.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpTensorSyncLocal =
R"doc(Operation that syncs tensor's local memory by mapping device data into
the local CPU memory. For TensorTypes::eDevice it will use a record
operation for the memory to be syncd into GPU memory which means that
the operation will be done in sync with GPU commands. For
TensorTypes::eHost it will only map the data into host memory which
will happen during preEval before the recorded commands are
dispatched.)doc";

static const char *__doc_kp_OpTensorSyncLocal_OpTensorSyncLocal =
R"doc(Default constructor with parameters that provides the core vulkan
resources and the tensors that will be used in the operation. The
tensors provided cannot be of type TensorTypes::eStorage.

@param tensors Tensors that will be used to create in operation.)doc";

static const char *__doc_kp_OpTensorSyncLocal_mTensors = R"doc()doc";

static const char *__doc_kp_OpTensorSyncLocal_postEval =
R"doc(For host tensors it performs the map command from the host memory into
local memory.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpTensorSyncLocal_preEval =
R"doc(Does not perform any preEval commands.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_OpTensorSyncLocal_record =
R"doc(For device tensors, it records the copy command for the tensor to copy
the data from its device to staging memory.

@param commandBuffer The command buffer to record the command into.)doc";

static const char *__doc_kp_Sequence = R"doc(Container of operations that can be sent to GPU as batch)doc";

static const char *__doc_kp_Sequence_Sequence =
R"doc(Main constructor for sequence which requires core vulkan components to
generate all dependent resources.

@param physicalDevice Vulkan physical device @param device Vulkan
logical device @param computeQueue Vulkan compute queue @param
queueIndex Vulkan compute queue index in device @param totalTimestamps
Maximum number of timestamps to allocate)doc";

static const char *__doc_kp_Sequence_begin =
R"doc(Begins recording commands for commands to be submitted into the
command buffer.

@return Boolean stating whether execution was successful.)doc";

static const char *__doc_kp_Sequence_clear =
R"doc(Clear function clears all operations currently recorded and starts
recording again.)doc";

static const char *__doc_kp_Sequence_createCommandBuffer = R"doc()doc";

static const char *__doc_kp_Sequence_createCommandPool = R"doc()doc";

static const char *__doc_kp_Sequence_createTimestampQueryPool = R"doc()doc";

static const char *__doc_kp_Sequence_destroy =
R"doc(Destroys and frees the GPU resources which include the buffer and
memory and sets the sequence as init=False.)doc";

static const char *__doc_kp_Sequence_end =
R"doc(Ends the recording and stops recording commands when the record
command is sent.

@return Boolean stating whether execution was successful.)doc";

static const char *__doc_kp_Sequence_eval =
R"doc(Eval sends all the recorded and stored operations in the vector of
operations into the gpu as a submit job synchronously (with a
barrier).

@return shared_ptr<Sequence> of the Sequence class itself)doc";

static const char *__doc_kp_Sequence_eval_2 =
R"doc(Resets all the recorded and stored operations, records the operation
provided and submits into the gpu as a submit job synchronously (with
a barrier).

@return shared_ptr<Sequence> of the Sequence class itself)doc";

static const char *__doc_kp_Sequence_eval_3 =
R"doc(Eval sends all the recorded and stored operations in the vector of
operations into the gpu as a submit job with a barrier.

@param tensors Vector of tensors to use for the operation @param TArgs
Template parameters that are used to initialise operation which allows
for extensible configurations on initialisation. @return
shared_ptr<Sequence> of the Sequence class itself)doc";

static const char *__doc_kp_Sequence_eval_4 =
R"doc(Eval sends all the recorded and stored operations in the vector of
operations into the gpu as a submit job with a barrier.

@param algorithm Algorithm to use for the record often used for OpAlgo
operations @param TArgs Template parameters that are used to
initialise operation which allows for extensible configurations on
initialisation. @return shared_ptr<Sequence> of the Sequence class
itself)doc";

static const char *__doc_kp_Sequence_evalAsync =
R"doc(Eval Async sends all the recorded and stored operations in the vector
of operations into the gpu as a submit job without a barrier.
EvalAwait() must ALWAYS be called after to ensure the sequence is
terminated correctly.

@return Boolean stating whether execution was successful.)doc";

static const char *__doc_kp_Sequence_evalAsync_2 =
R"doc(Clears currnet operations to record provided one in the vector of
operations into the gpu as a submit job without a barrier. EvalAwait()
must ALWAYS be called after to ensure the sequence is terminated
correctly.

@return Boolean stating whether execution was successful.)doc";

static const char *__doc_kp_Sequence_evalAsync_3 =
R"doc(Eval sends all the recorded and stored operations in the vector of
operations into the gpu as a submit job with a barrier.

@param tensors Vector of tensors to use for the operation @param TArgs
Template parameters that are used to initialise operation which allows
for extensible configurations on initialisation. @return
shared_ptr<Sequence> of the Sequence class itself)doc";

static const char *__doc_kp_Sequence_evalAsync_4 =
R"doc(Eval sends all the recorded and stored operations in the vector of
operations into the gpu as a submit job with a barrier.

@param algorithm Algorithm to use for the record often used for OpAlgo
operations @param TArgs Template parameters that are used to
initialise operation which allows for extensible configurations on
initialisation. @return shared_ptr<Sequence> of the Sequence class
itself)doc";

static const char *__doc_kp_Sequence_evalAwait =
R"doc(Eval Await waits for the fence to finish processing and then once it
finishes, it runs the postEval of all operations.

@param waitFor Number of milliseconds to wait before timing out.
@return shared_ptr<Sequence> of the Sequence class itself)doc";

static const char *__doc_kp_Sequence_getTimestamps =
R"doc(Return the timestamps that were latched at the beginning and after
each operation during the last eval() call.)doc";

static const char *__doc_kp_Sequence_isInit =
R"doc(Returns true if the sequence has been initialised, and it's based on
the GPU resources being refrenced.

@return Boolean stating if is initialized)doc";

static const char *__doc_kp_Sequence_isRecording =
R"doc(Returns true if the sequence is currently in recording activated.

@return Boolean stating if recording ongoing.)doc";

static const char *__doc_kp_Sequence_isRunning =
R"doc(Returns true if the sequence is currently running - mostly used for
async workloads.

@return Boolean stating if currently running.)doc";

static const char *__doc_kp_Sequence_mCommandBuffer = R"doc()doc";

static const char *__doc_kp_Sequence_mCommandPool = R"doc()doc";

static const char *__doc_kp_Sequence_mComputeQueue = R"doc()doc";

static const char *__doc_kp_Sequence_mDevice = R"doc()doc";

static const char *__doc_kp_Sequence_mFence = R"doc()doc";

static const char *__doc_kp_Sequence_mFreeCommandBuffer = R"doc()doc";

static const char *__doc_kp_Sequence_mFreeCommandPool = R"doc()doc";

static const char *__doc_kp_Sequence_mIsRunning = R"doc()doc";

static const char *__doc_kp_Sequence_mOperations = R"doc()doc";

static const char *__doc_kp_Sequence_mPhysicalDevice = R"doc()doc";

static const char *__doc_kp_Sequence_mQueueIndex = R"doc()doc";

static const char *__doc_kp_Sequence_mRecording = R"doc()doc";

static const char *__doc_kp_Sequence_record =
R"doc(Record function for operation to be added to the GPU queue in batch.
This template requires classes to be derived from the OpBase class.
This function also requires the Sequence to be recording, otherwise it
will not be able to add the operation.

@param op Object derived from kp::BaseOp that will be recoreded by the
sequence which will be used when the operation is evaluated. @return
shared_ptr<Sequence> of the Sequence class itself)doc";

static const char *__doc_kp_Sequence_record_2 =
R"doc(Record function for operation to be added to the GPU queue in batch.
This template requires classes to be derived from the OpBase class.
This function also requires the Sequence to be recording, otherwise it
will not be able to add the operation.

@param tensors Vector of tensors to use for the operation @param TArgs
Template parameters that are used to initialise operation which allows
for extensible configurations on initialisation. @return
shared_ptr<Sequence> of the Sequence class itself)doc";

static const char *__doc_kp_Sequence_record_3 =
R"doc(Record function for operation to be added to the GPU queue in batch.
This template requires classes to be derived from the OpBase class.
This function also requires the Sequence to be recording, otherwise it
will not be able to add the operation.

@param algorithm Algorithm to use for the record often used for OpAlgo
operations @param TArgs Template parameters that are used to
initialise operation which allows for extensible configurations on
initialisation. @return shared_ptr<Sequence> of the Sequence class
itself)doc";

static const char *__doc_kp_Sequence_rerecord =
R"doc(Clears command buffer and triggers re-record of all the current
operations saved, which is useful if the underlying kp::Tensors or
kp::Algorithms are modified and need to be re-recorded.)doc";

static const char *__doc_kp_Sequence_timestampQueryPool = R"doc()doc";

static const char *__doc_kp_Shader = R"doc(Shader utily class with functions to compile and process glsl files.)doc";

static const char *__doc_kp_Shader_compileSource =
R"doc(Compile a single glslang source from string value. Currently this
function uses the glslang C++ interface which is not thread safe so
this funciton should not be called from multiple threads concurrently.
If you have a online shader processing multithreading use-case that
can't use offline compilation please open an issue.

@param source An individual raw glsl shader in string format @param
entryPoint The function name to use as entry point @param definitions
List of pairs containing key value definitions @param resourcesLimit A
list that contains the resource limits for the GLSL compiler @return
The compiled SPIR-V binary in unsigned int32 format)doc";

static const char *__doc_kp_Shader_compileSources =
R"doc(Compile multiple sources with optional filenames. Currently this
function uses the glslang C++ interface which is not thread safe so
this funciton should not be called from multiple threads concurrently.
If you have a online shader processing multithreading use-case that
can't use offline compilation please open an issue.

@param sources A list of raw glsl shaders in string format @param
files A list of file names respective to each of the sources @param
entryPoint The function name to use as entry point @param definitions
List of pairs containing key value definitions @param resourcesLimit A
list that contains the resource limits for the GLSL compiler @return
The compiled SPIR-V binary in unsigned int32 format)doc";

static const char *__doc_kp_Tensor =
R"doc(Structured data used in GPU operations.

Tensors are the base building block in Kompute to perform operations
across GPUs. Each tensor would have a respective Vulkan memory and
buffer, which would be used to store their respective data. The
tensors can be used for GPU data storage or transfer.)doc";

static const char *__doc_kp_TensorT = R"doc()doc";

static const char *__doc_kp_TensorT_TensorT = R"doc()doc";

static const char *__doc_kp_TensorT_data = R"doc()doc";

static const char *__doc_kp_TensorT_dataType = R"doc()doc";

static const char *__doc_kp_TensorT_operator_array = R"doc()doc";

static const char *__doc_kp_TensorT_setData = R"doc()doc";

static const char *__doc_kp_TensorT_vector = R"doc()doc";

static const char *__doc_kp_Tensor_Tensor =
R"doc(Constructor with data provided which would be used to create the
respective vulkan buffer and memory.

@param physicalDevice The physical device to use to fetch properties
@param device The device to use to create the buffer and memory from
@param data Non-zero-sized vector of data that will be used by the
tensor @param tensorTypes Type for the tensor which is of type
TensorTypes)doc";

static const char *__doc_kp_Tensor_TensorDataTypes = R"doc()doc";

static const char *__doc_kp_Tensor_TensorDataTypes_eBool = R"doc()doc";

static const char *__doc_kp_Tensor_TensorDataTypes_eDouble = R"doc()doc";

static const char *__doc_kp_Tensor_TensorDataTypes_eFloat = R"doc()doc";

static const char *__doc_kp_Tensor_TensorDataTypes_eInt = R"doc()doc";

static const char *__doc_kp_Tensor_TensorDataTypes_eUnsignedInt = R"doc()doc";

static const char *__doc_kp_Tensor_TensorTypes =
R"doc(Type for tensors created: Device allows memory to be transferred from
staging buffers. Staging are host memory visible. Storage are device
visible but are not set up to transfer or receive data (only for
shader storage).)doc";

static const char *__doc_kp_Tensor_TensorTypes_eDevice = R"doc(< Type is device memory, source and destination)doc";

static const char *__doc_kp_Tensor_TensorTypes_eHost = R"doc(< Type is host memory, source and destination)doc";

static const char *__doc_kp_Tensor_TensorTypes_eStorage = R"doc(< Type is Device memory (only))doc";

static const char *__doc_kp_Tensor_allocateBindMemory = R"doc()doc";

static const char *__doc_kp_Tensor_allocateMemoryCreateGPUResources = R"doc()doc";

static const char *__doc_kp_Tensor_constructDescriptorBufferInfo =
R"doc(Constructs a vulkan descriptor buffer info which can be used to
specify and reference the underlying buffer component of the tensor
without exposing it.

@return Descriptor buffer info with own buffer)doc";

static const char *__doc_kp_Tensor_createBuffer = R"doc()doc";

static const char *__doc_kp_Tensor_data = R"doc()doc";

static const char *__doc_kp_Tensor_dataType =
R"doc(Retrieve the underlying data type of the Tensor

@return Data type of tensor of type kp::Tensor::TensorDataTypes)doc";

static const char *__doc_kp_Tensor_dataTypeMemorySize = R"doc()doc";

static const char *__doc_kp_Tensor_destroy =
R"doc(Destroys and frees the GPU resources which include the buffer and
memory.)doc";

static const char *__doc_kp_Tensor_getPrimaryBufferUsageFlags = R"doc()doc";

static const char *__doc_kp_Tensor_getPrimaryMemoryPropertyFlags = R"doc()doc";

static const char *__doc_kp_Tensor_getStagingBufferUsageFlags = R"doc()doc";

static const char *__doc_kp_Tensor_getStagingMemoryPropertyFlags = R"doc()doc";

static const char *__doc_kp_Tensor_isInit =
R"doc(Check whether tensor is initialized based on the created gpu
resources.

@returns Boolean stating whether tensor is initialized)doc";

static const char *__doc_kp_Tensor_mDataType = R"doc()doc";

static const char *__doc_kp_Tensor_mDataTypeMemorySize = R"doc()doc";

static const char *__doc_kp_Tensor_mDevice = R"doc()doc";

static const char *__doc_kp_Tensor_mFreePrimaryBuffer = R"doc()doc";

static const char *__doc_kp_Tensor_mFreePrimaryMemory = R"doc()doc";

static const char *__doc_kp_Tensor_mFreeStagingBuffer = R"doc()doc";

static const char *__doc_kp_Tensor_mFreeStagingMemory = R"doc()doc";

static const char *__doc_kp_Tensor_mPhysicalDevice = R"doc()doc";

static const char *__doc_kp_Tensor_mPrimaryBuffer = R"doc()doc";

static const char *__doc_kp_Tensor_mPrimaryMemory = R"doc()doc";

static const char *__doc_kp_Tensor_mRawData = R"doc()doc";

static const char *__doc_kp_Tensor_mSize = R"doc()doc";

static const char *__doc_kp_Tensor_mStagingBuffer = R"doc()doc";

static const char *__doc_kp_Tensor_mStagingMemory = R"doc()doc";

static const char *__doc_kp_Tensor_mTensorType = R"doc()doc";

static const char *__doc_kp_Tensor_mapRawData = R"doc()doc";

static const char *__doc_kp_Tensor_memorySize = R"doc()doc";

static const char *__doc_kp_Tensor_rawData = R"doc()doc";

static const char *__doc_kp_Tensor_rebuild =
R"doc(Function to trigger reinitialisation of the tensor buffer and memory
with new data as well as new potential device type.

@param data Vector of data to use to initialise vector from @param
tensorType The type to use for the tensor)doc";

static const char *__doc_kp_Tensor_recordBufferMemoryBarrier =
R"doc(Records the buffer memory barrier into the command buffer which
ensures that relevant data transfers are carried out correctly.

@param commandBuffer Vulkan Command Buffer to record the commands into
@param srcAccessMask Access flags for source access mask @param
dstAccessMask Access flags for destination access mask @param
scrStageMask Pipeline stage flags for source stage mask @param
dstStageMask Pipeline stage flags for destination stage mask)doc";

static const char *__doc_kp_Tensor_recordCopyBuffer = R"doc()doc";

static const char *__doc_kp_Tensor_recordCopyFrom =
R"doc(Records a copy from the memory of the tensor provided to the current
thensor. This is intended to pass memory into a processing, to perform
a staging buffer transfer, or to gather output (between others).

@param commandBuffer Vulkan Command Buffer to record the commands into
@param copyFromTensor Tensor to copy the data from @param
createBarrier Whether to create a barrier that ensures the data is
copied before further operations. Default is true.)doc";

static const char *__doc_kp_Tensor_recordCopyFromDeviceToStaging =
R"doc(Records a copy from the internal device memory to the staging memory
using an optional barrier to wait for the operation. This function
would only be relevant for kp::Tensors of type eDevice.

@param commandBuffer Vulkan Command Buffer to record the commands into
@param createBarrier Whether to create a barrier that ensures the data
is copied before further operations. Default is true.)doc";

static const char *__doc_kp_Tensor_recordCopyFromStagingToDevice =
R"doc(Records a copy from the internal staging memory to the device memory
using an optional barrier to wait for the operation. This function
would only be relevant for kp::Tensors of type eDevice.

@param commandBuffer Vulkan Command Buffer to record the commands into
@param createBarrier Whether to create a barrier that ensures the data
is copied before further operations. Default is true.)doc";

static const char *__doc_kp_Tensor_setRawData =
R"doc(Sets / resets the vector data of the tensor. This function does not
perform any copies into GPU memory and is only performed on the host.)doc";

static const char *__doc_kp_Tensor_size =
R"doc(Returns the size/magnitude of the Tensor, which will be the total
number of elements across all dimensions

@return Unsigned integer representing the total number of elements)doc";

static const char *__doc_kp_Tensor_tensorType =
R"doc(Retrieve the tensor type of the Tensor

@return Tensor type of tensor)doc";

static const char *__doc_kp_Tensor_unmapRawData = R"doc()doc";

static const char *__doc_kp_Tensor_vector = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

