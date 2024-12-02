.. role:: raw-html-m2r(raw)
   :format: html


C++ Examples
=================

The power of Kompute comes in when the interface is used for complex computations. This section contains an outline of the advanced / end-to-end examples available.

Simple examples
^^^^^^^^^^^^^^^

* `Create your custom Kompute Operations <#your-custom-kompute-operation>`_
* `Run Asynchronous Operations <#asynchronous-operations>`_
* `Run Parallel Operations Across Multiple GPU Queues <#parallel-operations>`_

End-to-end examples
^^^^^^^^^^^^^^^^^^^


* `Machine Learning Logistic Regression Implementation <https://towardsdatascience.com/machine-learning-and-data-processing-in-the-gpu-with-vulkan-kompute-c9350e5e5d3a>`_
* `Parallelizing GPU-intensive Workloads via Multi-Queue Operations <https://towardsdatascience.com/parallelizing-heavy-gpu-workloads-via-multi-queue-operations-50a38b15a1dc>`_
* `Android NDK Mobile Kompute ML Application <https://towardsdatascience.com/gpu-accelerated-machine-learning-in-your-mobile-applications-using-the-android-ndk-vulkan-kompute-1e9da37b7617>`_
* `Game Development Kompute ML in Godot Engine <https://towardsdatascience.com/supercharging-game-development-with-gpu-accelerated-ml-using-vulkan-kompute-the-godot-game-engine-4e75a84ea9f0>`_

Add Extensions
^^^^^^^^^^^^^^^^^^^^

Kompute provides a simple way to add extensions through kp::Manager initialisation. When debug is enabled you will be able to see logs that show what are the desired extensions requested and the ones that are added based on the available extensions on the current driver.

The example below shows how you can enable the "VK_EXT_shader_atomic_float" extension so we can use the adomicAdd for floats in the shaders.

.. code-block:: cpp
   :linenos:

   int main() {
       std::string shader(R"(
             #version 450

             #extension GL_EXT_shader_atomic_float: enable

             layout(push_constant) uniform PushConstants {
               float x;
               float y;
               float z;
             } pcs;

             layout (local_size_x = 1) in;

             layout(set = 0, binding = 0) buffer a { float pa[]; };

             void main() {
                 atomicAdd(pa[0], pcs.x);
                 atomicAdd(pa[1], pcs.y);
                 atomicAdd(pa[2], pcs.z);
             })");

       // See shader documentation section for compileSource
       std::vector<uint32_t> spirv = compileSource(shader);

       std::shared_ptr<kp::Sequence> sq = nullptr;

       {
           kp::Manager mgr(0, {}, { "VK_EXT_shader_atomic_float" });

           std::shared_ptr<kp::Tensor> tensor = mgr.tensor({ 0, 0, 0 });

           std::shared_ptr<kp::Algorithm> algo =
             mgr.algorithm({ tensor }, spirv, kp::Workgroup({ 1 }), {}, { 0.0, 0.0, 0.0 });

           sq = mgr.sequence()
                  ->record<kp::OpTensorSyncDevice>({ tensor })
                  ->record<kp::OpAlgoDispatch>(algo,
                                               std::vector<float>{ 0.1, 0.2, 0.3 })
                  ->record<kp::OpAlgoDispatch>(algo,
                                               std::vector<float>{ 0.3, 0.2, 0.1 })
                  ->record<kp::OpTensorSyncLocal>({ tensor })
                  ->eval();

           EXPECT_EQ(tensor->data(), std::vector<float>({ 0.4, 0.4, 0.4 }));
       }
   }


Your Custom Kompute Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build your own pre-compiled operations for domain specific workflows. Back to `examples list <#simple-examples>`_

We also provide tools that allow you to `convert shaders into C++ headers <https://github.com/KomputeProject/kompute/blob/master/scripts/convert_shaders.py#L40>`_.

.. code-block:: cpp
   :linenos:

   class OpMyCustom : public OpAlgoDispatch
   {
     public:
       OpMyCustom(std::vector<std::shared_ptr<Tensor>> tensors,
            std::shared_ptr<kp::Algorithm> algorithm)
         : OpAlgoBase(algorithm)
       {
            if (tensors.size() != 3) {
                throw std::runtime_error("Kompute OpMult expected 3 tensors but got " + tensors.size());
            }

           // See shader documentation section for compileSource
            std::vector<uint32_t> spirv = compileSource(R"(
                #version 450

                layout(set = 0, binding = 0) buffer tensorLhs {
                   float valuesLhs[ ];
                };

                layout(set = 0, binding = 1) buffer tensorRhs {
                   float valuesRhs[ ];
                };

                layout(set = 0, binding = 2) buffer tensorOutput {
                   float valuesOutput[ ];
                };

                layout (constant_id = 0) const uint LEN_LHS = 0;
                layout (constant_id = 1) const uint LEN_RHS = 0;
                layout (constant_id = 2) const uint LEN_OUT = 0;

                layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

                void main() 
                {
                    uint index = gl_GlobalInvocationID.x;

                    valuesOutput[index] = valuesLhs[index] * valuesRhs[index];
                }
            )");

            algorithm->rebuild(tensors, spirv);
       }
   }


   int main() {

       kp::Manager mgr; // Automatically selects Device 0

       // Create 3 tensors of default type float
       auto tensorLhs = mgr.tensor({ 0., 1., 2. });
       auto tensorRhs = mgr.tensor({ 2., 4., 6. });
       auto tensorOut = mgr.tensor({ 0., 0., 0. });

       mgr.sequence()
            ->record<kp::OpTensorSyncDevice>({tensorLhs, tensorRhs, tensorOut})
            ->record<kp::OpMyCustom>({tensorLhs, tensorRhs, tensorOut}, mgr.algorithm())
            ->record<kp::OpTensorSyncLocal>({tensorLhs, tensorRhs, tensorOut})
            ->eval();

       // Prints the output which is { 0, 4, 12 }
       std::cout << fmt::format("Output: {}", tensorOutput.data()) << std::endl;
   }

Async/Await Example
^^^^^^^^^^^^^^^^^^^^^

A simple example of asynchronous submission can be found below.

First we are able to create the manager as we normally would.

.. code-block:: cpp
    :linenos:

    // You can allow Kompute to create the GPU resources, or pass your existing ones
    kp::Manager mgr; // Selects device 0 unless explicitly requested

    // Creates tensor an initializes GPU memory (below we show more granularity)
    auto tensor = mgr.tensor(10, 0.0);

We can now run our first asynchronous command, which in this case we can use the default sequence.

Sequences can be executed in synchronously or asynchronously without having to change anything.

.. code-block:: cpp
    :linenos:

    // Create tensors data explicitly in GPU with an operation
    mgr.sequence()->eval<kp::OpTensorSyncDevice>({tensor});


While this is running we can actually do other things like in this case create the shader we'll be using.

In this case we create a shader that should take a couple of milliseconds to run.

.. code-block:: cpp
    :linenos:

    // Define your shader as a string (using string literals for simplicity)
    // (You can also pass the raw compiled bytes, or even path to file)
    std::string shader(R"(
        #version 450

        layout (local_size_x = 1) in;

        layout(set = 0, binding = 0) buffer b { float pb[]; };

        shared uint sharedTotal[1];

        void main() {
            uint index = gl_GlobalInvocationID.x;

            sharedTotal[0] = 0;

            // Iterating to simulate longer process
            for (int i = 0; i < 100000000; i++)
            {
                atomicAdd(sharedTotal[0], 1);
            }

            pb[index] = sharedTotal[0];
        }
    )");

    // See shader documentation section for compileSource
    auto algo = mgr.algorithm({tensor}, compileSource(shader));

Now we are able to run the await function on the default sequence. 

If we are using the manager, we need to make sure that we are awaiting the same named sequence that was triggered asynchronously.

If the sequence is not running or has finished running, it would return immediately.

The parameter provided is the maximum amount of time to wait in nanoseconds. When the timeout expires, the sequence would return (with false value), but it does not stop the processing in the GPU - the processing would continue as normal.

.. code-block:: cpp
    :linenos:

    auto sq = mgr.sequence()

    // Run Async Kompute operation on the parameters provided
    sq->evalAsync<kp::OpAlgoDispatch>(algo);

    // Here we can do other work

    // When we're ready we can wait 
    // The default wait time is UINT64_MAX
    sq.evalAwait()


Finally, below you can see that we can also run syncrhonous commands without having to change anything.

.. code-block:: cpp
    :linenos:

    // Sync the GPU memory back to the local tensor
    // We can still run synchronous jobs in our created sequence
    sq.eval<kp::OpTensorSyncLocal>({ tensor });

    // Prints the output: B: { 100000000, ... }
    std::cout << fmt::format("B: {}", 
        tensor.data()) << std::endl;


Parallel Operation Submission
-----------

In order to work with parallel execution of tasks, it is important that you understand some of the core GPU processing limitations, as these can be quite broad and hardware dependent, which means they will vary across NVIDIA / AMD / ETC video cards.

Conceptual Overview
^^^^^^^^^^^^^^^^^^^^^

If you are familiar with the Vulkan SDK, you will have experience that the first few things you do is fetching the physical Queues from the device. The queues themselves tend to have three main particular features - they can be GRAPHICS, TRANSFER and COMPUTE (between a few others we'll skip for simplicity).

Queues can have multiple properties - namely a queue can be of type GRAPHICS+TRANSFER+COMPUTE, etc. Now here comes the key point: the underlying hardware may (or may not) support parallelized processing at multiple levels.

Let's take a tangible example. The [NVIDIA 1650](http://vulkan.gpuinfo.org/displayreport.php?id=9700#queuefamilies) for example has 16 `GRAPHICS+TRANSFER+COMPUTE` queues on `familyIndex 0`, then 2 `TRANSFER` queues in `familyIndex 1` and finally 8 `COMPUTE+TRANSFER` queues in `familyIndex 2`.

With this in mind, the NVIDIA 1650 as of today does not support intra-family parallelization, which means that if you were to submit commands in multiple queues of the same family, these would still be exectured synchronously. 

However the NVIDIA 1650 does support inter-family parallelization, which means that if we were to submit commands across multiple queues from different families, these would execute in parallel.

This means that we would be able to execute parallel workloads as long as we're running them across multiple queue families. This is one of the reasons why Kompute enables users to explicitly select the underlying queues and queue families to run particular workloads on.

It is important that you understand what are the capabilities and limitations of your hardware, as parallelization capabilities can vary, so you will want to make sure you account for potential discrepancies in processing structures, mainyl to avoid undesired/unexpected race conditions.

Parallel Execution Example
^^^^^^^^^^^^^^^^^^^^^

In this example we will demonstrate how you can set up parallel processing across two compute families to achieve 2x speedups when running processing workloads.

To start, you will see that we do have to create the manager with extra parameters. This includes the GPU device index we want to use, together with the array of the queues that we want to enable.

In this case we are using only two queues, which as per the section above, these would be familyIndex 0 which is of type `GRAPHICS+COMPUTE+TRANSFER` and familyIndex 2 which is of type `COMPUTE+TRANSFER`.

In this case based on the specifications of the NVIDIA 1650 we could define up to 16 graphics queues (familyIndex 0), 2 transfer queues (familyIndex 1), and 8 compute queues (familyIndex 2) in no particular order. This means that we could have something like `{ 0, 1, 1, 2, 2, 2, 0, ... }` as our initialization value.

You will want to keep track of the indices you initialize your manager, as you will be referring back to this ordering when creating sequences with particular queues.

.. code-block:: cpp
    :linenos:

       // In this case we select device 0, and for queues, one queue from familyIndex 0
       // and one queue from familyIndex 2
       uint32_t deviceIndex(0);
       std::vector<uint32_t> familyIndices = {0, 2};

       // We create a manager with device index, and queues by queue family index
       kp::Manager mgr(deviceIndex, familyIndices);


We are now able to create sequences with a particular queue. 

By default the Kompute Manager is created with device 0, and with a single queue of the first compatible familyIndex. Similarly, by default sequences are created with the first available queue.

In this case we are able to specify which queue we want to use. Below we initialize "queueOne" named sequence with the graphics family queue, and "queueTwo" with the compute family queue.

It's worth mentioning you can have multiple sequences referencing the same queue.

.. code-block:: cpp
    :linenos:

       // We need to create explicit sequences with their respective queues
       // The second parameter is the index in the familyIndex array which is relative
       //      to the vector we created the manager with.
       sqOne = mgr.sequence(0);
       sqTwo = mgr.sequence(1);

We create the tensors without modifications.

.. code-block:: cpp
    :linenos:

       // Creates tensor an initializes GPU memory (below we show more granularity)
       auto tensorA = mgr.tensor({ 10, 0.0 });
       auto tensorB = mgr.tensor({ 10, 0.0 });

       // Copies the data into GPU memory
       mgr.sequence().eval<kp::OpTensorSyncDevice>({tensorA tensorB});

Similar to the asyncrhonous usecase above, we can still run synchronous commands without modifications.

.. code-block:: cpp
    :linenos:

       // Define your shader as a string (using string literals for simplicity)
       // (You can also pass the raw compiled bytes, or even path to file)
       std::string shader(R"(
           #version 450

           layout (local_size_x = 1) in;

           layout(set = 0, binding = 0) buffer b { float pb[]; };

           shared uint sharedTotal[1];

           void main() {
               uint index = gl_GlobalInvocationID.x;

               sharedTotal[0] = 0;

               // Iterating to simulate longer process
               for (int i = 0; i < 100000000; i++)
               {
                   atomicAdd(sharedTotal[0], 1);
               }

               pb[index] = sharedTotal[0];
           }
       )");

       // See shader documentation section for compileSource
       std::vector<uint32_t> spirv = compileSource(shader);

       std::shared_ptr<kp::Algorithm> algo = mgr.algorithm({tensorA, tenssorB}, spirv);

Now we can actually trigger the parallel processing, running two OpAlgoBase Operations - each in a different sequence / queue.

.. code-block:: cpp
    :linenos:

       // Run the first parallel operation in the `queueOne` sequence
       sqOne->evalAsync<kp::OpAlgoDispatch>(algo);

       // Run the second parallel operation in the `queueTwo` sequence
       sqTwo->evalAsync<kp::OpAlgoDispatch>(algo);


Similar to the asynchronous example above, we are able to do other work whilst the tasks are executing.

We are able to wait for the tasks to complete by triggering the `evalOpAwait` on the respective sequence.

.. code-block:: cpp
    :linenos:

       // Here we can do other work

       // We can now wait for the two parallel tasks to finish
       sqOne.evalOpAwait()
       sqTwo.evalOpAwait()

       // Sync the GPU memory back to the local tensor
       mgr.sequence()->eval<kp::OpTensorSyncLocal>({ tensorA, tensorB });

       // Prints the output: A: 100000000 B: 100000000
       std::cout << fmt::format("A: {}, B: {}", 
           tensorA.data()[0], tensorB.data()[0]) << std::endl;


