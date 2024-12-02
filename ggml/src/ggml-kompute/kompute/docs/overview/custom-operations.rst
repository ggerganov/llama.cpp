
Extending Kompute with Custom C++ Operations
=================

Kompute provides an extenisble architecture which allows for the core components to be extended by building custom operations.

Building operations is intuitive however it requires knowing some nuances around the order in which each of the class functions across the operation are called as a sequence is executed.

These nuances are important for more advanced users of Kompute, as this will provide further intuition in what are the specific functions and components that the native functions (like OpTensorCreate, OpAlgoBase, etc) contain which define their specific behaviour.

Flow of Function Calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The top level operation which all operations inherit from is the :class:`kp::OpBase` class. Some of the "Core Native Operations" like :class:`kp::OpTensorCopy`, :class:`kp::OpTensorCreate`, etc all inherit from the base operation class.

The `kp::OpAlgoBase` is another base operation that is specifically built to enable users to create their own operations that contain custom shader logic (i.e. requiring Compute Pipelines, DescriptorSets, etc). The next section contains an example which shows how to extend the OpAlgoBase class.

Below you 

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - OpBase(..., tensors, freeTensors)
     - Constructor for class where you can load/define resources such as shaders, etc.
   * - ~OpBase()
     - Destructor that frees GPU resources (if owned) which should be used to manage any memory allocations created through the operation.
   * - init()
     - Init function gets called in the Sequence / Manager inside the record step. This function allows for relevant objects to be initialised within the operation.
   * - record()
     - Record function that gets called in the Sequence / Manager inside the record step after init(). In this function you can directly record to the vk::CommandBuffer.
   * - preEval()
     - When the Sequence is Evaluated this preEval is called across all operations before dispatching the batch of recorded commands to the GPU. This is useful for example if you need to copy data from local to host memory.
   * - postEval()
     - After the sequence is Evaluated this postEval is called across all operations. When running asynchronously the postEval is called when you call `evalAwait()`, which is why it's important to always run evalAwait() to ensure the process doesn't go into inconsistent state.


Simple Operation Extending OpAlgoBase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can find an example in the `Advanced Examples documentation section <advanced-examples.rst>`_ that shows how to create your own custom function.

You can also see an implementation in the codebase through the `OpMult` class:


.. literalinclude:: ../../src/include/kompute/operations/OpMult.hpp
   :language: cpp


Then the implementation outlines all the implementations that perform the actions above:
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../src/OpMult.cpp
   :language: cpp


