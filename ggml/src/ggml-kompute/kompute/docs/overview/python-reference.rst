

Python Class Documentation & Reference
========

This section provides a breakdown of the Python classes and what each of their functions provide.
Below is a diagram that provides insights on the relationship between Kompute objects and Vulkan SDK resources, which primarily encompass ownership of either CPU and/or GPU memory.

Manager
-------

The Kompute Manager provides a high level interface to simplify interaction with underlying :class:`kp.Sequence` of Operations.

.. autoclass:: kp.Manager
   :members:


Sequence
-------

The Kompute Sequence consists of batches of Kompute Operations, which are executed on a respective GPU queue. The execution of sequences can be synchronous or asynchronous, and it can be coordinated through its respective vk::Fence.

.. autoclass:: kp.Sequence
   :members:


Tensor
-------

The Kompute Tensor is the atomic unit in Kompute, and it is used primarily for handling Host and GPU Device data.

.. autoclass:: kp.Tensor
   :members:


TensorType
-------

.. automodule:: kp
   :members:

