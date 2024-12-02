
Variable Types for Tensors and Constants
=============

By default the initial interfaces you may interact with, will be primarily using float values by default, which is enough to get through the basic conceptual examples. However as real world applications are being developed, more specialized types may be required for kp::Tensor, as well as for SpecializationConstants and PushConstants.

Before diving into the practical classes and interfaces that can be used to take advantage of the variable type support of Kompute, we want to provide some high level intution on what each of these components are.

Variable Tensor Types
------

For the kp::Tensor class, Kompute provides under the hood an interface to have more seamless interaction with multiple different underlying data types. This is done through the introduction of the class kp::TensorT<type> and parent class kp::Tensor, however you as a developer you will be primarily interacting with the top level kp::Tensor class, as this is what is provided through the high level kp::Manager class.

The kp::Tensor class does provide an "integrated" experience, which allows users to "seamlessly" retrieve the underlying data through the `data()` and `vector()` functions. This is done by leveraging C++ templates, as well as limiting the types that can be used, which are namely:

* float
* uint32
* int32
* double
* bool

Any other data type provided would result in an error, and for the time being Kompute will focus on primarily provide support for these classes.

The tests under `TestTensor.cpp` and `test_tensor_types.py` provide an overview of how users can take advantage of these features using std::vector for C++ and numpy array for Python.

C++ Tensor Types Usage
^^^^^^^

Below you can see how it is possible to define different types in C++.

.. literalinclude:: ../../test/TestTensor.cpp
   :language: cpp
   :lines: 21-

Python Tensor Types Usage
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../python/test/test_tensor_types.py
   :language: python
   :lines: 26-46

Variable Push Constants
----

Push constants are a relatively non-expensive way to provide dynamic data to a GPU Algorithm (shader) as further CPU compute is performed. Although Push Constants are a more efficient way to provide data, it is also a limited manner as there is a memory limit for push constants.

Push constants with Kompute are flexible as it is possible to pass user-defined structs in C++. In Python it is limited to providing numpy arrays with multiple elements of the same type.

C++ Push Consts Types Usage
^^^^^^^

As mentioned above, this test under `TestPushConstants.cpp` shows how it is possible to use user-defined structs for multiple elements from different types, which is not possible for specialized constants or tensors.

These are defined in the `algorithm` function of the `kp::Manager`, and once it push constant is set, all other push constants provided have to consist of the same types and element size.

More specifically, when passing a custom struct it is possible to pass a single element, or alternatively passing multiple scalar values as part of the vector, and access them as outlined in the rest of the tests.

.. literalinclude:: ../../test/TestPushConstant.cpp
   :language: cpp
   :lines: 182-231


Python Push Consts Types Usage
^^^^^^^^^^^^^^^^^

In python the push constants are limited to a single list of elements of the same type. These are provided by passing a numpy array to the `algorithm` function or the `kp::OpAlgoDispatch` operation.

.. literalinclude:: ../../python/test/test_tensor_types.py
   :language: python
   :lines: 207-242

Variable Specialization Constants
------

Specialization constants are analogous to push constants, but these are not dynamic, can only be set on initialization or rebuild of `kp::Algorithm` and cannot be changed unless a `rebuild` is carried out.

The usage of specailization constants is very similar to the push constants, but the only limitation are:

* These are defined using the constant_id in the glsl shader
* Spec constants do not support complex types (i.e. user defined struct)
* Kompute supports an array of elements of same type for specialization constants

C++ Push Consts Types Usage
^^^^

The specialization constant example shows how it is possible to define as a std::vector.

.. literalinclude:: ../../test/TestSpecializationConstant.cpp
   :language: cpp
   :lines: 57-


