
Memory Management Principles
=====================

The principle in Kompute on memory management is summarised as follows:

* Memory management by Kompute is optional and only in place if resource is created by Kompute
* Memory management ownership architecture are acyclic and with a single top manager
* Operations do not manage any GPU memory or resources
* Top level manager is main owner of GPU resources and removes all resources when destroyed
* Manager holds weak pointers to ensure that if object created outside is destroyed it's released
* Once a resource is destroyed it cannot be recreated
* Resources can only be rebuilt if they haven't been destroyed

Kompute is responsible for managing both the CPU and GPU memory allocations and resources that it creates, and is important that they are able to explicitly define when these objects are released or destroyed. Similarly, it's important that the memory resources created by the application are released safely.

Kompute is built with the BYOV principle in mind (Bring your own VulkanSDK). This means that even though the top level resources are managing the memory to its owned resources, they themselves may not have full ownership of the GPU / Vulkan SDK components - this is in the case that you may want to use Kompute with an existing Vulkan SDK enabled application, and may want to initialise Kompute components with existing Vulkan SDK resources.

The memory ownership is hierarchically outlined in the component architecture - in this diagram, the arrows provide an intuition on the memory management ownership relationships. It's worth mentioning that the memory relationship may be different to the way components interact with each other - for this, you can see the high level component overview. More specifically:
* The purple arrows denote GPU memory management

.. image:: ../images/kompute-vulkan-architecture.jpg
   :width: 100%

Optional Memory Management
-------------

As outlined above, resource memory is only managed by Kompute if the resources are created by Kompute. Each of the Kompute components can also be initialised with externally managed resources. The :class:`kp::Manager` for example can be initialized with an external vk::Device. The first principle ensures that all memory ownership is explicitly defined when managing and creating Kompute resources.



