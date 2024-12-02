# Changelog

## [v0.8.1](https://github.com/KomputeProject/kompute/tree/v0.8.1)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.8.0...v0.8.1)

**Closed issues:**

- Discord link in README and docs is broken [\#276](https://github.com/KomputeProject/kompute/issues/276)
- Website examples typo's and 6500 XT unknown GPU [\#275](https://github.com/KomputeProject/kompute/issues/275)
- \[Question\] How to disable all log ? [\#274](https://github.com/KomputeProject/kompute/issues/274)
- full diagram 404 [\#271](https://github.com/KomputeProject/kompute/issues/271)
- Error when enabling `KOMPUTE\_ENABLE\_SPDLOG` [\#268](https://github.com/KomputeProject/kompute/issues/268)
- Add KOMPUTE\_LOG\_ACTIVE\_LEVEL instead of current SPDLOG\_ACTIVE\_LEVEL [\#267](https://github.com/KomputeProject/kompute/issues/267)
- Update/Fix Android project [\#264](https://github.com/KomputeProject/kompute/issues/264)
- Update compileSource function in examples/docs to correct one [\#261](https://github.com/KomputeProject/kompute/issues/261)
- Technically can Kompute be modified to support data visualization? [\#260](https://github.com/KomputeProject/kompute/issues/260)
- Data-transfer for Integrated GPU [\#258](https://github.com/KomputeProject/kompute/issues/258)
- Python "getting started" example fails [\#252](https://github.com/KomputeProject/kompute/issues/252)
- Python example in README doesn't work [\#248](https://github.com/KomputeProject/kompute/issues/248)
- Running Android app [\#234](https://github.com/KomputeProject/kompute/issues/234)

**Merged pull requests:**

- Added active log level definitions for kompute [\#280](https://github.com/KomputeProject/kompute/pull/280) ([axsaucedo](https://github.com/axsaucedo))
- Fix TestDestroy.TestDestroyTensorSingle [\#279](https://github.com/KomputeProject/kompute/pull/279) ([ScheissSchiesser](https://github.com/ScheissSchiesser))
- Updated discord link [\#277](https://github.com/KomputeProject/kompute/pull/277) ([axsaucedo](https://github.com/axsaucedo))
- style\(src/Algorithm\): fix typo [\#273](https://github.com/KomputeProject/kompute/pull/273) ([tpoisonooo](https://github.com/tpoisonooo))
- Fix Android Example confirmed with blog post steps [\#266](https://github.com/KomputeProject/kompute/pull/266) ([axsaucedo](https://github.com/axsaucedo))
- Adding Governance with TSC charter [\#263](https://github.com/KomputeProject/kompute/pull/263) ([axsaucedo](https://github.com/axsaucedo))
- Updating array\_mutiplication example to work correctly [\#262](https://github.com/KomputeProject/kompute/pull/262) ([axsaucedo](https://github.com/axsaucedo))
- Updated formatting [\#257](https://github.com/KomputeProject/kompute/pull/257) ([axsaucedo](https://github.com/axsaucedo))
- Fix first two python examples in the docs [\#256](https://github.com/KomputeProject/kompute/pull/256) ([lopuhin](https://github.com/lopuhin))
- Remove nonexisting "single\_include" from INSTALL\_INTERFACE [\#254](https://github.com/KomputeProject/kompute/pull/254) ([ItsBasi](https://github.com/ItsBasi))
- Added community page [\#253](https://github.com/KomputeProject/kompute/pull/253) ([axsaucedo](https://github.com/axsaucedo))
- Updated readme to reflect shader utils [\#249](https://github.com/KomputeProject/kompute/pull/249) ([axsaucedo](https://github.com/axsaucedo))
- Avoid using pointers to temporary copies of desired extensions. [\#247](https://github.com/KomputeProject/kompute/pull/247) ([ItsBasi](https://github.com/ItsBasi))

## [v0.8.0](https://github.com/KomputeProject/kompute/tree/v0.8.0) (2021-09-16)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.7.0...v0.8.0)

**Closed issues:**

- An unset KOMPUTE\_ENV\_DEBUG\_LAYERS leads KP\_LOG\_DEBUG to pass envLayerNamesVal==nullptr along to fmt, which rejects that due to "string pointer is null". [\#245](https://github.com/KomputeProject/kompute/issues/245)
- Extend utils shader helpers in test for windows [\#240](https://github.com/KomputeProject/kompute/issues/240)
- Python segfaults after import kp [\#230](https://github.com/KomputeProject/kompute/issues/230)
- Simple and extended python examples do not work \(v 0.7.0\) [\#228](https://github.com/KomputeProject/kompute/issues/228)
- Python macOS issue \(ImportError: dlopen\(...\): no suitable image found. Did find: ...: mach-o, but wrong architecture\)  [\#223](https://github.com/KomputeProject/kompute/issues/223)
- Python macOS issue \(Symbol not found: \_\_PyThreadState\_Current ... Expected in: flat namespace\) [\#221](https://github.com/KomputeProject/kompute/issues/221)
- Finalise Migration of Kompute into Linux Foundation [\#216](https://github.com/KomputeProject/kompute/issues/216)
- CMake Error: Imported target "kompute::kompute" includes non-existent path "/usr/local/single\_include" [\#212](https://github.com/KomputeProject/kompute/issues/212)
- Incompatibality inroduced with \#168 on Vulkan 1.1.x  [\#209](https://github.com/KomputeProject/kompute/issues/209)
- external  libraries [\#201](https://github.com/KomputeProject/kompute/issues/201)
- Starting slack group or discord for alternative / faster version of asking questions [\#198](https://github.com/KomputeProject/kompute/issues/198)
- Test SingleSequenceRecord is not thread safe and fails in AMD card [\#196](https://github.com/KomputeProject/kompute/issues/196)
- Update Kompute headers to reference the glslang headers for install vs build interfaces [\#193](https://github.com/KomputeProject/kompute/issues/193)
- Integrate with GLSLang find\_package file when issue is resolved in the glslang repo [\#191](https://github.com/KomputeProject/kompute/issues/191)
- Release 0.7.0 [\#187](https://github.com/KomputeProject/kompute/issues/187)
- Get number of available devices [\#185](https://github.com/KomputeProject/kompute/issues/185)
- Deep Learning Convolutional Neural Network \(CNN\) example implementation [\#162](https://github.com/KomputeProject/kompute/issues/162)
- Create example compiling and running in raspberry pi with Mesa Vulkan drivers [\#131](https://github.com/KomputeProject/kompute/issues/131)
- Add support for VK\_EXT\_debug\_utils labels [\#110](https://github.com/KomputeProject/kompute/issues/110)

**Merged pull requests:**

- Fix for null debug log causing exception in fmt lib [\#246](https://github.com/KomputeProject/kompute/pull/246) ([axsaucedo](https://github.com/axsaucedo))
- 0.8.0 Release  [\#244](https://github.com/KomputeProject/kompute/pull/244) ([axsaucedo](https://github.com/axsaucedo))
- Adding support for different types for spec and push consts [\#242](https://github.com/KomputeProject/kompute/pull/242) ([axsaucedo](https://github.com/axsaucedo))
- Extend shader helper functions in tests to support windows  [\#241](https://github.com/KomputeProject/kompute/pull/241) ([axsaucedo](https://github.com/axsaucedo))
- Increase test cov across codebase [\#239](https://github.com/KomputeProject/kompute/pull/239) ([axsaucedo](https://github.com/axsaucedo))
- Updated collab link for C++ notebook [\#237](https://github.com/KomputeProject/kompute/pull/237) ([axsaucedo](https://github.com/axsaucedo))
- Updating repo licenses and links [\#236](https://github.com/KomputeProject/kompute/pull/236) ([axsaucedo](https://github.com/axsaucedo))
- Removing GLSLang as core dependency [\#235](https://github.com/KomputeProject/kompute/pull/235) ([axsaucedo](https://github.com/axsaucedo))
- Matrix multiplication example showcasing iterative improvements in performance [\#233](https://github.com/KomputeProject/kompute/pull/233) ([Corentin-pro](https://github.com/Corentin-pro))
- Fixed typo in CMakeLists.txt \(ANDOID =\> ANDROID\) [\#232](https://github.com/KomputeProject/kompute/pull/232) ([Corentin-pro](https://github.com/Corentin-pro))
- Set kp\_debug, kp\_info, kp\_warning and kp\_error to py::none\(\) when the program terminates. [\#231](https://github.com/KomputeProject/kompute/pull/231) ([thinking-tower](https://github.com/thinking-tower))
- VGG7 Python example [\#227](https://github.com/KomputeProject/kompute/pull/227) ([20kdc](https://github.com/20kdc))
- Add documentation for CMake flags [\#224](https://github.com/KomputeProject/kompute/pull/224) ([thinking-tower](https://github.com/thinking-tower))
- Set PYTHON\_INCLUDE\_DIR and PYTHON\_LIBRARY during installation [\#222](https://github.com/KomputeProject/kompute/pull/222) ([thinking-tower](https://github.com/thinking-tower))
- Removing xxd.exe binary and add instructions to build [\#220](https://github.com/KomputeProject/kompute/pull/220) ([axsaucedo](https://github.com/axsaucedo))
- \[PYTHON\] Ensure numpy array increments refcount of tensor to keep valid  [\#219](https://github.com/KomputeProject/kompute/pull/219) ([axsaucedo](https://github.com/axsaucedo))
- Added destroy for manager [\#218](https://github.com/KomputeProject/kompute/pull/218) ([axsaucedo](https://github.com/axsaucedo))
- Revert "Fixed the issue that caused CMake to look for non-existent path after being installed" [\#217](https://github.com/KomputeProject/kompute/pull/217) ([axsaucedo](https://github.com/axsaucedo))
- Fixed the issue that caused CMake to look for non-existent path after being installed [\#213](https://github.com/KomputeProject/kompute/pull/213) ([unexploredtest](https://github.com/unexploredtest))
- omitted .data\(\) because it is incompatible with vulkan 1.1.x [\#211](https://github.com/KomputeProject/kompute/pull/211) ([unexploredtest](https://github.com/unexploredtest))
- vkEnumeratePhysicalDevices\(\*\(this-\>mInstance\) ... doesn't work on Linux i386 [\#208](https://github.com/KomputeProject/kompute/pull/208) ([unexploredtest](https://github.com/unexploredtest))
- Raises an error when having no/exceeding vulkan device's limit [\#207](https://github.com/KomputeProject/kompute/pull/207) ([unexploredtest](https://github.com/unexploredtest))
- Updated README and fixed a syntax error on C++'s example [\#206](https://github.com/KomputeProject/kompute/pull/206) ([unexploredtest](https://github.com/unexploredtest))
- removed the extra comma after KOMPUTE\_OPT\_REPO\_SUBMODULE\_BUILD [\#205](https://github.com/KomputeProject/kompute/pull/205) ([unexploredtest](https://github.com/unexploredtest))
- Extending list\_devices test for multiple devices [\#204](https://github.com/KomputeProject/kompute/pull/204) ([axsaucedo](https://github.com/axsaucedo))
- Fix \#include \<SPIRV/GlslangToSpv.h\> [\#200](https://github.com/KomputeProject/kompute/pull/200) ([unexploredtest](https://github.com/unexploredtest))
- Added memory barrier on test [\#199](https://github.com/KomputeProject/kompute/pull/199) ([axsaucedo](https://github.com/axsaucedo))
- Add function to list physical devices [\#195](https://github.com/KomputeProject/kompute/pull/195) ([axsaucedo](https://github.com/axsaucedo))
- v0.7.0 release [\#189](https://github.com/KomputeProject/kompute/pull/189) ([axsaucedo](https://github.com/axsaucedo))
- Add instructions for running on Pi4 [\#180](https://github.com/KomputeProject/kompute/pull/180) ([hpgmiskin](https://github.com/hpgmiskin))

## [v0.7.0](https://github.com/KomputeProject/kompute/tree/v0.7.0) (2021-03-14)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.6.0...v0.7.0)

**Implemented enhancements:**

- Extend non-spdlog print functions to use std::format [\#158](https://github.com/KomputeProject/kompute/issues/158)
- Add code coverage reports with codecov [\#145](https://github.com/KomputeProject/kompute/issues/145)
- Explore removing `std::vector mData;` completely from Tensor in favour of always storing data in hostVisible buffer memory \(TBC\) [\#144](https://github.com/KomputeProject/kompute/issues/144)
- Update all examples to match breaking changes in 0.7.0 [\#141](https://github.com/KomputeProject/kompute/issues/141)
- Avoid copy when returning python numpy / array [\#139](https://github.com/KomputeProject/kompute/issues/139)
- Cover all Python & C++ tests in CI  [\#121](https://github.com/KomputeProject/kompute/issues/121)
- Add C++ Test for Simple Work Groups Example [\#117](https://github.com/KomputeProject/kompute/issues/117)
- Expose push constants in OpAlgo [\#54](https://github.com/KomputeProject/kompute/issues/54)
- Expose ability to create barriers in OpTensor operations [\#45](https://github.com/KomputeProject/kompute/issues/45)
- Create delete function in manager to free / destroy sequence [\#36](https://github.com/KomputeProject/kompute/issues/36)
- Make specialisation data extensible [\#12](https://github.com/KomputeProject/kompute/issues/12)
- Support multiple types for Kompute Tensors [\#2](https://github.com/KomputeProject/kompute/issues/2)
- Added re-record sequence functionality and updated docs [\#171](https://github.com/KomputeProject/kompute/pull/171) ([axsaucedo](https://github.com/axsaucedo))
- Extend non-spdlog print functions to use fmt::format / fmt::print [\#159](https://github.com/KomputeProject/kompute/pull/159) ([axsaucedo](https://github.com/axsaucedo))
- Added support for custom SpecializedConstants and removed KomputeWorkgroup class [\#151](https://github.com/KomputeProject/kompute/pull/151) ([axsaucedo](https://github.com/axsaucedo))
- Added destroy functions for tensors and sequences \(named and object\) [\#146](https://github.com/KomputeProject/kompute/pull/146) ([axsaucedo](https://github.com/axsaucedo))

**Fixed bugs:**

- push\_constant not working in my case? [\#168](https://github.com/KomputeProject/kompute/issues/168)
- DescriptorPool set is not being freed [\#155](https://github.com/KomputeProject/kompute/issues/155)
- Updated memory barriers to include staging buffers [\#182](https://github.com/KomputeProject/kompute/pull/182) ([axsaucedo](https://github.com/axsaucedo))
- Adds push const ranges in pipelinelayout to fix \#168 [\#174](https://github.com/KomputeProject/kompute/pull/174) ([axsaucedo](https://github.com/axsaucedo))
- Added destructor for staging tensors [\#134](https://github.com/KomputeProject/kompute/pull/134) ([axsaucedo](https://github.com/axsaucedo))

**Closed issues:**

- Add ability to specify whether to build shared or static dependencies as well as option for Kompute lib [\#190](https://github.com/KomputeProject/kompute/issues/190)
- Update memory barriers to align with tensor staging/primary memory revamp [\#181](https://github.com/KomputeProject/kompute/issues/181)
- Move shader defaultResource inside kp::Shader class [\#175](https://github.com/KomputeProject/kompute/issues/175)
- Reach at least 90% code coverage on tests [\#170](https://github.com/KomputeProject/kompute/issues/170)
- Add functionality to re-record sequence as now it's possible to update the underlying algorithm [\#169](https://github.com/KomputeProject/kompute/issues/169)
- Use numpy arrays as default return value [\#166](https://github.com/KomputeProject/kompute/issues/166)
- Update all shared\_ptr value passes to be by ref or const ref [\#161](https://github.com/KomputeProject/kompute/issues/161)
- Amend memory hierarchy for kp::Operations so they can be created separately [\#160](https://github.com/KomputeProject/kompute/issues/160)
- Customise theme of documentation [\#156](https://github.com/KomputeProject/kompute/issues/156)
- Remove KomputeWorkgroup class in favour of std::array\<uint32\_t, 3\> [\#152](https://github.com/KomputeProject/kompute/issues/152)
- Passing raw GLSL string to Shader Module depricated so remove this method from supported approach [\#150](https://github.com/KomputeProject/kompute/issues/150)
- Add python backwards compatibility for eval\_tensor\_create\_def [\#147](https://github.com/KomputeProject/kompute/issues/147)
- Document breaking changes for 0.7.0 [\#140](https://github.com/KomputeProject/kompute/issues/140)
- Tensor memory management and memory hierarchy redesign [\#136](https://github.com/KomputeProject/kompute/issues/136)
- Staging tensor GPU memory is not freed as part of OpCreateTensor removal [\#133](https://github.com/KomputeProject/kompute/issues/133)
- eStorage Tensors are currently unusable as OpTensorCreate calls mapDataIntoHostMemory [\#132](https://github.com/KomputeProject/kompute/issues/132)
- 0.6.0 Release [\#126](https://github.com/KomputeProject/kompute/issues/126)
- java.lang.UnsatisfiedLinkError: dlopen failed: library "libkompute-jni.so" not found [\#125](https://github.com/KomputeProject/kompute/issues/125)
- Initial exploration: Include explicit GLSL to SPIRV compilation [\#107](https://github.com/KomputeProject/kompute/issues/107)
- Add support for push constants [\#106](https://github.com/KomputeProject/kompute/issues/106)

**Merged pull requests:**

- Resolve moving all functions from tensor HPP to CPP [\#186](https://github.com/KomputeProject/kompute/pull/186) ([axsaucedo](https://github.com/axsaucedo))
- Device Properties [\#184](https://github.com/KomputeProject/kompute/pull/184) ([alexander-g](https://github.com/alexander-g))
- Too many warnings [\#183](https://github.com/KomputeProject/kompute/pull/183) ([alexander-g](https://github.com/alexander-g))
- Add support for bool, double, int32, uint32 and float32 on Tensors via TensorT [\#177](https://github.com/KomputeProject/kompute/pull/177) ([axsaucedo](https://github.com/axsaucedo))
- Support for Timestamping [\#176](https://github.com/KomputeProject/kompute/pull/176) ([alexander-g](https://github.com/alexander-g))
- Test for ShaderResources [\#165](https://github.com/KomputeProject/kompute/pull/165) ([unexploredtest](https://github.com/unexploredtest))
- Amend memory hierarchy to enable for push constants and functional interface for more flexible operations [\#164](https://github.com/KomputeProject/kompute/pull/164) ([axsaucedo](https://github.com/axsaucedo))
- made changes for include paths for complete installation [\#163](https://github.com/KomputeProject/kompute/pull/163) ([unexploredtest](https://github.com/unexploredtest))
- Added dark mode on docs [\#157](https://github.com/KomputeProject/kompute/pull/157) ([axsaucedo](https://github.com/axsaucedo))
- Glslang implementation for online shader compilation [\#154](https://github.com/KomputeProject/kompute/pull/154) ([axsaucedo](https://github.com/axsaucedo))
- Adding test code coverage using gcov and lcov [\#149](https://github.com/KomputeProject/kompute/pull/149) ([axsaucedo](https://github.com/axsaucedo))
- Added temporary backwards compatibility for eval\_tensor\_create\_def function [\#148](https://github.com/KomputeProject/kompute/pull/148) ([axsaucedo](https://github.com/axsaucedo))
- Amend memory ownership hierarchy to have Tensor owned by Manager instead of OpCreateTensor / OpBase [\#138](https://github.com/KomputeProject/kompute/pull/138) ([axsaucedo](https://github.com/axsaucedo))
- Removed Staging Tensors in favour of having two buffer & memory in a Tensor to minimise data transfer [\#137](https://github.com/KomputeProject/kompute/pull/137) ([axsaucedo](https://github.com/axsaucedo))

## [v0.6.0](https://github.com/KomputeProject/kompute/tree/v0.6.0) (2021-01-31)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.5.1...v0.6.0)

**Implemented enhancements:**

- Add simple test for Python `log\_level` function [\#120](https://github.com/KomputeProject/kompute/issues/120)
- Add further numpy support [\#104](https://github.com/KomputeProject/kompute/issues/104)
- SWIG syntax error - change order of keywords. [\#94](https://github.com/KomputeProject/kompute/issues/94)
- Create mocks to isolate unit tests for components [\#8](https://github.com/KomputeProject/kompute/issues/8)
- Disallowing zero sized tensors [\#129](https://github.com/KomputeProject/kompute/pull/129) ([alexander-g](https://github.com/alexander-g))
- Added further tests to CI and provide Dockerimage with builds to swiftshader [\#119](https://github.com/KomputeProject/kompute/pull/119) ([axsaucedo](https://github.com/axsaucedo))
- Workgroups for Python [\#116](https://github.com/KomputeProject/kompute/pull/116) ([alexander-g](https://github.com/alexander-g))
- Ubuntu CI [\#115](https://github.com/KomputeProject/kompute/pull/115) ([alexander-g](https://github.com/alexander-g))
- Faster set\_data\(\) [\#109](https://github.com/KomputeProject/kompute/pull/109) ([alexander-g](https://github.com/alexander-g))
- String parameter for eval\_algo\_str methods in Python [\#105](https://github.com/KomputeProject/kompute/pull/105) ([alexander-g](https://github.com/alexander-g))
- Added numpy\(\) method [\#103](https://github.com/KomputeProject/kompute/pull/103) ([alexander-g](https://github.com/alexander-g))

**Fixed bugs:**

- \[PYTHON\] Support string parameter instead of list for eval\_algo\_data when passing raw shader as string [\#93](https://github.com/KomputeProject/kompute/issues/93)
- \[PYTHON\] Fix log\_level on the python implementation \(using pybind's logging functions\) [\#92](https://github.com/KomputeProject/kompute/issues/92)

**Closed issues:**

- Add documentation for custom operations [\#128](https://github.com/KomputeProject/kompute/issues/128)
- Numpy Array Support and Work Group Configuration in Python Kompute [\#124](https://github.com/KomputeProject/kompute/issues/124)
- Remove references to spdlog in python module [\#122](https://github.com/KomputeProject/kompute/issues/122)
- Setup automated CI testing for PRs using GitHub actions [\#114](https://github.com/KomputeProject/kompute/issues/114)
- Python example type error \(pyshader\). [\#111](https://github.com/KomputeProject/kompute/issues/111)
- Update all references to operations to not use template [\#101](https://github.com/KomputeProject/kompute/issues/101)
- Getting a undefined reference error while creating a Kompute Manager [\#100](https://github.com/KomputeProject/kompute/issues/100)

**Merged pull requests:**

- 122 remove spdlog references in python [\#123](https://github.com/KomputeProject/kompute/pull/123) ([axsaucedo](https://github.com/axsaucedo))
- Native logging for Python [\#118](https://github.com/KomputeProject/kompute/pull/118) ([alexander-g](https://github.com/alexander-g))
- Fixes for the c++ Simple and Extended examples in readme [\#108](https://github.com/KomputeProject/kompute/pull/108) ([unexploredtest](https://github.com/unexploredtest))
- Fix building shaders on native linux [\#102](https://github.com/KomputeProject/kompute/pull/102) ([unexploredtest](https://github.com/unexploredtest))

## [v0.5.1](https://github.com/KomputeProject/kompute/tree/v0.5.1) (2020-11-12)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.5.0...v0.5.1)

**Implemented enhancements:**

- Remove the template params from OpAlgoBase for dispatch layout [\#57](https://github.com/KomputeProject/kompute/issues/57)
- Enable layout to be configured dynamically within shaders [\#26](https://github.com/KomputeProject/kompute/issues/26)
- replaced "static unsigned const" to "static const unsigned" to avoid SWIG parsing error. [\#95](https://github.com/KomputeProject/kompute/pull/95) ([0x0f0f0f](https://github.com/0x0f0f0f))

**Closed issues:**

- Support for MoltenVK? [\#96](https://github.com/KomputeProject/kompute/issues/96)
- Update all examples to use spir-v bytes by default [\#86](https://github.com/KomputeProject/kompute/issues/86)

**Merged pull requests:**

- Python extensions for end to end example [\#97](https://github.com/KomputeProject/kompute/pull/97) ([axsaucedo](https://github.com/axsaucedo))

## [v0.5.0](https://github.com/KomputeProject/kompute/tree/v0.5.0) (2020-11-08)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.4.1...v0.5.0)

**Implemented enhancements:**

- Adding Python package for Kompute [\#87](https://github.com/KomputeProject/kompute/issues/87)
- Python shader extension [\#91](https://github.com/KomputeProject/kompute/pull/91) ([axsaucedo](https://github.com/axsaucedo))
- Added python bindings with kp as python module  [\#88](https://github.com/KomputeProject/kompute/pull/88) ([axsaucedo](https://github.com/axsaucedo))

**Closed issues:**

- Examples segfault \(Linux / mesa / amdgpu\) [\#84](https://github.com/KomputeProject/kompute/issues/84)
- Kompute support for newer Vulkan HPP headers [\#81](https://github.com/KomputeProject/kompute/issues/81)

## [v0.4.1](https://github.com/KomputeProject/kompute/tree/v0.4.1) (2020-11-01)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.4.0...v0.4.1)

**Implemented enhancements:**

- Upgrade build to support VulkanHPP 1.2.154 \< 1.2.158 [\#82](https://github.com/KomputeProject/kompute/issues/82)
- Add Android example for Kompute [\#23](https://github.com/KomputeProject/kompute/issues/23)
- Enhanced python build [\#89](https://github.com/KomputeProject/kompute/pull/89) ([axsaucedo](https://github.com/axsaucedo))
- Fix compatibility for Vulkan HPP 1.2.155 and above [\#83](https://github.com/KomputeProject/kompute/pull/83) ([axsaucedo](https://github.com/axsaucedo))
- codespell spelling fixes [\#80](https://github.com/KomputeProject/kompute/pull/80) ([pH5](https://github.com/pH5))

**Closed issues:**

- Android example throws runtime error.  [\#77](https://github.com/KomputeProject/kompute/issues/77)
- Document the utilities to convert shaders into C++ header files [\#53](https://github.com/KomputeProject/kompute/issues/53)
- Document the three types of memory ownership in classes - never, optional and always [\#31](https://github.com/KomputeProject/kompute/issues/31)

**Merged pull requests:**

- Add link to official Vulkan website to download the SDK [\#79](https://github.com/KomputeProject/kompute/pull/79) ([DonaldWhyte](https://github.com/DonaldWhyte))
- 77 Fix end to end examples by creating tensors on separate sequence [\#78](https://github.com/KomputeProject/kompute/pull/78) ([axsaucedo](https://github.com/axsaucedo))

## [v0.4.0](https://github.com/KomputeProject/kompute/tree/v0.4.0) (2020-10-18)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.3.2...v0.4.0)

**Implemented enhancements:**

- Error compiling on ubuntu 20.04 [\#67](https://github.com/KomputeProject/kompute/issues/67)
- Add explicit multi-threading interfaces to ensure correctness when running in parallel [\#51](https://github.com/KomputeProject/kompute/issues/51)

**Fixed bugs:**

- Ensure sequences are cleared when begin is run [\#74](https://github.com/KomputeProject/kompute/issues/74)

**Merged pull requests:**

- 74 Fixing manager default sequence creation [\#75](https://github.com/KomputeProject/kompute/pull/75) ([axsaucedo](https://github.com/axsaucedo))
- Adding Asynchronous Processing Capabilities with Multiple Queue Support [\#73](https://github.com/KomputeProject/kompute/pull/73) ([axsaucedo](https://github.com/axsaucedo))
- Fix README typo [\#71](https://github.com/KomputeProject/kompute/pull/71) ([nihui](https://github.com/nihui))

## [v0.3.2](https://github.com/KomputeProject/kompute/tree/v0.3.2) (2020-10-04)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.3.1...v0.3.2)

**Implemented enhancements:**

- Fix compiler errors on compilers other than msvc [\#66](https://github.com/KomputeProject/kompute/pull/66) ([Dudecake](https://github.com/Dudecake))

**Fixed bugs:**

- Fix bug in OpAlgoRhsLhs [\#61](https://github.com/KomputeProject/kompute/issues/61)

**Closed issues:**

- Change c++ to 14 from 17 for support with older frameworks [\#59](https://github.com/KomputeProject/kompute/issues/59)

**Merged pull requests:**

- Updated readme and single kompute for 0.3.2 [\#69](https://github.com/KomputeProject/kompute/pull/69) ([axsaucedo](https://github.com/axsaucedo))
- Added android example and upgraded build configurations [\#68](https://github.com/KomputeProject/kompute/pull/68) ([axsaucedo](https://github.com/axsaucedo))
- Added readme to explain high level explanation for Godot example [\#65](https://github.com/KomputeProject/kompute/pull/65) ([axsaucedo](https://github.com/axsaucedo))
- Removing vulkan dependencies in examples [\#64](https://github.com/KomputeProject/kompute/pull/64) ([axsaucedo](https://github.com/axsaucedo))
- Updated godot example to use logistic regression usecase [\#63](https://github.com/KomputeProject/kompute/pull/63) ([axsaucedo](https://github.com/axsaucedo))

## [v0.3.1](https://github.com/KomputeProject/kompute/tree/v0.3.1) (2020-09-20)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.3.0...v0.3.1)

**Implemented enhancements:**

- Add example of how vulkan kompute can be used for ML in Godot Game Engine [\#60](https://github.com/KomputeProject/kompute/issues/60)

**Merged pull requests:**

- Adding godot example [\#62](https://github.com/KomputeProject/kompute/pull/62) ([axsaucedo](https://github.com/axsaucedo))

## [v0.3.0](https://github.com/KomputeProject/kompute/tree/v0.3.0) (2020-09-19)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/0.3.0...v0.3.0)

**Implemented enhancements:**

- Make Kompute installable locally to work with examples [\#58](https://github.com/KomputeProject/kompute/issues/58)
- Remove ability to copy output parameters from OpAlgoBase now that there's OpTensorSync [\#56](https://github.com/KomputeProject/kompute/issues/56)

## [0.3.0](https://github.com/KomputeProject/kompute/tree/0.3.0) (2020-09-13)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.2.0...0.3.0)

**Implemented enhancements:**

- Add tests and documentation for loops passing data to/from device [\#50](https://github.com/KomputeProject/kompute/issues/50)
- Add preSubmit function to OpBase to account for multiple eval commands in parallel [\#47](https://github.com/KomputeProject/kompute/issues/47)
- Remove vulkan commandbuffer from Tensor [\#42](https://github.com/KomputeProject/kompute/issues/42)
- Provide further granularity on handling staging tensors [\#40](https://github.com/KomputeProject/kompute/issues/40)
- Create operation to copy data from local to device memory with staging [\#39](https://github.com/KomputeProject/kompute/issues/39)
- Add more advanced ML implementations \(starting with LR, then DL, etc\) [\#19](https://github.com/KomputeProject/kompute/issues/19)

**Fixed bugs:**

- OpCreateTensor doesn't map data into GPU with OpCreateTensor for host tensors [\#43](https://github.com/KomputeProject/kompute/issues/43)

## [v0.2.0](https://github.com/KomputeProject/kompute/tree/v0.2.0) (2020-09-05)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/v0.1.0...v0.2.0)

**Implemented enhancements:**

- Migrate to GTest  [\#37](https://github.com/KomputeProject/kompute/issues/37)
- Move all todos in the code into github issues [\#33](https://github.com/KomputeProject/kompute/issues/33)
- Remove spdlog as a required dependency [\#30](https://github.com/KomputeProject/kompute/issues/30)
- Improve access to tensor underlying data for speed and ease of access [\#18](https://github.com/KomputeProject/kompute/issues/18)
- Enable for compute shaders to be provided in raw form [\#17](https://github.com/KomputeProject/kompute/issues/17)
- Enable OpCreateTensor for more than 1 tensor  [\#13](https://github.com/KomputeProject/kompute/issues/13)
- Add specialisation data to algorithm with default tensor size [\#11](https://github.com/KomputeProject/kompute/issues/11)
- Add documentation with Doxygen and Sphinx [\#9](https://github.com/KomputeProject/kompute/issues/9)

**Fixed bugs:**

- Diagnose memory profiling to ensure there are no memory leaks on objects created  \[CPU\] [\#15](https://github.com/KomputeProject/kompute/issues/15)

**Merged pull requests:**

- Migrating to gtest [\#38](https://github.com/KomputeProject/kompute/pull/38) ([axsaucedo](https://github.com/axsaucedo))

## [v0.1.0](https://github.com/KomputeProject/kompute/tree/v0.1.0) (2020-08-28)

[Full Changelog](https://github.com/KomputeProject/kompute/compare/2879d3d274967e87087d567bcc659804b1707d0a...v0.1.0)



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
