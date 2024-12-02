#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include <kompute/Kompute.hpp>

#include "docstrings.hpp"
#include "utils.hpp"

namespace py = pybind11;

// used in Core.hpp
py::object kp_trace, kp_debug, kp_info, kp_warning, kp_error;

std::unique_ptr<kp::OpAlgoDispatch>
opAlgoDispatchPyInit(std::shared_ptr<kp::Algorithm>& algorithm,
                     const py::array& push_consts)
{
    const py::buffer_info info = push_consts.request();
    KP_LOG_DEBUG("Kompute Python Manager creating tensor_T with push_consts "
                 "size {} dtype {}",
                 push_consts.size(),
                 std::string(py::str(push_consts.dtype())));

    if (push_consts.dtype().is(py::dtype::of<std::float_t>())) {
        std::vector<float> dataVec((float*)info.ptr,
                                   ((float*)info.ptr) + info.size);
        return std::unique_ptr<kp::OpAlgoDispatch>{ new kp::OpAlgoDispatch(
          algorithm, dataVec) };
    } else if (push_consts.dtype().is(py::dtype::of<std::uint32_t>())) {
        std::vector<uint32_t> dataVec((uint32_t*)info.ptr,
                                      ((uint32_t*)info.ptr) + info.size);
        return std::unique_ptr<kp::OpAlgoDispatch>{ new kp::OpAlgoDispatch(
          algorithm, dataVec) };
    } else if (push_consts.dtype().is(py::dtype::of<std::int32_t>())) {
        std::vector<int32_t> dataVec((int32_t*)info.ptr,
                                     ((int32_t*)info.ptr) + info.size);
        return std::unique_ptr<kp::OpAlgoDispatch>{ new kp::OpAlgoDispatch(
          algorithm, dataVec) };
    } else if (push_consts.dtype().is(py::dtype::of<std::double_t>())) {
        std::vector<double> dataVec((double*)info.ptr,
                                    ((double*)info.ptr) + info.size);
        return std::unique_ptr<kp::OpAlgoDispatch>{ new kp::OpAlgoDispatch(
          algorithm, dataVec) };
    } else {
        throw std::runtime_error("Kompute Python no valid dtype supported");
    }
}

PYBIND11_MODULE(kp, m)
{

    // The logging modules are used in the Kompute.hpp file
    py::module_ logging = py::module_::import("logging");
    py::object kp_logger = logging.attr("getLogger")("kp");
    kp_trace = kp_logger.attr(
      "debug"); // Same as for debug since python has no trace logging level
    kp_debug = kp_logger.attr("debug");
    kp_info = kp_logger.attr("info");
    kp_warning = kp_logger.attr("warning");
    kp_error = kp_logger.attr("error");
    logging.attr("basicConfig")();

    py::module_ np = py::module_::import("numpy");

    py::enum_<kp::Tensor::TensorTypes>(m, "TensorTypes")
      .value("device",
             kp::Tensor::TensorTypes::eDevice,
             DOC(kp, Tensor, TensorTypes, eDevice))
      .value("host",
             kp::Tensor::TensorTypes::eHost,
             DOC(kp, Tensor, TensorTypes, eHost))
      .value("storage",
             kp::Tensor::TensorTypes::eStorage,
             DOC(kp, Tensor, TensorTypes, eStorage))
      .export_values();

    py::class_<kp::OpBase, std::shared_ptr<kp::OpBase>>(
      m, "OpBase", DOC(kp, OpBase));

    py::class_<kp::OpTensorSyncDevice,
               kp::OpBase,
               std::shared_ptr<kp::OpTensorSyncDevice>>(
      m, "OpTensorSyncDevice", DOC(kp, OpTensorSyncDevice))
      .def(py::init<const std::vector<std::shared_ptr<kp::Tensor>>&>(),
           DOC(kp, OpTensorSyncDevice, OpTensorSyncDevice));

    py::class_<kp::OpTensorSyncLocal,
               kp::OpBase,
               std::shared_ptr<kp::OpTensorSyncLocal>>(
      m, "OpTensorSyncLocal", DOC(kp, OpTensorSyncLocal))
      .def(py::init<const std::vector<std::shared_ptr<kp::Tensor>>&>(),
           DOC(kp, OpTensorSyncLocal, OpTensorSyncLocal));

    py::class_<kp::OpTensorCopy, kp::OpBase, std::shared_ptr<kp::OpTensorCopy>>(
      m, "OpTensorCopy", DOC(kp, OpTensorCopy))
      .def(py::init<const std::vector<std::shared_ptr<kp::Tensor>>&>(),
           DOC(kp, OpTensorCopy, OpTensorCopy));

    py::class_<kp::OpAlgoDispatch,
               kp::OpBase,
               std::shared_ptr<kp::OpAlgoDispatch>>(
      m, "OpAlgoDispatch", DOC(kp, OpAlgoDispatch))
      .def(py::init<const std::shared_ptr<kp::Algorithm>&,
                    const std::vector<float>&>(),
           DOC(kp, OpAlgoDispatch, OpAlgoDispatch),
           py::arg("algorithm"),
           py::arg("push_consts") = std::vector<float>())
      .def(py::init(&opAlgoDispatchPyInit),
           DOC(kp, OpAlgoDispatch, OpAlgoDispatch),
           py::arg("algorithm"),
           py::arg("push_consts"));

    py::class_<kp::OpMult, kp::OpBase, std::shared_ptr<kp::OpMult>>(
      m, "OpMult", DOC(kp, OpMult))
      .def(py::init<const std::vector<std::shared_ptr<kp::Tensor>>&,
                    const std::shared_ptr<kp::Algorithm>&>(),
           DOC(kp, OpMult, OpMult));

    py::class_<kp::Algorithm, std::shared_ptr<kp::Algorithm>>(
      m, "Algorithm", DOC(kp, Algorithm, Algorithm))
      .def("get_tensors",
           &kp::Algorithm::getTensors,
           DOC(kp, Algorithm, getTensors))
      .def("destroy", &kp::Algorithm::destroy, DOC(kp, Algorithm, destroy))
      .def("is_init", &kp::Algorithm::isInit, DOC(kp, Algorithm, isInit));

    py::class_<kp::Tensor, std::shared_ptr<kp::Tensor>>(
      m, "Tensor", DOC(kp, Tensor))
      .def(
        "data",
        [](kp::Tensor& self) {
            // Non-owning container exposing the underlying pointer
            switch (self.dataType()) {
                case kp::Tensor::TensorDataTypes::eFloat:
                    return py::array(
                      self.size(), self.data<float>(), py::cast(&self));
                case kp::Tensor::TensorDataTypes::eUnsignedInt:
                    return py::array(
                      self.size(), self.data<uint32_t>(), py::cast(&self));
                case kp::Tensor::TensorDataTypes::eInt:
                    return py::array(
                      self.size(), self.data<int32_t>(), py::cast(&self));
                case kp::Tensor::TensorDataTypes::eDouble:
                    return py::array(
                      self.size(), self.data<double>(), py::cast(&self));
                case kp::Tensor::TensorDataTypes::eBool:
                    return py::array(
                      self.size(), self.data<bool>(), py::cast(&self));
                default:
                    throw std::runtime_error(
                      "Kompute Python data type not supported");
            }
        },
        DOC(kp, Tensor, data))
      .def("size", &kp::Tensor::size, DOC(kp, Tensor, size))
      .def("__len__", &kp::Tensor::size, DOC(kp, Tensor, size))
      .def("tensor_type", &kp::Tensor::tensorType, DOC(kp, Tensor, tensorType))
      .def("data_type", &kp::Tensor::dataType, DOC(kp, Tensor, dataType))
      .def("is_init", &kp::Tensor::isInit, DOC(kp, Tensor, isInit))
      .def("destroy", &kp::Tensor::destroy, DOC(kp, Tensor, destroy));

    py::class_<kp::Sequence, std::shared_ptr<kp::Sequence>>(m, "Sequence")
      .def(
        "record",
        [](kp::Sequence& self, std::shared_ptr<kp::OpBase> op) {
            return self.record(op);
        },
        DOC(kp, Sequence, record))
      .def(
        "eval",
        [](kp::Sequence& self) { return self.eval(); },
        DOC(kp, Sequence, eval))
      .def(
        "eval",
        [](kp::Sequence& self, std::shared_ptr<kp::OpBase> op) {
            return self.eval(op);
        },
        DOC(kp, Sequence, eval_2))
      .def(
        "eval_async",
        [](kp::Sequence& self) { return self.eval(); },
        DOC(kp, Sequence, evalAwait))
      .def(
        "eval_async",
        [](kp::Sequence& self, std::shared_ptr<kp::OpBase> op) {
            return self.evalAsync(op);
        },
        DOC(kp, Sequence, evalAsync))
      .def(
        "eval_await",
        [](kp::Sequence& self) { return self.evalAwait(); },
        DOC(kp, Sequence, evalAwait))
      .def(
        "eval_await",
        [](kp::Sequence& self, uint32_t wait) { return self.evalAwait(wait); },
        DOC(kp, Sequence, evalAwait))
      .def("is_recording",
           &kp::Sequence::isRecording,
           DOC(kp, Sequence, isRecording))
      .def("is_running", &kp::Sequence::isRunning, DOC(kp, Sequence, isRunning))
      .def("is_init", &kp::Sequence::isInit, DOC(kp, Sequence, isInit))
      .def("clear", &kp::Sequence::clear, DOC(kp, Sequence, clear))
      .def("rerecord", &kp::Sequence::rerecord, DOC(kp, Sequence, rerecord))
      .def("get_timestamps",
           &kp::Sequence::getTimestamps,
           DOC(kp, Sequence, getTimestamps))
      .def("destroy", &kp::Sequence::destroy, DOC(kp, Sequence, destroy));

    py::class_<kp::Manager, std::shared_ptr<kp::Manager>>(
      m, "Manager", DOC(kp, Manager))
      .def(py::init(), DOC(kp, Manager, Manager))
      .def(py::init<uint32_t>(), DOC(kp, Manager, Manager_2))
      .def(py::init<uint32_t,
                    const std::vector<uint32_t>&,
                    const std::vector<std::string>&>(),
           DOC(kp, Manager, Manager_2),
           py::arg("device") = 0,
           py::arg("family_queue_indices") = std::vector<uint32_t>(),
           py::arg("desired_extensions") = std::vector<std::string>())
      .def("destroy", &kp::Manager::destroy, DOC(kp, Manager, destroy))
      .def("sequence",
           &kp::Manager::sequence,
           DOC(kp, Manager, sequence),
           py::arg("queue_index") = 0,
           py::arg("total_timestamps") = 0)
      .def(
        "tensor",
        [np](kp::Manager& self,
             const py::array_t<float>& data,
             kp::Tensor::TensorTypes tensor_type) {
            const py::array_t<float>& flatdata = np.attr("ravel")(data);
            const py::buffer_info info = flatdata.request();
            KP_LOG_DEBUG("Kompute Python Manager tensor() creating tensor "
                         "float with data size {}",
                         flatdata.size());
            return self.tensor(info.ptr,
                               flatdata.size(),
                               sizeof(float),
                               kp::Tensor::TensorDataTypes::eFloat,
                               tensor_type);
        },
        DOC(kp, Manager, tensor),
        py::arg("data"),
        py::arg("tensor_type") = kp::Tensor::TensorTypes::eDevice)
      .def(
        "tensor_t",
        [np](kp::Manager& self,
             const py::array& data,
             kp::Tensor::TensorTypes tensor_type) {
            // TODO: Suppport strides in numpy format
            const py::array& flatdata = np.attr("ravel")(data);
            const py::buffer_info info = flatdata.request();
            KP_LOG_DEBUG("Kompute Python Manager creating tensor_T with data "
                         "size {} dtype {}",
                         flatdata.size(),
                         std::string(py::str(flatdata.dtype())));
            if (flatdata.dtype().is(py::dtype::of<std::float_t>())) {
                return self.tensor(info.ptr,
                                   flatdata.size(),
                                   sizeof(float),
                                   kp::Tensor::TensorDataTypes::eFloat,
                                   tensor_type);
            } else if (flatdata.dtype().is(py::dtype::of<std::uint32_t>())) {
                return self.tensor(info.ptr,
                                   flatdata.size(),
                                   sizeof(uint32_t),
                                   kp::Tensor::TensorDataTypes::eUnsignedInt,
                                   tensor_type);
            } else if (flatdata.dtype().is(py::dtype::of<std::int32_t>())) {
                return self.tensor(info.ptr,
                                   flatdata.size(),
                                   sizeof(int32_t),
                                   kp::Tensor::TensorDataTypes::eInt,
                                   tensor_type);
            } else if (flatdata.dtype().is(py::dtype::of<std::double_t>())) {
                return self.tensor(info.ptr,
                                   flatdata.size(),
                                   sizeof(double),
                                   kp::Tensor::TensorDataTypes::eDouble,
                                   tensor_type);
            } else if (flatdata.dtype().is(py::dtype::of<bool>())) {
                return self.tensor(info.ptr,
                                   flatdata.size(),
                                   sizeof(bool),
                                   kp::Tensor::TensorDataTypes::eBool,
                                   tensor_type);
            } else {
                throw std::runtime_error(
                  "Kompute Python no valid dtype supported");
            }
        },
        DOC(kp, Manager, tensorT),
        py::arg("data"),
        py::arg("tensor_type") = kp::Tensor::TensorTypes::eDevice)
      .def(
        "algorithm",
        [](kp::Manager& self,
           const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
           const py::bytes& spirv,
           const kp::Workgroup& workgroup,
           const std::vector<float>& spec_consts,
           const std::vector<float>& push_consts) {
            py::buffer_info info(py::buffer(spirv).request());
            const char* data = reinterpret_cast<const char*>(info.ptr);
            size_t length = static_cast<size_t>(info.size);
            std::vector<uint32_t> spirvVec((uint32_t*)data,
                                           (uint32_t*)(data + length));
            return self.algorithm(
              tensors, spirvVec, workgroup, spec_consts, push_consts);
        },
        DOC(kp, Manager, algorithm),
        py::arg("tensors"),
        py::arg("spirv"),
        py::arg("workgroup") = kp::Workgroup(),
        py::arg("spec_consts") = std::vector<float>(),
        py::arg("push_consts") = std::vector<float>())
      .def(
        "algorithm",
        [np](kp::Manager& self,
             const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
             const py::bytes& spirv,
             const kp::Workgroup& workgroup,
             const py::array& spec_consts,
             const py::array& push_consts) {
            py::buffer_info info(py::buffer(spirv).request());
            const char* data = reinterpret_cast<const char*>(info.ptr);
            size_t length = static_cast<size_t>(info.size);
            std::vector<uint32_t> spirvVec((uint32_t*)data,
                                           (uint32_t*)(data + length));

            const py::buffer_info pushInfo = push_consts.request();
            const py::buffer_info specInfo = spec_consts.request();

            KP_LOG_DEBUG("Kompute Python Manager creating Algorithm_T with "
                         "push consts data size {} dtype {} and spec const "
                         "data size {} dtype {}",
                         push_consts.size(),
                         std::string(py::str(push_consts.dtype())),
                         spec_consts.size(),
                         std::string(py::str(spec_consts.dtype())));

            // We have to iterate across a combination of parameters due to the
            // lack of support for templating
            if (spec_consts.dtype().is(py::dtype::of<std::float_t>())) {
                std::vector<float> specConstsVec(
                  (float*)specInfo.ptr, ((float*)specInfo.ptr) + specInfo.size);
                if (spec_consts.dtype().is(py::dtype::of<std::float_t>())) {
                    std::vector<float> pushConstsVec((float*)pushInfo.ptr,
                                                     ((float*)pushInfo.ptr) +
                                                       pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specConstsVec,
                                          pushConstsVec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::int32_t>())) {
                    std::vector<int32_t> pushConstsVec(
                      (int32_t*)pushInfo.ptr,
                      ((int32_t*)pushInfo.ptr) + pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specConstsVec,
                                          pushConstsVec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::uint32_t>())) {
                    std::vector<uint32_t> pushConstsVec(
                      (uint32_t*)pushInfo.ptr,
                      ((uint32_t*)pushInfo.ptr) + pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specConstsVec,
                                          pushConstsVec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::double_t>())) {
                    std::vector<double> pushConstsVec((double*)pushInfo.ptr,
                                                      ((double*)pushInfo.ptr) +
                                                        pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specConstsVec,
                                          pushConstsVec);
                }
            } else if (spec_consts.dtype().is(py::dtype::of<std::int32_t>())) {
                std::vector<int32_t> specconstsvec((int32_t*)specInfo.ptr,
                                                   ((int32_t*)specInfo.ptr) +
                                                     specInfo.size);
                if (spec_consts.dtype().is(py::dtype::of<std::float_t>())) {
                    std::vector<float> pushconstsvec((float*)pushInfo.ptr,
                                                     ((float*)pushInfo.ptr) +
                                                       pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::int32_t>())) {
                    std::vector<int32_t> pushconstsvec(
                      (int32_t*)pushInfo.ptr,
                      ((int32_t*)pushInfo.ptr) + pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::uint32_t>())) {
                    std::vector<uint32_t> pushconstsvec(
                      (uint32_t*)pushInfo.ptr,
                      ((uint32_t*)pushInfo.ptr) + pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::double_t>())) {
                    std::vector<double> pushconstsvec((double*)pushInfo.ptr,
                                                      ((double*)pushInfo.ptr) +
                                                        pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                }
            } else if (spec_consts.dtype().is(py::dtype::of<std::uint32_t>())) {
                std::vector<uint32_t> specconstsvec((uint32_t*)specInfo.ptr,
                                                    ((uint32_t*)specInfo.ptr) +
                                                      specInfo.size);
                if (spec_consts.dtype().is(py::dtype::of<std::float_t>())) {
                    std::vector<float> pushconstsvec((float*)pushInfo.ptr,
                                                     ((float*)pushInfo.ptr) +
                                                       pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::int32_t>())) {
                    std::vector<int32_t> pushconstsvec(
                      (int32_t*)pushInfo.ptr,
                      ((int32_t*)pushInfo.ptr) + pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::uint32_t>())) {
                    std::vector<uint32_t> pushconstsvec(
                      (uint32_t*)pushInfo.ptr,
                      ((uint32_t*)pushInfo.ptr) + pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::double_t>())) {
                    std::vector<double> pushconstsvec((double*)pushInfo.ptr,
                                                      ((double*)pushInfo.ptr) +
                                                        pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                }
            } else if (spec_consts.dtype().is(py::dtype::of<std::double_t>())) {
                std::vector<double> specconstsvec((double*)specInfo.ptr,
                                                  ((double*)specInfo.ptr) +
                                                    specInfo.size);
                if (spec_consts.dtype().is(py::dtype::of<std::float_t>())) {
                    std::vector<float> pushconstsvec((float*)pushInfo.ptr,
                                                     ((float*)pushInfo.ptr) +
                                                       pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::int32_t>())) {
                    std::vector<float> pushconstsvec((int32_t*)pushInfo.ptr,
                                                     ((int32_t*)pushInfo.ptr) +
                                                       pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::uint32_t>())) {
                    std::vector<float> pushconstsvec((uint32_t*)pushInfo.ptr,
                                                     ((uint32_t*)pushInfo.ptr) +
                                                       pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                } else if (spec_consts.dtype().is(
                             py::dtype::of<std::double_t>())) {
                    std::vector<float> pushconstsvec((double*)pushInfo.ptr,
                                                     ((double*)pushInfo.ptr) +
                                                       pushInfo.size);
                    return self.algorithm(tensors,
                                          spirvVec,
                                          workgroup,
                                          specconstsvec,
                                          pushconstsvec);
                }
            }
            // If reach then no valid dtype supported
            throw std::runtime_error("Kompute Python no valid dtype supported");
        },
        DOC(kp, Manager, algorithm),
        py::arg("tensors"),
        py::arg("spirv"),
        py::arg("workgroup") = kp::Workgroup(),
        py::arg("spec_consts") = std::vector<float>(),
        py::arg("push_consts") = std::vector<float>())
      .def(
        "list_devices",
        [](kp::Manager& self) {
            const std::vector<vk::PhysicalDevice> devices = self.listDevices();
            py::list list;
            for (const vk::PhysicalDevice& device : devices) {
                list.append(kp::py::vkPropertiesToDict(device.getProperties()));
            }
            return list;
        },
        "Return a dict containing information about the device")
      .def(
        "get_device_properties",
        [](kp::Manager& self) {
            const vk::PhysicalDeviceProperties properties =
              self.getDeviceProperties();

            return kp::py::vkPropertiesToDict(properties);
        },
        "Return a dict containing information about the device");

    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        kp_trace = py::none();
        kp_debug = py::none();
        kp_info = py::none();
        kp_warning = py::none();
        kp_error = py::none();
    }));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
