#pragma once

#include <string>
#include <vector>

#include "ggml-qnn.h"

#include "logger.hpp"
#include "qnn-lib.hpp"
#include "qnn-types.hpp"
#include "tensor.hpp"

namespace qnn {
class ggml_qnn_op_config {
public:
    explicit ggml_qnn_op_config(const std::string &name, const std::string &package_name, const std::string &op_type) :
        _name(name), _package_name(package_name), _op_type(op_type) {}

    void set_input_tensors(const std::vector<std::shared_ptr<ggml_qnn_tensor>> &tensor_inputs) {
        _qnn_tensor_inputs.resize(tensor_inputs.size());
        for (size_t i = 0; i < tensor_inputs.size(); i++) {
            _qnn_tensor_inputs[i] = tensor_inputs[i]->get_qnn_tensor();
        }
    }

    void set_output_tensors(const std::vector<std::shared_ptr<ggml_qnn_tensor>> &tensor_outputs) {
        _qnn_tensor_outputs.resize(tensor_outputs.size());
        for (size_t i = 0; i < tensor_outputs.size(); i++) {
            _qnn_tensor_outputs[i] = tensor_outputs[i]->get_qnn_tensor();
        }
    }

    void add_scalar_param(const std::string &name, const Qnn_Scalar_t scalar) {
        _param_names.push_back(name);
        Qnn_Param_t param = QNN_PARAM_INIT;
        param.paramType = QNN_PARAMTYPE_SCALAR;
        param.name = _param_names.back().c_str();
        param.scalarParam = scalar;
        _parameters.push_back(param);
    }

    std::vector<Qnn_Tensor_t> &get_qnn_input_tensors() { return _qnn_tensor_inputs; }
    std::vector<Qnn_Tensor_t> &get_qnn_output_tensors() { return _qnn_tensor_outputs; }

    Qnn_OpConfig_t get_op_config() {
        Qnn_OpConfig_t config = QNN_OPCONFIG_INIT;
        config.version = QNN_OPCONFIG_VERSION_1;
        auto &op_config = config.v1;
        op_config.name = _name.c_str();
        op_config.packageName = _package_name.c_str();
        op_config.typeName = _op_type.c_str();
        op_config.numOfParams = (uint32_t)_parameters.size();
        op_config.params = _parameters.data();
        op_config.numOfInputs = (uint32_t)_qnn_tensor_inputs.size();
        op_config.inputTensors = _qnn_tensor_inputs.data();
        op_config.numOfOutputs = (uint32_t)_qnn_tensor_outputs.size();
        op_config.outputTensors = _qnn_tensor_outputs.data();
        return config;
    }

private:
    std::string _name;
    std::string _package_name;
    std::string _op_type;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_outputs;
    std::vector<Qnn_Param_t> _parameters;
    std::vector<std::string> _param_names;

    DISABLE_COPY(ggml_qnn_op_config);
    DISABLE_MOVE(ggml_qnn_op_config);
};
} // namespace qnn
