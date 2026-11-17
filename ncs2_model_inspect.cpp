// Offline IR inspection. This binary reads the OpenVINO model graph only;
// it never opens the USB device or compiles for MYRIAD.
#include <openvino/openvino.hpp>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
constexpr size_t tensor_limit = 16384;

std::string json_quote(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : text) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec << std::setfill(' ');
                } else {
                    out << c;
                }
        }
    }
    return out << '"', out.str();
}

std::string json_string_array(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << json_quote(values[i]);
    }
    return out << ']', out.str();
}

std::string shape_json(const ov::Shape& shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    return out << ']', out.str();
}

size_t element_count(const ov::Shape& shape) {
    if (shape.empty() || shape.size() > 8)
        throw std::runtime_error("Tensor rank must be 1..8.");
    size_t result = 1;
    for (const auto dimension : shape) {
        if (dimension == 0 || result > tensor_limit / dimension)
            throw std::runtime_error("Tensor exceeds 16384 elements.");
        result *= dimension;
    }
    return result;
}

struct TensorInfo {
    size_t index;
    std::string name;
    std::string dtype;
    ov::PartialShape partial_shape;
    bool static_shape;
    size_t elements = 0;
    bool element_limit_ok = false;
};

TensorInfo describe(size_t index, const ov::Output<ov::Node>& port) {
    TensorInfo info{index, port.get_any_name(), port.get_element_type().get_type_name(),
                    port.get_partial_shape(), port.get_partial_shape().is_static()};
    if (info.static_shape) {
        try {
            info.elements = element_count(info.partial_shape.to_shape());
            info.element_limit_ok = true;
        } catch (const std::exception&) {
            info.element_limit_ok = false;
        }
    }
    return info;
}

std::string tensor_json(const TensorInfo& info) {
    std::ostringstream out;
    out << "{\"index\":" << info.index
        << ",\"name\":" << json_quote(info.name)
        << ",\"dtype\":" << json_quote(info.dtype)
        << ",\"static\":" << (info.static_shape ? "true" : "false")
        << ",\"rank\":";
    if (info.partial_shape.rank().is_static()) out << info.partial_shape.rank().get_length();
    else out << "null";
    out << ",\"shape\":";
    if (info.static_shape) out << shape_json(info.partial_shape.to_shape());
    else out << "null";
    out << ",\"elements\":";
    if (info.static_shape && info.element_limit_ok) out << info.elements;
    else out << "null";
    out << ",\"element_limit_ok\":" << (info.element_limit_ok ? "true" : "false") << '}';
    return out.str();
}

bool base_tensor_support(const TensorInfo& info) {
    return info.dtype == "f32" && info.static_shape && info.partial_shape.rank().is_static()
        && info.partial_shape.rank().get_length() >= 1
        && info.partial_shape.rank().get_length() <= 8 && info.element_limit_ok;
}

std::string compatibility_json(const std::vector<TensorInfo>& inputs,
                               const std::vector<TensorInfo>& outputs,
                               bool schema_one) {
    std::vector<std::string> reasons;
    if (schema_one && (inputs.size() != 1 || outputs.size() != 1))
        reasons.emplace_back("requires exactly one input and one output");
    if (inputs.empty() || inputs.size() > 32 || outputs.empty() || outputs.size() > 32)
        reasons.emplace_back("requires 1..32 inputs and outputs");
    size_t input_elements = 0, output_elements = 0;
    for (const auto& info : inputs) {
        if (!base_tensor_support(info)) reasons.emplace_back("all inputs must be static f32 tensors with rank 1..8 and at most 16384 elements");
        if (info.element_limit_ok) input_elements += info.elements;
    }
    for (const auto& info : outputs) {
        if (!base_tensor_support(info)) reasons.emplace_back("all outputs must be static f32 tensors with rank 1..8 and at most 16384 elements");
        if (info.element_limit_ok) output_elements += info.elements;
    }
    if (input_elements > tensor_limit) reasons.emplace_back("aggregate input elements exceed 16384");
    if (output_elements > tensor_limit) reasons.emplace_back("aggregate output elements exceed 16384");
    std::ostringstream out;
    out << "{\"compatible\":" << (reasons.empty() ? "true" : "false")
        << ",\"reasons\":" << json_string_array(reasons) << '}';
    return out.str();
}

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--model")
            throw std::runtime_error("Usage: ncs2_model_inspect --model XML");
        const fs::path xml = fs::path(argv[2]);
        ov::Core core;
        const auto model = core.read_model(xml.string());
        std::vector<TensorInfo> inputs, outputs;
        for (size_t i = 0; i < model->inputs().size(); ++i)
            inputs.push_back(describe(i, model->input(i)));
        for (size_t i = 0; i < model->outputs().size(); ++i)
            outputs.push_back(describe(i, model->output(i)));
        const auto version = ov::get_openvino_version();
        const std::string openvino_build = version.buildNumber ? version.buildNumber : "unknown";
        const auto version_separator = openvino_build.find('-');
        const std::string openvino_version = openvino_build.substr(0, version_separator);
        std::cout << "{\"schema_version\":1,\"type\":\"model_inspection\","
                  << "\"hardware_accessed\":false,\"openvino_version\":"
                  << json_quote(openvino_version) << ",\"openvino_build\":"
                  << json_quote(openvino_build) << ",\"model\":{\"xml\":"
                  << json_quote(xml.string()) << ",\"inputs\":[";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << tensor_json(inputs[i]);
        }
        std::cout << "],\"outputs\":[";
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << tensor_json(outputs[i]);
        }
        std::cout << "],\"input_count\":" << inputs.size()
                  << ",\"output_count\":" << outputs.size() << "},\"compatibility\":{"
                  << "\"schema_1\":" << compatibility_json(inputs, outputs, true)
                  << ",\"schema_2\":" << compatibility_json(inputs, outputs, false)
                  << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model-inspect: " << error.what() << '\n';
        return 1;
    }
}
