#include "mcp_messages.hpp"
#include <iostream>

using json = nlohmann::json;

const std::string mcp::JsonRpcVersion = "2.0";
const std::string mcp::McpVersion     = "2024-11-05";
const std::string mcp::ClientVersion  = "1.0.0";
const std::string mcp::ClientName     = "llama.cpp";

mcp::message::message(std::optional<nlohmann::json> id) : id_(std::move(id))
{
}

void mcp::message::id(std::optional<nlohmann::json> id) {
    id_ = std::move(id);
}

const std::optional<nlohmann::json> & mcp::message::id() const {
    return id_;
}

mcp::request::request(std::optional<nlohmann::json> id,
                      std::string method,
                      std::optional<nlohmann::json> params)

    : message(id), method_(std::move(method)), params_(std::move(params))
{
}

json mcp::request::toJson() const {
    json j;
    j["jsonrpc"] = JsonRpcVersion;
    j["method"] = method();
    if (id()) {
        j["id"] = id().value();
    }
    if (params()) {
        j["params"] = params().value();
    }
    return j;
}

void mcp::request::method(std::string method) {
    method_ = std::move(method);
}

const std::string & mcp::request::method() const {
    return method_;
}

void mcp::request::params(std::optional<nlohmann::json> params) {
    params_ = std::move(params);
}

const std::optional<nlohmann::json> & mcp::request::params() const {
    return params_;
}

mcp::response::response(std::optional<nlohmann::json> id,
                        std::optional<nlohmann::json> result,
                        std::optional<mcp::response::error> error)

    : message(id), result_(result), error_(error)
{
}

json mcp::response::error::toJson() const {
    json j;
    j["code"] = code;
    j["message"] = message;
    if (data) {
        j["data"] = data.value();
    }
    return j;
}

json mcp::response::toJson() const {
    json j;
    j["jsonrpc"] = JsonRpcVersion;
    if (id()) {
        j["id"] = id().value();
    }
    if (result()) {
        j["result"] = result().value();
    } else if (getError()) {
        j["error"] = getError()->toJson();
    }
    return j;
}

void mcp::response::result(std::optional<nlohmann::json> result) {
    result_ = std::move(result);
}

const std::optional<nlohmann::json> & mcp::response::result() const {
    return result_;
}

void mcp::response::setError(std::optional<mcp::response::error> error) {
    error_ = std::move(error);
}

const std::optional<mcp::response::error> & mcp::response::getError() const {
    return error_;
}

mcp::notification::notification(
    std::string method, std::optional<nlohmann::json> params)
    : message(), method_(method), params_(params)
{
}

json mcp::notification::toJson() const {
    json j;
    j["jsonrpc"] = JsonRpcVersion;
    j["method"] = method();
    if (params()) {
        j["params"] = params().value();
    }
    return j;
}

void mcp::notification::method(std::string method) {
    method_ = std::move(method);
}

const std::string & mcp::notification::method() const {
    return method_;
}

void mcp::notification::params(std::optional<nlohmann::json> params) {
    params_ = std::move(params);
}

const std::optional<nlohmann::json> & mcp::notification::params() const {
    return params_;
}

mcp::initialize_request::initialize_request(nlohmann::json id, mcp::capabilities caps)
    : request(id, "initialize"), caps_(std::move(caps))
{
     refreshParams();
}

void mcp::initialize_request::refreshParams() {
    json params;
    params["protocolVersion"] = protoVersion();
    params["clientInfo"]["name"] = name();
    params["clientInfo"]["version"] = version();
    params["capabilities"] = {};

    for (auto cap = caps_.cbegin(); cap != caps_.cend(); ++cap) {
        json cap_json;

        if (cap->subscribe) {
            cap_json["subscribe"] = true;
        }
        if (cap->listChanged) {
            cap_json["listChanged"] = true;
        }

        params["capabilities"][cap->name] = cap_json;
    }

    this->params(std::move(params));
}

void mcp::initialize_request::capabilities(mcp::capabilities caps) {
    caps_ = std::move(caps);
    refreshParams();
}

const mcp::capabilities & mcp::initialize_request::capabilities() const {
    return caps_;
}

mcp::initialize_response::initialize_response(
    nlohmann::json id, std::string name, std::string version, std::string protoVersion,
    mcp::capabilities caps)
    : response(id), name_(std::move(name)), version_(std::move(version)),
      protoVersion_(std::move(protoVersion)), caps_(std::move(caps))
{
    refreshResult();
}

void mcp::initialize_response::refreshResult() {
    json result;
    result["protocolVersion"] = protoVersion();
    result["serverInfo"]["name"] = name();
    result["serverInfo"]["version"] = version();
    result["capabilities"] = {};

    for (auto cap = caps_.cbegin(); cap != caps_.cend(); ++cap) {
        json cap_json;

        if (cap->subscribe) {
            cap_json["subscribe"] = true;
        }
        if (cap->listChanged) {
            cap_json["listChanged"] = true;
        }

        result["capabilities"][cap->name] = cap_json;
    }

    this->result(std::move(result));
}

void mcp::initialize_response::name(std::string name) {
    name_ = std::move(name);
    refreshResult();
}

const std::string & mcp::initialize_response::name() const {
    return name_;
}

void mcp::initialize_response::version(std::string version) {
    version_ = std::move(version);
    refreshResult();
}

const std::string & mcp::initialize_response::version() const {
    return version_;
}

void mcp::initialize_response::protoVersion(std::string protoVersion) {
    protoVersion_ = std::move(protoVersion);
    refreshResult();
}

const std::string & mcp::initialize_response::protoVersion() const {
    return protoVersion_;
}

void mcp::initialize_response::capabilities(mcp::capabilities caps) {
    caps_ = std::move(caps);
    refreshResult();
}

const mcp::capabilities & mcp::initialize_response::capabilities() const {
    return caps_;
}

mcp::initialize_response mcp::initialize_response::fromJson(const nlohmann::json& j) {
    std::string name = j["result"]["serverInfo"]["name"];
    std::string version = j["result"]["serverInfo"]["version"];
    std::string protoVersion = j["result"]["protocolVersion"];

    mcp::capabilities caps;
    if (j["result"].contains("capabilities")) {
        for (const auto& [key, value] : j["result"]["capabilities"].items()) {
            capability cap;
            cap.name = key;
            cap.subscribe = value.value("subscribe", false);
            cap.listChanged = value.value("listChanged", false);
            caps.push_back(cap);
        }
    }

    return initialize_response(j["id"], name, version, protoVersion, caps);
}

mcp::initialized_notification::initialized_notification()
    : notification("notifications/initialized")
{
}
