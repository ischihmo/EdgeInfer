//
// EdgeInfer - JSON Parser Implementation
//

#include "../../include/tools/json.h"

#include <fstream>
#include <stdexcept>

namespace edgeinfer {
namespace tools {

// --- Static Parsers ---

Json Json::Parse(const std::string& text) {
    return Json(nlohmann::json::parse(text));
}

Json Json::ParseFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Json::ParseFile: cannot open " + path);
    }
    return Json(nlohmann::json::parse(file));
}

// --- Serialization ---

std::string Json::Dump(int indent) const {
    return value_.dump(indent);
}

// --- Element Access ---

Json Json::operator[](const std::string& key) const {
    if (!value_.is_object() || !value_.contains(key)) {
        return Json();
    }
    return Json(value_[key]);
}

Json Json::operator[](size_t index) const {
    if (!value_.is_array() || index >= value_.size()) {
        return Json();
    }
    return Json(value_[index]);
}

// --- Type Queries ---

bool Json::IsNull() const    { return value_.is_null(); }
bool Json::IsBool() const    { return value_.is_boolean(); }
bool Json::IsNumber() const  { return value_.is_number(); }
bool Json::IsInteger() const { return value_.is_number_integer(); }
bool Json::IsString() const  { return value_.is_string(); }
bool Json::IsArray() const   { return value_.is_array(); }
bool Json::IsObject() const  { return value_.is_object(); }

// --- Key Lookup ---

bool Json::Contains(const std::string& key) const {
    return value_.is_object() && value_.contains(key);
}

bool Json::Has(const std::string& key) const {
    return Contains(key);
}

// --- Size ---

size_t Json::Size() const { return value_.size(); }
bool Json::Empty() const  { return value_.empty(); }

// --- Direct Value Access ---

std::string Json::GetString() const { return value_.get<std::string>(); }
int64_t     Json::GetInt() const    { return value_.get<int64_t>(); }
uint64_t    Json::GetUint() const   { return value_.get<uint64_t>(); }
double      Json::GetDouble() const { return value_.get<double>(); }
bool        Json::GetBool() const   { return value_.get<bool>(); }

// --- Key-based Value Access ---

std::string Json::GetString(const std::string& key, const std::string& def) const {
    return Value<std::string>(key, def);
}

int64_t Json::GetInt(const std::string& key, int64_t def) const {
    return Value<int64_t>(key, def);
}

uint64_t Json::GetUint(const std::string& key, uint64_t def) const {
    return Value<uint64_t>(key, def);
}

double Json::GetDouble(const std::string& key, double def) const {
    return Value<double>(key, def);
}

bool Json::GetBool(const std::string& key, bool def) const {
    return Value<bool>(key, def);
}

Json Json::GetObject(const std::string& key) const {
    if (value_.is_object() && value_.contains(key) && value_[key].is_object()) {
        return Json(value_[key]);
    }
    return Json();
}

Json Json::GetArray(const std::string& key) const {
    if (value_.is_object() && value_.contains(key) && value_[key].is_array()) {
        return Json(value_[key]);
    }
    return Json();
}

// --- Object Iteration ---

void Json::ForEach(std::function<void(const std::string&, const Json&)> fn) const {
    if (!value_.is_object()) return;
    for (auto it = value_.begin(); it != value_.end(); ++it) {
        fn(it.key(), Json(it.value()));
    }
}

// --- Array Items ---

std::vector<Json> Json::ArrayItems() const {
    std::vector<Json> result;
    if (!value_.is_array()) return result;
    result.reserve(value_.size());
    for (const auto& item : value_) {
        result.emplace_back(item);
    }
    return result;
}

} // namespace tools
} // namespace edgeinfer
