//
// EdgeInfer - JSON Parser (nlohmann/json wrapper)
//
// The underlying library is at third_party/nlohmann/include/json.hpp.
// Include path: add -I third_party/nlohmann/include to your build.
//

#ifndef EDGEINFER_JSON_H
#define EDGEINFER_JSON_H

#include <json.hpp>

#include <string>
#include <vector>
#include <functional>

namespace edgeinfer {
namespace tools {

// ============================================================
// Json - wrapper around nlohmann::json
// ============================================================

class Json {
public:
    // --- Constructors ---
    Json() = default;
    explicit Json(nlohmann::json value) : value_(std::move(value)) {}

    // --- Static Parsers ---
    static Json Parse(const std::string& text);
    static Json ParseFile(const std::string& path);

    // --- Serialization ---
    std::string Dump(int indent = -1) const;

    // --- Element Access ---
    Json operator[](const std::string& key) const;
    Json operator[](size_t index) const;

    // --- Type Queries ---
    bool IsNull() const;
    bool IsBool() const;
    bool IsNumber() const;
    bool IsInteger() const;
    bool IsString() const;
    bool IsArray() const;
    bool IsObject() const;

    // --- Key Lookup ---
    bool Contains(const std::string& key) const;
    bool Has(const std::string& key) const;

    // --- Size ---
    size_t Size() const;
    bool Empty() const;

    // --- Value Extraction (templated, must stay inline) ---
    template<typename T>
    T Get() const { return value_.get<T>(); }

    template<typename T>
    T Value(const std::string& key, T default_val) const {
        if (!value_.is_object() || !value_.contains(key)) return default_val;
        try { return value_[key].get<T>(); }
        catch (...) { return default_val; }
    }

    template<typename T>
    T Value(size_t index, T default_val) const {
        if (!value_.is_array() || index >= value_.size()) return default_val;
        try { return value_[index].get<T>(); }
        catch (...) { return default_val; }
    }

    // --- Direct Value Access ---
    std::string GetString() const;
    int64_t     GetInt() const;
    uint64_t    GetUint() const;
    double      GetDouble() const;
    bool        GetBool() const;

    // --- Key-based Value Access ---
    std::string GetString(const std::string& key, const std::string& def = "") const;
    int64_t     GetInt(const std::string& key, int64_t def = 0) const;
    uint64_t    GetUint(const std::string& key, uint64_t def = 0) const;
    double      GetDouble(const std::string& key, double def = 0.0) const;
    bool        GetBool(const std::string& key, bool def = false) const;

    // --- Sub-object / Sub-array ---
    Json GetObject(const std::string& key) const;
    Json GetArray(const std::string& key) const;

    // --- Iteration ---
    void ForEach(std::function<void(const std::string&, const Json&)> fn) const;
    std::vector<Json> ArrayItems() const;

    // --- Raw access (escape hatch) ---
    nlohmann::json& Raw() { return value_; }
    const nlohmann::json& Raw() const { return value_; }

private:
    nlohmann::json value_;
};

} // namespace tools
} // namespace edgeinfer

#endif // EDGEINFER_JSON_H
