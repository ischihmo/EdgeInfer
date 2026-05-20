//
// EdgeInfer - YAML Parser (rapidyaml wrapper)
//
// The underlying library is at third_party/rapidyaml/include/rapidyaml.hpp.
// Include path: add -I third_party/rapidyaml/include to your build.
//
// IMPORTANT: rapidyaml is a single-header library. Link with
// src/tools/yaml.cpp which defines RYML_SINGLE_HDR_DEFINE_NOW.
// Do NOT include this header before defining RYML_SINGLE_HDR_DEFINE_NOW
// in your own code.
//

#ifndef EDGEINFER_YAML_H
#define EDGEINFER_YAML_H

#include <rapidyaml.hpp>

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace edgeinfer {
namespace tools {

// ============================================================
// Yaml - wrapper around rapidyaml (ryml)
// ============================================================

class Yaml {
public:
    Yaml();
    ~Yaml() = default;

    // --- Static Parsers ---
    static Yaml Parse(const std::string& text);
    static Yaml ParseFile(const std::string& path);

    // --- Serialization ---
    std::string Dump() const;

    // --- Element Access ---
    Yaml operator[](const std::string& key) const;
    Yaml operator[](size_t index) const;

    // --- Type Queries ---
    bool IsNull() const;
    bool IsMap() const;
    bool IsSeq() const;
    bool IsVal() const;

    // --- Key Lookup ---
    bool Has(const std::string& key) const;
    bool Contains(const std::string& key) const;

    // --- Size ---
    size_t Size() const;
    bool Empty() const;

    // --- Direct Value Extraction ---
    std::string GetString(const std::string& def = "") const;
    int64_t     GetInt(int64_t def = 0) const;
    double      GetDouble(double def = 0.0) const;
    bool        GetBool(bool def = false) const;

    // --- Key-based Value Extraction ---
    std::string GetString(const std::string& key, const std::string& def) const;
    int64_t     GetInt(const std::string& key, int64_t def) const;
    double      GetDouble(const std::string& key, double def) const;
    bool        GetBool(const std::string& key, bool def) const;

    // --- Sub-tree Extraction ---
    Yaml GetMap(const std::string& key) const;
    Yaml GetSeq(const std::string& key) const;

    // --- Iteration ---
    void ForEach(std::function<void(const std::string&, const Yaml&)> fn) const;
    std::vector<Yaml> SeqItems() const;

    // Internal constructor (uses ryml types — not intended for end-user use)
    Yaml(std::shared_ptr<std::string> buf,
         std::shared_ptr<c4::yml::Tree> tree,
         size_t node_id);

private:
    bool Valid() const;
    c4::yml::ConstNodeRef Node() const;

    std::shared_ptr<std::string>        buffer_;
    std::shared_ptr<c4::yml::Tree>      tree_;
    size_t                              node_id_;

    static constexpr size_t kNpos = static_cast<size_t>(-1);
};

} // namespace tools
} // namespace edgeinfer

#endif // EDGEINFER_YAML_H
