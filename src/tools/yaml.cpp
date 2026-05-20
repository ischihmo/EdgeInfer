//
// EdgeInfer - YAML Parser Implementation
//
// This file defines RYML_SINGLE_HDR_DEFINE_NOW so that rapidyaml
// function definitions are emitted in this translation unit.
//

#define RYML_SINGLE_HDR_DEFINE_NOW
#include "../../include/tools/yaml.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

// Use explicit c4::yml:: namespace — more portable than ryml::
namespace ryml_alias = c4::yml;

namespace edgeinfer {
namespace tools {

// ============================================================
// Anonymous helpers using c4:: types
// ============================================================

namespace {

inline c4::csubstr ToCsubstr(const std::string& s) {
    return c4::csubstr(s.data(), s.size());
}

inline c4::substr ToSubstr(std::string& s) {
    return c4::substr(s.data(), s.size());
}

inline std::string FromCsubstr(c4::csubstr cs) {
    return cs.empty() ? std::string() : std::string(cs.data(), cs.size());
}

inline int64_t CsubstrToInt(c4::csubstr cs) {
    if (cs.empty()) return 0;
    int64_t val = 0;
    c4::atoi(cs, &val);
    return val;
}

inline double CsubstrToDouble(c4::csubstr cs) {
    if (cs.empty()) return 0.0;
    float fval = 0.0f;
    c4::atof(cs, &fval);
    return static_cast<double>(fval);
}

inline bool CsubstrToBool(c4::csubstr cs) {
    if (cs.empty()) return false;
    std::string s(cs.data(), cs.size());
    return s == "true" || s == "True" || s == "TRUE" ||
           s == "yes"  || s == "Yes"  || s == "YES" ||
           s == "1";
}

} // anonymous namespace

// ============================================================
// Constructor helpers
// ============================================================

static constexpr size_t kNpos = static_cast<size_t>(-1);

Yaml::Yaml()
    : buffer_(nullptr), tree_(nullptr), node_id_(kNpos) {}

Yaml::Yaml(std::shared_ptr<std::string> buf,
           std::shared_ptr<c4::yml::Tree> tree,
           size_t node_id)
    : buffer_(std::move(buf)), tree_(std::move(tree)), node_id_(node_id) {}

bool Yaml::Valid() const {
    return tree_ && node_id_ != kNpos && node_id_ < tree_->size();
}

c4::yml::ConstNodeRef Yaml::Node() const {
    if (!Valid()) {
        static c4::yml::Tree dummy_tree;
        return dummy_tree.rootref();
    }
    return c4::yml::ConstNodeRef(tree_.get(), node_id_);
}

// --- Parsing ---

Yaml Yaml::Parse(const std::string& text) {
    auto buf = std::make_shared<std::string>(text);
    auto tree = std::make_shared<c4::yml::Tree>();

    // parse_in_place requires a mutable substr (may insert null terminators)
    c4::substr mutable_src = ToSubstr(*buf);
    c4::yml::parse_in_place(mutable_src, tree.get());

    return Yaml(buf, tree, tree->root_id());
}

Yaml Yaml::ParseFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Yaml::ParseFile: cannot open " + path);
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return Parse(oss.str());
}

std::string Yaml::Dump() const {
    if (!tree_) return "";
    return c4::yml::emitrs_yaml<std::string>(*tree_);
}

// --- Element Access ---

Yaml Yaml::operator[](const std::string& key) const {
    if (!Valid()) return Yaml();
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_map()) return Yaml();
    c4::csubstr ckey = ToCsubstr(key);
    if (!node.has_child(ckey)) return Yaml();
    size_t child_id = node.find_child(ckey).id();
    return Yaml(buffer_, tree_, child_id);
}

Yaml Yaml::operator[](size_t index) const {
    if (!Valid()) return Yaml();
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_seq()) return Yaml();
    if (index >= node.num_children()) return Yaml();
    size_t child_id = node.child(index).id();
    return Yaml(buffer_, tree_, child_id);
}

// --- Type Queries ---

bool Yaml::IsNull() const {
    if (!Valid()) return true;
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_val()) return false;
    if (!node.has_val()) return true;
    c4::csubstr val = node.val();
    return val.empty() || val == "~" || val == "null" || val == "Null" || val == "NULL";
}

bool Yaml::IsMap() const { return Valid() && Node().is_map(); }
bool Yaml::IsSeq() const { return Valid() && Node().is_seq(); }
bool Yaml::IsVal() const { return Valid() && Node().is_val(); }

bool Yaml::Has(const std::string& key) const {
    if (!Valid()) return false;
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_map()) return false;
    return node.has_child(ToCsubstr(key));
}

bool Yaml::Contains(const std::string& key) const {
    return Has(key);
}

size_t Yaml::Size() const {
    if (!Valid()) return 0;
    c4::yml::ConstNodeRef node = Node();
    if (node.is_seq()) return node.num_children();
    if (node.is_map()) return node.num_children();
    return 0;
}

bool Yaml::Empty() const { return Size() == 0; }

// --- Direct Value Extraction ---

std::string Yaml::GetString(const std::string& def) const {
    if (!Valid()) return def;
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_val() || !node.has_val()) return def;
    return FromCsubstr(node.val());
}

int64_t Yaml::GetInt(int64_t def) const {
    if (!Valid()) return def;
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_val() || !node.has_val()) return def;
    return CsubstrToInt(node.val());
}

double Yaml::GetDouble(double def) const {
    if (!Valid()) return def;
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_val() || !node.has_val()) return def;
    return CsubstrToDouble(node.val());
}

bool Yaml::GetBool(bool def) const {
    if (!Valid()) return def;
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_val() || !node.has_val()) return def;
    return CsubstrToBool(node.val());
}

// --- Key-based Value Extraction ---

std::string Yaml::GetString(const std::string& key, const std::string& def) const {
    Yaml child = (*this)[key];
    if (!child.Valid()) return def;
    return child.GetString(def);
}

int64_t Yaml::GetInt(const std::string& key, int64_t def) const {
    Yaml child = (*this)[key];
    if (!child.Valid()) return def;
    return child.GetInt(def);
}

double Yaml::GetDouble(const std::string& key, double def) const {
    Yaml child = (*this)[key];
    if (!child.Valid()) return def;
    return child.GetDouble(def);
}

bool Yaml::GetBool(const std::string& key, bool def) const {
    Yaml child = (*this)[key];
    if (!child.Valid()) return def;
    return child.GetBool(def);
}

Yaml Yaml::GetMap(const std::string& key) const {
    Yaml child = (*this)[key];
    return (child.Valid() && child.IsMap()) ? child : Yaml();
}

Yaml Yaml::GetSeq(const std::string& key) const {
    Yaml child = (*this)[key];
    return (child.Valid() && child.IsSeq()) ? child : Yaml();
}

// --- Iteration ---

void Yaml::ForEach(std::function<void(const std::string&, const Yaml&)> fn) const {
    if (!Valid()) return;
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_map()) return;
    for (size_t i = 0; i < node.num_children(); ++i) {
        c4::yml::ConstNodeRef child = node.child(i);
        std::string key = FromCsubstr(child.key());
        Yaml val(buffer_, tree_, child.id());
        fn(key, val);
    }
}

std::vector<Yaml> Yaml::SeqItems() const {
    std::vector<Yaml> result;
    if (!Valid()) return result;
    c4::yml::ConstNodeRef node = Node();
    if (!node.is_seq()) return result;
    result.reserve(node.num_children());
    for (size_t i = 0; i < node.num_children(); ++i) {
        result.emplace_back(buffer_, tree_, node.child(i).id());
    }
    return result;
}

} // namespace tools
} // namespace edgeinfer
