/// \file Json.hpp
/// \brief Self-contained JSON reader/writer and a validating config reader.
///
/// SparLab parses its own JSON rather than vendoring a dependency, so the build
/// needs nothing beyond Eigen and a C++17 compiler. The parser implements
/// RFC 8259 with two documented conveniences:
///   * `//` line comments and `/* */` block comments are accepted, because
///     annotated input decks are far easier to maintain;
///   * a trailing comma before `}` or `]` is accepted.
/// Everything else (string escapes, numbers, nesting) follows the standard, and
/// syntax errors report the offending line and column.
///
/// `ConfigNode` wraps a parsed value with the *path* it came from, so a bad
/// entry produces "configuration error: forming.tools[0].radius must be a
/// number, got string" instead of a bare type error. Every access is recorded
/// in a set shared by all nodes of one document, so `unused_keys` can list keys
/// present in the deck that nothing ever read - which is how a misspelled key
/// gets caught instead of silently taking its default.
#pragma once

#include "sparlab/core/Types.hpp"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace sparlab {
namespace json {

class Value {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Value() = default;
  static Value make_bool(bool v);
  static Value make_number(double v);
  static Value make_string(std::string v);
  static Value make_array();
  static Value make_object();

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool bool_value() const { return bool_; }
  double number_value() const { return number_; }
  const std::string& string_value() const { return string_; }
  const std::vector<Value>& array_items() const { return array_; }
  const std::vector<std::pair<std::string, Value>>& members() const { return members_; }

  /// Pointer to the member with this key, or nullptr.
  const Value* find(const std::string& key) const;

  /// Mutators used when building output documents.
  /// \{
  void push_back(Value v);
  void set(const std::string& key, Value v);
  /// \}

  static std::string type_name(Type t);
  std::string type_name() const { return type_name(type_); }

 private:
  Type type_ = Type::Null;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Value> array_;
  std::vector<std::pair<std::string, Value>> members_;
};

/// Parse a JSON document.
/// \throws ConfigError with line/column information on a syntax error.
Value parse(const std::string& text, const std::string& source_name = "<string>");

/// Parse a JSON file.
/// \throws IoError when the file cannot be read, ConfigError on syntax errors.
Value parse_file(const std::string& path);

/// Serialise to a string. `indent <= 0` produces a compact single line.
std::string dump(const Value& value, int indent = 2);

/// Helpers for building result documents.
/// \{
Value array_of(const std::vector<std::string>& v);
/// \}

}  // namespace json

/// Path-aware, typo-catching reader over a parsed JSON document.
class ConfigNode {
 public:
  /// Build a root node over `value`, which must outlive the node.
  explicit ConfigNode(const json::Value& value, std::string path = "");

  bool exists() const { return value_ != nullptr && !value_->is_null(); }
  const std::string& path() const { return path_; }
  const json::Value* raw() const { return value_; }

  /// Child node for `key`. A missing key yields a non-existent node, so
  /// `child("a").number_or("b", 1.0)` works on absent sections.
  ConfigNode child(const std::string& key) const;

  /// Child node for `key`, throwing ConfigError when absent.
  ConfigNode require(const std::string& key) const;

  /// Typed accessors for this node's own value.
  /// \{
  Scalar number() const;
  int integer() const;
  bool boolean() const;
  std::string string() const;
  /// Two- or three-entry numeric array as a 3-vector (z = 0 when two entries
  /// are given). `dim`, when 2 or 3, requires the entry count to match the
  /// mesh dimension so a component is never silently dropped or invented;
  /// 0 accepts either form.
  Vector3 vector3(int dim = 0) const;
  std::vector<Scalar> number_list() const;
  std::vector<Index> index_list() const;
  /// \}

  /// Typed accessors for a child key, with a fallback when absent.
  /// \{
  Scalar number_or(const std::string& key, Scalar fallback) const;
  int integer_or(const std::string& key, int fallback) const;
  bool boolean_or(const std::string& key, bool fallback) const;
  std::string string_or(const std::string& key, const std::string& fallback) const;
  Vector3 vector3_or(const std::string& key, const Vector3& fallback, int dim = 0) const;
  /// \}

  /// Required number that must be strictly positive.
  Scalar positive_number(const std::string& key) const;

  /// Array elements of this node (throws if this node is not an array).
  std::vector<ConfigNode> items() const;

  /// Array elements of `key`; empty when the key is absent.
  std::vector<ConfigNode> array(const std::string& key) const;

  /// Keys present in the document that no accessor ever touched.
  /// Only meaningful on the root node.
  std::vector<std::string> unused_keys() const;

 private:
  ConfigNode(const json::Value* value, std::string path,
             std::shared_ptr<std::set<std::string>> visited);
  void note_visit() const;

  const json::Value* value_ = nullptr;
  std::string path_;
  std::shared_ptr<std::set<std::string>> visited_;
  const json::Value* root_ = nullptr;
};

}  // namespace sparlab
