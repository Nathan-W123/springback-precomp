#include "sparlab/io/Json.hpp"

#include "sparlab/core/Exceptions.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace sparlab {
namespace json {
namespace {

class Parser {
 public:
  Parser(const std::string& text, const std::string& source)
      : text_(text), source_(source) {}

  Value parse_document() {
    skip_whitespace();
    Value v = parse_value();
    skip_whitespace();
    if (pos_ != text_.size()) {
      fail("unexpected trailing content after the top-level value");
    }
    return v;
  }

 private:
  [[noreturn]] void fail(const std::string& message) const {
    std::size_t line = 1;
    std::size_t col = 1;
    for (std::size_t i = 0; i < pos_ && i < text_.size(); ++i) {
      if (text_[i] == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    std::ostringstream os;
    os << source_ << ":" << line << ":" << col << ": " << message;
    throw ConfigError(os.str());
  }

  bool at_end() const { return pos_ >= text_.size(); }
  char peek() const { return at_end() ? '\0' : text_[pos_]; }

  void skip_whitespace() {
    while (!at_end()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else if (c == '/' && pos_ + 1 < text_.size() && text_[pos_ + 1] == '/') {
        while (!at_end() && text_[pos_] != '\n') ++pos_;
      } else if (c == '/' && pos_ + 1 < text_.size() && text_[pos_ + 1] == '*') {
        const std::size_t start = pos_;
        pos_ += 2;
        bool closed = false;
        while (pos_ + 1 < text_.size()) {
          if (text_[pos_] == '*' && text_[pos_ + 1] == '/') {
            pos_ += 2;
            closed = true;
            break;
          }
          ++pos_;
        }
        if (!closed) {
          pos_ = start;
          fail("unterminated block comment");
        }
      } else {
        return;
      }
    }
  }

  void expect(char c) {
    skip_whitespace();
    if (peek() != c) {
      std::ostringstream os;
      os << "expected '" << c << "'";
      if (at_end()) {
        os << " but reached the end of the document";
      } else {
        os << " but found '" << text_[pos_] << "'";
      }
      fail(os.str());
    }
    ++pos_;
  }

  bool consume_literal(const char* literal) {
    const std::size_t n = std::strlen(literal);
    if (text_.compare(pos_, n, literal) == 0) {
      pos_ += n;
      return true;
    }
    return false;
  }

  Value parse_value() {
    skip_whitespace();
    if (at_end()) fail("unexpected end of document where a value was expected");
    const char c = peek();
    switch (c) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Value::make_string(parse_string());
      case 't':
        if (consume_literal("true")) return Value::make_bool(true);
        fail("invalid literal; expected 'true'");
      case 'f':
        if (consume_literal("false")) return Value::make_bool(false);
        fail("invalid literal; expected 'false'");
      case 'n':
        if (consume_literal("null")) return Value();
        fail("invalid literal; expected 'null'");
      default:
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail(std::string("unexpected character '") + c + "' where a value was expected");
    }
  }

  Value parse_object() {
    expect('{');
    Value obj = Value::make_object();
    skip_whitespace();
    if (peek() == '}') {
      ++pos_;
      return obj;
    }
    while (true) {
      skip_whitespace();
      if (peek() != '"') fail("object keys must be double-quoted strings");
      const std::string key = parse_string();
      if (obj.find(key) != nullptr) fail("duplicate object key '" + key + "'");
      expect(':');
      obj.set(key, parse_value());
      skip_whitespace();
      if (peek() == ',') {
        ++pos_;
        skip_whitespace();
        if (peek() == '}') {  // tolerated trailing comma
          ++pos_;
          return obj;
        }
        continue;
      }
      if (peek() == '}') {
        ++pos_;
        return obj;
      }
      fail("expected ',' or '}' in object");
    }
  }

  Value parse_array() {
    expect('[');
    Value arr = Value::make_array();
    skip_whitespace();
    if (peek() == ']') {
      ++pos_;
      return arr;
    }
    while (true) {
      arr.push_back(parse_value());
      skip_whitespace();
      if (peek() == ',') {
        ++pos_;
        skip_whitespace();
        if (peek() == ']') {  // tolerated trailing comma
          ++pos_;
          return arr;
        }
        continue;
      }
      if (peek() == ']') {
        ++pos_;
        return arr;
      }
      fail("expected ',' or ']' in array");
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (at_end()) fail("unterminated string");
      const char c = text_[pos_++];
      if (c == '"') return out;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (at_end()) fail("unterminated escape sequence");
      const char esc = text_[pos_++];
      switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (pos_ + 4 > text_.size()) fail("truncated \\u escape");
          unsigned int code = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = text_[pos_ + static_cast<std::size_t>(i)];
            code <<= 4;
            if (h >= '0' && h <= '9') {
              code |= static_cast<unsigned int>(h - '0');
            } else if (h >= 'a' && h <= 'f') {
              code |= static_cast<unsigned int>(h - 'a' + 10);
            } else if (h >= 'A' && h <= 'F') {
              code |= static_cast<unsigned int>(h - 'A' + 10);
            } else {
              fail("invalid hexadecimal digit in \\u escape");
            }
          }
          pos_ += 4;
          // Encode as UTF-8. Surrogate pairs are not recombined; configuration
          // files are ASCII in practice and this keeps the parser small.
          if (code < 0x80) {
            out.push_back(static_cast<char>(code));
          } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          } else {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          }
          break;
        }
        default:
          fail(std::string("unknown escape sequence '\\") + esc + "'");
      }
    }
  }

  Value parse_number() {
    const std::size_t start = pos_;
    if (peek() == '-') ++pos_;
    while (!at_end() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    if (peek() == '.') {
      ++pos_;
      while (!at_end() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    if (peek() == 'e' || peek() == 'E') {
      ++pos_;
      if (peek() == '+' || peek() == '-') ++pos_;
      while (!at_end() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    const std::string token = text_.substr(start, pos_ - start);
    try {
      std::size_t consumed = 0;
      const double value = std::stod(token, &consumed);
      if (consumed != token.size()) {
        pos_ = start;
        fail("malformed number '" + token + "'");
      }
      return Value::make_number(value);
    } catch (const std::exception&) {
      pos_ = start;
      fail("malformed or out-of-range number '" + token + "'");
    }
  }

  const std::string& text_;
  const std::string& source_;
  std::size_t pos_ = 0;
};

void dump_value(const Value& v, int indent, int depth, std::ostringstream& os);

void dump_string(const std::string& s, std::ostringstream& os) {
  os << '"';
  for (char c : s) {
    switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\b': os << "\\b"; break;
      case '\f': os << "\\f"; break;
      case '\n': os << "\\n"; break;
      case '\r': os << "\\r"; break;
      case '\t': os << "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<int>(static_cast<unsigned char>(c)) << std::dec
             << std::setfill(' ');
        } else {
          os << c;
        }
    }
  }
  os << '"';
}

void dump_number(double v, std::ostringstream& os) {
  if (!std::isfinite(v)) {
    // JSON has no NaN/Inf. Emitting null keeps documents parseable and is
    // visible to the reader instead of silently becoming 0.
    os << "null";
    return;
  }
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%.17g", v);
  os << buffer;
}

void newline_indent(int indent, int depth, std::ostringstream& os) {
  if (indent <= 0) return;
  os << '\n';
  os << std::string(static_cast<std::size_t>(indent * depth), ' ');
}

void dump_value(const Value& v, int indent, int depth, std::ostringstream& os) {
  switch (v.type()) {
    case Value::Type::Null: os << "null"; return;
    case Value::Type::Bool: os << (v.bool_value() ? "true" : "false"); return;
    case Value::Type::Number: dump_number(v.number_value(), os); return;
    case Value::Type::String: dump_string(v.string_value(), os); return;
    case Value::Type::Array: {
      if (v.array_items().empty()) {
        os << "[]";
        return;
      }
      os << '[';
      bool first = true;
      for (const Value& item : v.array_items()) {
        if (!first) os << (indent > 0 ? "," : ",");
        first = false;
        newline_indent(indent, depth + 1, os);
        dump_value(item, indent, depth + 1, os);
      }
      newline_indent(indent, depth, os);
      os << ']';
      return;
    }
    case Value::Type::Object: {
      if (v.members().empty()) {
        os << "{}";
        return;
      }
      os << '{';
      bool first = true;
      for (const auto& kv : v.members()) {
        if (!first) os << ',';
        first = false;
        newline_indent(indent, depth + 1, os);
        dump_string(kv.first, os);
        os << (indent > 0 ? ": " : ":");
        dump_value(kv.second, indent, depth + 1, os);
      }
      newline_indent(indent, depth, os);
      os << '}';
      return;
    }
  }
}

/// Recursively collect every object key path in a document.
void collect_paths(const Value& v, const std::string& prefix,
                   std::vector<std::string>& out) {
  if (v.is_object()) {
    for (const auto& kv : v.members()) {
      const std::string path = prefix.empty() ? kv.first : prefix + "." + kv.first;
      out.push_back(path);
      collect_paths(kv.second, path, out);
    }
  } else if (v.is_array()) {
    for (std::size_t i = 0; i < v.array_items().size(); ++i) {
      std::ostringstream os;
      os << prefix << "[" << i << "]";
      collect_paths(v.array_items()[i], os.str(), out);
    }
  }
}

}  // namespace

Value Value::make_bool(bool v) {
  Value out;
  out.type_ = Type::Bool;
  out.bool_ = v;
  return out;
}

Value Value::make_number(double v) {
  Value out;
  out.type_ = Type::Number;
  out.number_ = v;
  return out;
}

Value Value::make_string(std::string v) {
  Value out;
  out.type_ = Type::String;
  out.string_ = std::move(v);
  return out;
}

Value Value::make_array() {
  Value out;
  out.type_ = Type::Array;
  return out;
}

Value Value::make_object() {
  Value out;
  out.type_ = Type::Object;
  return out;
}

const Value* Value::find(const std::string& key) const {
  if (type_ != Type::Object) return nullptr;
  for (const auto& kv : members_) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

void Value::push_back(Value v) {
  if (type_ != Type::Array) {
    type_ = Type::Array;
    array_.clear();
  }
  array_.push_back(std::move(v));
}

void Value::set(const std::string& key, Value v) {
  if (type_ != Type::Object) {
    type_ = Type::Object;
    members_.clear();
  }
  for (auto& kv : members_) {
    if (kv.first == key) {
      kv.second = std::move(v);
      return;
    }
  }
  members_.emplace_back(key, std::move(v));
}

std::string Value::type_name(Type t) {
  switch (t) {
    case Type::Null: return "null";
    case Type::Bool: return "boolean";
    case Type::Number: return "number";
    case Type::String: return "string";
    case Type::Array: return "array";
    case Type::Object: return "object";
  }
  return "unknown";
}

Value parse(const std::string& text, const std::string& source_name) {
  Parser parser(text, source_name);
  return parser.parse_document();
}

Value parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw IoError("cannot open configuration file '" + path + "' for reading");
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return parse(buffer.str(), path);
}

std::string dump(const Value& value, int indent) {
  std::ostringstream os;
  dump_value(value, indent, 0, os);
  return os.str();
}

Value array_of(const std::vector<std::string>& v) {
  Value out = Value::make_array();
  for (const std::string& s : v) out.push_back(Value::make_string(s));
  return out;
}

}  // namespace json

// ----------------------------------------------------------------------------
// ConfigNode
// ----------------------------------------------------------------------------

ConfigNode::ConfigNode(const json::Value& value, std::string path)
    : value_(&value),
      path_(std::move(path)),
      visited_(std::make_shared<std::set<std::string>>()),
      root_(&value) {}

ConfigNode::ConfigNode(const json::Value* value, std::string path,
                       std::shared_ptr<std::set<std::string>> visited)
    : value_(value), path_(std::move(path)), visited_(std::move(visited)) {}

void ConfigNode::note_visit() const {
  if (visited_ && !path_.empty()) visited_->insert(path_);
}

ConfigNode ConfigNode::child(const std::string& key) const {
  const std::string path = path_.empty() ? key : path_ + "." + key;
  const json::Value* next = nullptr;
  if (value_ != nullptr) {
    if (!value_->is_object() && !value_->is_null()) {
      throw ConfigError("'" + (path_.empty() ? std::string("<root>") : path_) +
                        "' must be an object to read '" + key + "', but it is a " +
                        value_->type_name());
    }
    next = value_->find(key);
  }
  ConfigNode node(next, path, visited_);
  node.root_ = root_;
  node.note_visit();
  return node;
}

ConfigNode ConfigNode::require(const std::string& key) const {
  ConfigNode node = child(key);
  if (!node.exists()) {
    throw ConfigError("required key '" + node.path() + "' is missing");
  }
  return node;
}

Scalar ConfigNode::number() const {
  if (!exists()) throw ConfigError("required numeric key '" + path_ + "' is missing");
  if (!value_->is_number()) {
    throw ConfigError("'" + path_ + "' must be a number, got " + value_->type_name());
  }
  return value_->number_value();
}

int ConfigNode::integer() const {
  const Scalar v = number();
  const Scalar rounded = std::round(v);
  if (std::abs(v - rounded) > 1.0e-9) {
    std::ostringstream os;
    os << "'" << path_ << "' must be an integer, got " << v;
    throw ConfigError(os.str());
  }
  return static_cast<int>(rounded);
}

bool ConfigNode::boolean() const {
  if (!exists()) throw ConfigError("required boolean key '" + path_ + "' is missing");
  if (!value_->is_bool()) {
    throw ConfigError("'" + path_ + "' must be true or false, got " +
                      value_->type_name());
  }
  return value_->bool_value();
}

std::string ConfigNode::string() const {
  if (!exists()) throw ConfigError("required string key '" + path_ + "' is missing");
  if (!value_->is_string()) {
    throw ConfigError("'" + path_ + "' must be a string, got " + value_->type_name());
  }
  return value_->string_value();
}

Vector3 ConfigNode::vector3(int dim) const {
  if (!exists()) throw ConfigError("required vector key '" + path_ + "' is missing");
  const std::size_t count = value_->is_array() ? value_->array_items().size() : 0;
  const bool ok = value_->is_array() && (count == 2 || count == 3);
  if (!ok) {
    std::ostringstream os;
    os << "'" << path_ << "' must be an array of two or three numbers [x, y] or [x, y, z]";
    if (value_->is_array()) os << ", got " << count << " entries";
    throw ConfigError(os.str());
  }
  if (dim == 2 && count == 3) {
    std::ostringstream os;
    os << "'" << path_
       << "' has three entries but the mesh is two-dimensional; give [x, y] only";
    throw ConfigError(os.str());
  }
  if (dim == 3 && count == 2) {
    std::ostringstream os;
    os << "'" << path_
       << "' has two entries but the mesh is three-dimensional; give [x, y, z]";
    throw ConfigError(os.str());
  }
  Vector3 out = Vector3::Zero();
  for (std::size_t i = 0; i < count; ++i) {
    const json::Value& item = value_->array_items()[i];
    if (!item.is_number()) {
      std::ostringstream os;
      os << "'" << path_ << "[" << i << "]' must be a number, got " << item.type_name();
      throw ConfigError(os.str());
    }
    out(static_cast<Eigen::Index>(i)) = item.number_value();
  }
  return out;
}

std::vector<Scalar> ConfigNode::number_list() const {
  if (!exists()) return {};
  if (!value_->is_array()) {
    throw ConfigError("'" + path_ + "' must be an array of numbers, got " +
                      value_->type_name());
  }
  std::vector<Scalar> out;
  out.reserve(value_->array_items().size());
  for (std::size_t i = 0; i < value_->array_items().size(); ++i) {
    const json::Value& item = value_->array_items()[i];
    if (!item.is_number()) {
      std::ostringstream os;
      os << "'" << path_ << "[" << i << "]' must be a number, got " << item.type_name();
      throw ConfigError(os.str());
    }
    out.push_back(item.number_value());
  }
  return out;
}

std::vector<Index> ConfigNode::index_list() const {
  const std::vector<Scalar> raw = number_list();
  std::vector<Index> out;
  out.reserve(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const Scalar rounded = std::round(raw[i]);
    if (std::abs(raw[i] - rounded) > 1.0e-9 || rounded < 0.0) {
      std::ostringstream os;
      os << "'" << path_ << "[" << i << "]' must be a non-negative integer index, got "
         << raw[i];
      throw ConfigError(os.str());
    }
    out.push_back(static_cast<Index>(rounded));
  }
  return out;
}

Scalar ConfigNode::number_or(const std::string& key, Scalar fallback) const {
  ConfigNode node = child(key);
  return node.exists() ? node.number() : fallback;
}

int ConfigNode::integer_or(const std::string& key, int fallback) const {
  ConfigNode node = child(key);
  return node.exists() ? node.integer() : fallback;
}

bool ConfigNode::boolean_or(const std::string& key, bool fallback) const {
  ConfigNode node = child(key);
  return node.exists() ? node.boolean() : fallback;
}

std::string ConfigNode::string_or(const std::string& key,
                                  const std::string& fallback) const {
  ConfigNode node = child(key);
  return node.exists() ? node.string() : fallback;
}

Vector3 ConfigNode::vector3_or(const std::string& key, const Vector3& fallback,
                               int dim) const {
  ConfigNode node = child(key);
  return node.exists() ? node.vector3(dim) : fallback;
}

Scalar ConfigNode::positive_number(const std::string& key) const {
  ConfigNode node = require(key);
  const Scalar v = node.number();
  if (!(v > 0.0)) {
    std::ostringstream os;
    os << "'" << node.path() << "' must be positive, got " << v;
    throw ConfigError(os.str());
  }
  return v;
}

std::vector<ConfigNode> ConfigNode::items() const {
  std::vector<ConfigNode> out;
  if (!exists()) return out;
  if (!value_->is_array()) {
    throw ConfigError("'" + path_ + "' must be an array, got " + value_->type_name());
  }
  for (std::size_t i = 0; i < value_->array_items().size(); ++i) {
    std::ostringstream os;
    os << path_ << "[" << i << "]";
    ConfigNode node(&value_->array_items()[i], os.str(), visited_);
    node.root_ = root_;
    node.note_visit();
    out.push_back(node);
  }
  return out;
}

std::vector<ConfigNode> ConfigNode::array(const std::string& key) const {
  return child(key).items();
}

std::vector<std::string> ConfigNode::unused_keys() const {
  std::vector<std::string> all;
  if (root_ != nullptr) json::collect_paths(*root_, "", all);
  std::vector<std::string> unused;
  for (const std::string& p : all) {
    if (visited_ && visited_->count(p) == 0) unused.push_back(p);
  }
  // Drop paths whose parent is already reported, so one typo yields one message.
  std::vector<std::string> filtered;
  for (const std::string& p : unused) {
    bool covered = false;
    for (const std::string& q : unused) {
      if (q.size() < p.size() && p.compare(0, q.size(), q) == 0 && p[q.size()] == '.') {
        covered = true;
        break;
      }
    }
    if (!covered) filtered.push_back(p);
  }
  return filtered;
}

}  // namespace sparlab
