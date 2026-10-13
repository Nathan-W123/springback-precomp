// Where this fits: see include/sparlab/io/Config.hpp

#include "sparlab/io/Config.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Logging.hpp"
#include "sparlab/elements/Quadrature.hpp"

#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>

namespace sparlab {
namespace {

StressState parse_stress_state(const std::string& text) {
  if (text == "plane_stress") return StressState::PlaneStress;
  if (text == "plane_strain") return StressState::PlaneStrain;
  if (text == "three_dimensional" || text == "3d") return StressState::ThreeDimensional;
  throw ConfigError("unknown stress state '" + text +
                    "' (expected plane_stress|plane_strain|three_dimensional)");
}

int parse_axis(const ConfigNode& node, const std::string& key) {
  const std::string text = node.string_or(key, "z");
  if (text == "x") return 0;
  if (text == "y") return 1;
  if (text == "z") return 2;
  throw ConfigError("'" + node.path() + "." + key + "' must be \"x\", \"y\" or \"z\", got \"" +
                    text + "\"");
}

Selector parse_selector_primitive(const ConfigNode& node, int dim) {
  Selector sel;
  sel.tolerance = node.number_or("tolerance", 0.0);

  int matched = 0;
  if (node.child("all").exists()) {
    sel.kind = SelectorKind::All;
    ++matched;
  }
  if (node.child("box").exists()) {
    const ConfigNode box = node.child("box");
    sel.kind = SelectorKind::Box;
    sel.xmin = box.number_or("xmin", -std::numeric_limits<Scalar>::infinity());
    sel.xmax = box.number_or("xmax", std::numeric_limits<Scalar>::infinity());
    sel.ymin = box.number_or("ymin", -std::numeric_limits<Scalar>::infinity());
    sel.ymax = box.number_or("ymax", std::numeric_limits<Scalar>::infinity());
    sel.zmin = box.number_or("zmin", -std::numeric_limits<Scalar>::infinity());
    sel.zmax = box.number_or("zmax", std::numeric_limits<Scalar>::infinity());
    if (dim == 2 && (box.child("zmin").exists() || box.child("zmax").exists())) {
      throw ConfigError("'" + box.path() +
                        "' sets zmin/zmax but the mesh is two-dimensional");
    }
    if (sel.xmin > sel.xmax || sel.ymin > sel.ymax || sel.zmin > sel.zmax) {
      std::ostringstream os;
      os << "'" << box.path() << "' has an empty interval: x in [" << sel.xmin << ", "
         << sel.xmax << "], y in [" << sel.ymin << ", " << sel.ymax << "]";
      if (dim == 3) os << ", z in [" << sel.zmin << ", " << sel.zmax << "]";
      throw ConfigError(os.str());
    }
    ++matched;
  }
  if (node.child("circle").exists()) {
    const ConfigNode c = node.child("circle");
    sel.kind = SelectorKind::Circle;
    sel.center = c.require("center").vector3(dim);
    sel.radius = c.positive_number("radius");
    sel.axis = parse_axis(c, "axis");
    if (dim == 2 && sel.axis != 2) {
      throw ConfigError("'" + c.path() + ".axis' must be \"z\" on a two-dimensional mesh");
    }
    ++matched;
  }
  if (node.child("annulus").exists()) {
    const ConfigNode c = node.child("annulus");
    sel.kind = SelectorKind::Annulus;
    sel.center = c.require("center").vector3(dim);
    sel.inner_radius = c.number_or("inner_radius", 0.0);
    sel.radius = c.positive_number("radius");
    sel.axis = parse_axis(c, "axis");
    if (dim == 2 && sel.axis != 2) {
      throw ConfigError("'" + c.path() + ".axis' must be \"z\" on a two-dimensional mesh");
    }
    if (sel.inner_radius >= sel.radius) {
      std::ostringstream os;
      os << "'" << c.path() << "' needs inner_radius (" << sel.inner_radius
         << " m) strictly below radius (" << sel.radius << " m)";
      throw ConfigError(os.str());
    }
    ++matched;
  }
  if (node.child("sphere").exists()) {
    const ConfigNode c = node.child("sphere");
    sel.kind = SelectorKind::Sphere;
    sel.center = c.require("center").vector3(dim);
    sel.radius = c.positive_number("radius");
    ++matched;
  }
  if (node.child("node_ids").exists()) {
    sel.kind = SelectorKind::NodeIds;
    sel.ids = node.child("node_ids").index_list();
    if (sel.ids.empty()) {
      throw ConfigError("'" + node.child("node_ids").path() + "' is empty");
    }
    ++matched;
  }
  if (node.child("element_ids").exists()) {
    sel.kind = SelectorKind::ElementIds;
    sel.ids = node.child("element_ids").index_list();
    if (sel.ids.empty()) {
      throw ConfigError("'" + node.child("element_ids").path() + "' is empty");
    }
    ++matched;
  }
  if (node.child("nearest_node").exists()) {
    sel.kind = SelectorKind::NearestNode;
    sel.point = node.child("nearest_node").vector3(dim);
    ++matched;
  }
  if (node.child("group").exists()) {
    sel.kind = SelectorKind::Group;
    sel.group = node.child("group").string();
    if (sel.group.empty()) {
      throw ConfigError("'" + node.child("group").path() + "' is an empty group name");
    }
    ++matched;
  }

  if (matched == 0) {
    throw ConfigError(
        "'" + node.path() +
        "' does not name a region primitive; expected one of all, box, circle, "
        "annulus, sphere, node_ids, element_ids, nearest_node, group (or an 'any_of' "
        "list)");
  }
  if (matched > 1) {
    throw ConfigError("'" + node.path() +
                      "' names more than one region primitive; use 'any_of' to combine "
                      "several regions");
  }
  return sel;
}

}  // namespace

std::string to_string(MeshKind kind) {
  switch (kind) {
    case MeshKind::StructuredHex: return "structured_hex";
    case MeshKind::File: return "file";
  }
  return "unknown";
}

SelectorGroup parse_region(const ConfigNode& node, const std::string& default_name,
                           int dim) {
  if (!node.exists()) {
    throw ConfigError("required region '" + node.path() + "' is missing");
  }
  SelectorGroup group;
  group.name = node.string_or("name", default_name);
  group.invert = node.boolean_or("invert", false);

  const std::vector<ConfigNode> any_of = node.array("any_of");
  if (!any_of.empty()) {
    for (const ConfigNode& item : any_of) {
      group.members.push_back(parse_selector_primitive(item, dim));
    }
  } else {
    group.members.push_back(parse_selector_primitive(node, dim));
  }
  return group;
}

namespace {

/// A three-component direction of the material frame (in 2-D too: the
/// transverse direction may be the model's z).
Vector3 frame_vector(const ConfigNode& node) {
  const std::vector<Scalar> v = node.number_list();
  if (v.size() != 3) {
    throw ConfigError("'" + node.path() + "' must be a direction of three numbers [x, y, z]");
  }
  return Vector3(v[0], v[1], v[2]);
}

/// The Hill48 block of a plasticity section: one calibration (r-values,
/// stress ratios or coefficients) and the material frame.
Hill48Parameters parse_anisotropy(const ConfigNode& node) {
  Hill48Parameters h;
  const bool r_values = node.child("r0").exists() || node.child("r45").exists() ||
                        node.child("r90").exists();
  const ConfigNode ratios = node.child("stress_ratios");
  const ConfigNode coefficients = node.child("coefficients");
  const int forms =
      (r_values ? 1 : 0) + (ratios.exists() ? 1 : 0) + (coefficients.exists() ? 1 : 0);
  if (forms != 1) {
    throw ConfigError("'" + node.path() + "' needs exactly one calibration of the Hill48 "
                      "criterion: the r-values \"r0\", \"r45\", \"r90\", or "
                      "\"stress_ratios\", or \"coefficients\" (got " +
                      std::to_string(forms) + ")");
  }
  if (r_values) {
    h.calibration = HillCalibration::RValues;
    h.r0 = node.require("r0").number();
    h.r45 = node.require("r45").number();
    h.r90 = node.require("r90").number();
  } else if (ratios.exists()) {
    h.calibration = HillCalibration::StressRatios;
    h.sigma45 = ratios.require("sigma_45").number();
    h.sigma90 = ratios.require("sigma_90").number();
    h.sigma_biaxial = ratios.require("sigma_biaxial").number();
  } else {
    h.calibration = HillCalibration::Coefficients;
    h.F = coefficients.require("F").number();
    h.G = coefficients.require("G").number();
    h.H = coefficients.require("H").number();
    h.L = coefficients.require("L").number();
    h.M = coefficients.require("M").number();
    h.N = coefficients.require("N").number();
  }
  const ConfigNode shear = node.child("out_of_plane_shear");
  if (shear.exists()) {
    if (coefficients.exists()) {
      throw ConfigError("'" + shear.path() + "' is given by L and M of the \"coefficients\"");
    }
    const std::vector<Scalar> lm = shear.number_list();
    if (lm.size() != 2) {
      throw ConfigError("'" + shear.path() + "' must be the two numbers [L, M]");
    }
    h.shear_l = lm[0];
    h.shear_m = lm[1];
  }
  const ConfigNode angle = node.child("rolling_angle");
  if (angle.exists()) {
    if (node.child("rolling_direction").exists() || node.child("sheet_normal").exists()) {
      throw ConfigError("'" + angle.path() + "' excludes \"rolling_direction\" and "
                        "\"sheet_normal\": it rolls the sheet in the x-y plane, normal z");
    }
    const Scalar a = angle.number() * 3.14159265358979323846 / 180.0;
    h.rolling_direction = Vector3(std::cos(a), std::sin(a), 0.0);
    h.sheet_normal = Vector3::UnitZ();
  } else {
    if (node.child("rolling_direction").exists()) {
      h.rolling_direction = frame_vector(node.child("rolling_direction"));
    }
    if (node.child("sheet_normal").exists()) {
      h.sheet_normal = frame_vector(node.child("sheet_normal"));
    }
  }
  return h;
}

/// A material section: elastic constants, density, the thermal properties
/// and the plasticity. `dim` is the mesh dimension (a plane model needs z
/// along an axis of the Hill48 frame).
IsotropicMaterial parse_material(const ConfigNode& mat, const std::string& default_name, int dim) {
  IsotropicMaterial material(mat.positive_number("youngs_modulus"),
                             mat.require("poisson_ratio").number(),
                             mat.number_or("density", 0.0),
                             mat.string_or("name", default_name));
  const Scalar alpha = mat.number_or("thermal_expansion", 0.0);
  const Scalar t_ref = mat.number_or("reference_temperature", 0.0);
  const Scalar conductivity = mat.number_or("conductivity", 0.0);
  material.set_thermal(alpha, t_ref, conductivity);
  const ConfigNode plastic = mat.child("plasticity");
  if (plastic.exists()) {
    PlasticityParameters p;
    p.yield_stress = plastic.positive_number("yield_stress");
    p.hardening_modulus = plastic.number_or("hardening_modulus", 0.0);
    p.kinematic_hardening_modulus = plastic.number_or("kinematic_hardening_modulus", 0.0);
    p.saturation_stress = plastic.number_or("saturation_stress", 0.0);
    p.saturation_rate = plastic.number_or("saturation_rate", 0.0);
    const std::string criterion = plastic.string_or("yield_criterion", "von_mises");
    const ConfigNode anisotropy = plastic.child("anisotropy");
    if (criterion == "hill48") {
      if (!anisotropy.exists()) {
        throw ConfigError("'" + plastic.path() + "' has \"yield_criterion\": \"hill48\" but no "
                          "\"anisotropy\" block to calibrate it");
      }
      p.criterion = YieldCriterion::Hill48;
      p.hill = parse_anisotropy(anisotropy);
    } else if (criterion == "von_mises") {
      if (anisotropy.exists()) {
        throw ConfigError("'" + anisotropy.path() + "' needs \"yield_criterion\": \"hill48\"");
      }
    } else {
      throw ConfigError("'" + plastic.path() + ".yield_criterion' is \"" + criterion +
                        "\"; expected \"von_mises\" or \"hill48\"");
    }
    const std::vector<ConfigNode> backstresses = plastic.array("backstresses");
    if (backstresses.size() > static_cast<std::size_t>(kMaxBackstresses)) {
      throw ConfigError("'" + plastic.path() + ".backstresses' has " +
                        std::to_string(backstresses.size()) + " entries; at most " +
                        std::to_string(kMaxBackstresses) + " are supported");
    }
    p.num_backstresses = static_cast<int>(backstresses.size());
    for (std::size_t i = 0; i < backstresses.size(); ++i) {
      p.backstresses[i].modulus = backstresses[i].require("modulus").number();
      p.backstresses[i].recovery = backstresses[i].number_or("recovery", 0.0);
    }
    const std::string integration = plastic.string_or("kinematic_integration", "exponential");
    if (integration == "exponential") {
      p.kinematic_integration = KinematicIntegration::Exponential;
    } else if (integration == "backward_euler") {
      p.kinematic_integration = KinematicIntegration::BackwardEuler;
    } else {
      throw ConfigError("'" + plastic.path() + ".kinematic_integration' is \"" + integration +
                        "\"; expected \"exponential\" or \"backward_euler\"");
    }
    try {
      material.set_plasticity(p);
    } catch (const ConfigError& e) {
      throw ConfigError("'" + plastic.path() + "': " + e.what());
    }
    if (dim == 2 && !material.plasticity().plane_compatible()) {
      // A plane model has no out-of-plane shear strain, which only a frame
      // with z along one of its axes leaves uncoupled from the in-plane flow.
      throw ConfigError("'" + anisotropy.path() + "': in a plane model the out-of-plane axis z "
                        "must be the rolling, transverse or normal direction of the sheet");
    }
  }
  return material;
}

}  // namespace

const IsotropicMaterial& Configuration::material() const {
  if (!material_.has_value()) {
    throw ConfigError("the configuration has no material section");
  }
  return *material_;
}

void Configuration::set_material(IsotropicMaterial material) {
  material_ = std::move(material);
}

int Configuration::dim() const {
  switch (mesh_kind) {
    case MeshKind::StructuredHex: return 3;
    case MeshKind::File:
      if (file_mesh == nullptr) {
        throw ConfigError("the configuration names a mesh file that has not been read");
      }
      return file_mesh->dim();
  }
  throw ConfigError("unhandled mesh type");
}

std::string Configuration::describe_mesh() const {
  std::ostringstream os;
  if (mesh_kind == MeshKind::File) {
    os << "file '" << mesh_file.path << "'";
    if (file_mesh != nullptr) {
      os << " (" << to_string(file_mesh->element_type()) << ", " << file_mesh->num_elements()
         << " elements)";
    }
    return os.str();
  }
  os << to_string(mesh_kind) << " " << mesh_spec.nx << " x " << mesh_spec.ny;
  if (dim() == 3) os << " x " << mesh_spec.nz;
  return os.str();
}

namespace {

/// A displacement constraint: its region, the components it fixes and their
/// values.
DisplacementConstraint parse_displacement_constraint(const ConfigNode& bc,
                                                     const std::string& default_name, int dim) {
  DisplacementConstraint constraint;
  constraint.region = parse_region(bc.require("region"), bc.string_or("name", default_name), dim);
  const std::vector<ConfigNode> fix = bc.array("fix");
  if (fix.empty()) {
    throw ConfigError("'" + bc.path() +
                      ".fix' must list the components to constrain, e.g. "
                      "[\"x\", \"y\"]");
  }
  for (const ConfigNode& component : fix) {
    const std::string c = component.string();
    if (c == "x") {
      constraint.fix_x = true;
    } else if (c == "y") {
      constraint.fix_y = true;
    } else if (c == "z" && dim == 3) {
      constraint.fix_z = true;
    } else if (c == "z") {
      throw ConfigError("'" + component.path() +
                        "' fixes \"z\" but the mesh is two-dimensional");
    } else {
      throw ConfigError("'" + component.path() + "' must be \"x\", \"y\"" +
                        (dim == 3 ? " or \"z\"" : "") + ", got \"" + c + "\"");
    }
  }
  const Vector3 values = bc.vector3_or("value", Vector3::Zero(), dim);
  constraint.value_x = values.x();
  constraint.value_y = values.y();
  constraint.value_z = values.z();
  return constraint;
}

std::vector<std::string> string_list(const ConfigNode& parent, const std::string& key) {
  std::vector<std::string> out;
  for (const ConfigNode& item : parent.array(key)) out.push_back(item.string());
  return out;
}

/// The `explicit` block of a form_explicit step (docs/forming.md, "Explicit
/// forming").
void parse_explicit_step(const ConfigNode& ex, FormingStep& step, const std::string& label) {
  step.tool_speed = ex.number_or("tool_speed", 0.0);
  step.duration = ex.number_or("duration", 0.0);
  const bool speed = ex.child("tool_speed").exists();
  const bool duration = ex.child("duration").exists();
  if (speed == duration) {
    throw ConfigError("'" + ex.path() + "' needs either 'tool_speed' [m/s] (the tools travel "
                      "along their paths at that speed) or 'duration' [s], not both");
  }
  if ((speed && !(step.tool_speed > 0.0)) || (duration && !(step.duration > 0.0))) {
    throw ConfigError("'" + ex.path() + "': 'tool_speed' and 'duration' must be positive");
  }
  ExplicitOptions& o = step.explicit_options;
  const ConfigNode ms = ex.child("mass_scaling");
  if (ms.exists()) {
    o.mass_scaling.mode = parse_mass_scaling_mode(ms.string_or("mode", "none"));
    o.mass_scaling.target_time_step = ms.number_or("target_time_step", 0.0);
    o.mass_scaling.max_added_mass_fraction =
        ms.number_or("max_added_mass_fraction", o.mass_scaling.max_added_mass_fraction);
    o.mass_scaling.dynamic = ms.boolean_or("dynamic", o.mass_scaling.dynamic);
  }
  const ConfigNode ss = ex.child("stable_step");
  if (ss.exists()) {
    o.stable_step.method = parse_stable_step_method(ss.string_or("method", "element_eigenvalue"));
    o.stable_step.safety = ss.number_or("safety", o.stable_step.safety);
    o.stable_step.update_every = ss.integer_or("update_every", o.stable_step.update_every);
    o.stable_step.power_iterations =
        ss.integer_or("power_iterations", o.stable_step.power_iterations);
  }
  o.mass_damping = ex.number_or("damping", o.mass_damping);
  o.contact_stiffness = ex.number_or("contact_stiffness", o.contact_stiffness);
  o.history_every = ex.integer_or("history_every", o.history_every);
  o.snapshot_every = ex.integer_or("snapshot_every", o.snapshot_every);
  o.energy_tolerance = ex.number_or("energy_tolerance", o.energy_tolerance);
  o.energy_limit = ex.number_or("energy_limit", o.energy_limit);
  o.kinetic_ratio_warning = ex.number_or("kinetic_ratio_warning", o.kinetic_ratio_warning);
  o.penetration_warning = ex.number_or("penetration_warning", o.penetration_warning);
  o.validate(label + ", '" + ex.path() + "'");
}

}  // namespace

Configuration parse_configuration(const json::Value& document, const std::string& source,
                                  bool strict, const std::string& base_directory) {
  if (!document.is_object()) {
    throw ConfigError(source + ": the top-level configuration must be a JSON object");
  }
  const ConfigNode root(document, "");

  Configuration config;
  config.document = document;
  config.source_path = source;
  config.name = root.string_or("name", "case");
  config.description = root.string_or("description", "");
  // Informational only, but read so that it is not flagged as an unused key.
  (void)root.string_or("units", "SI");

  // --- mesh ---------------------------------------------------------------
  {
    const ConfigNode mesh = root.require("mesh");
    // Required: before the 2-D meshes went, a missing type meant structured_quad,
    // so a default of structured_hex would silently run decks that used to fail.
    const std::string type = mesh.require("type").string();
    if (type == "structured_hex") {
      config.mesh_kind = MeshKind::StructuredHex;
    } else if (type == "file") {
      config.mesh_kind = MeshKind::File;
    } else {
      throw ConfigError("'" + mesh.path() + ".type' must be structured_hex or file; got '" +
                        type + "'");
    }
    if (config.mesh_kind == MeshKind::File) {
      for (const char* key : {"nx", "ny", "nz", "lx", "ly", "lz", "x0", "y0", "z0"}) {
        if (mesh.child(key).exists()) {
          throw ConfigError("'" + mesh.path() + "." + key +
                            "' does not apply to a mesh read from a file: its resolution "
                            "and extent come from the mesher. Remove the key, or refine "
                            "the mesh in the generator");
        }
      }
      MeshFileConfig& file = config.mesh_file;
      file.path = mesh.require("path").string();
      std::filesystem::path resolved(file.path);
      if (resolved.is_relative() && !base_directory.empty()) {
        resolved = std::filesystem::path(base_directory) / resolved;
      }
      file.resolved_path = resolved.lexically_normal().string();
      file.read.format = mesh.string_or("format", "auto");
      file.read.scale = mesh.number_or("scale", 1.0);
      if (!(file.read.scale > 0.0) || !std::isfinite(file.read.scale)) {
        std::ostringstream os;
        os << "'" << mesh.path() << ".scale' must be a positive factor (e.g. 0.001 for "
           << "a mesh in millimetres), got " << file.read.scale;
        throw ConfigError(os.str());
      }
      file.read.merge_duplicate_nodes = mesh.boolean_or("merge_duplicate_nodes", false);
      file.read.duplicate_tolerance = mesh.number_or("duplicate_tolerance", 0.0);
      // The reader's messages name the file and line; the deck key is added
      // in front, and the error keeps its category (I/O versus mesh).
      const std::string where = "'" + mesh.path() + ".path' = \"" + file.path + "\": ";
      try {
        config.file_mesh = std::make_shared<const Mesh>(
            read_mesh_file(file.resolved_path, file.read, &config.mesh_report));
      } catch (const IoError& e) {
        throw IoError(where + e.what());
      } catch (const MeshError& e) {
        throw MeshError(where + e.what());
      } catch (const ConfigError& e) {
        throw ConfigError(where + e.what());
      }
    } else {
      config.mesh_spec.nx = mesh.require("nx").integer();
      config.mesh_spec.ny = mesh.require("ny").integer();
      config.mesh_spec.lx = mesh.positive_number("lx");
      config.mesh_spec.ly = mesh.positive_number("ly");
      config.mesh_spec.x0 = mesh.number_or("x0", 0.0);
      config.mesh_spec.y0 = mesh.number_or("y0", 0.0);
      config.mesh_spec.nz = mesh.require("nz").integer();
      config.mesh_spec.lz = mesh.positive_number("lz");
      config.mesh_spec.z0 = mesh.number_or("z0", 0.0);
    }
  }
  const int dim = config.dim();

  // --- material -----------------------------------------------------------
  {
    const ConfigNode mat = root.require("material");
    config.set_material(parse_material(mat, "material", dim));
    int index = 0;
    for (const ConfigNode& mr : root.array("material_regions")) {
      const std::string name = mr.string_or("name", "material_region" + std::to_string(index++));
      config.material_regions.push_back(
          {name, parse_region(mr.require("region"), name, dim),
           parse_material(mr.require("material"), name, dim)});
    }
  }

  // --- model --------------------------------------------------------------
  {
    const ConfigNode model = root.child("model");
    // A unit out-of-plane thickness is the conventional default for a 2-D
    // plane problem; it applies whether or not a "model" section is present.
    // A solid mesh has no thickness and rejects any other value.
    config.thickness = model.number_or("thickness", 1.0);
    if (!(config.thickness > 0.0)) {
      std::ostringstream os;
      os << "'model.thickness' must be positive, got " << config.thickness << " m";
      throw ConfigError(os.str());
    }
    if (dim == 3 && config.thickness != 1.0) {
      std::ostringstream os;
      os << "'model.thickness' is " << config.thickness << " m but the "
         << config.describe_mesh() << " mesh is a solid with no thickness; remove the key";
      throw ConfigError(os.str());
    }
    config.stress_state = parse_stress_state(
        model.string_or("stress_state", dim == 3 ? "three_dimensional" : "plane_stress"));
    if (stress_state_dimension(config.stress_state) != dim) {
      std::ostringstream os;
      os << "'model.stress_state' = \"" << to_string(config.stress_state) << "\" is a "
         << stress_state_dimension(config.stress_state) << "-D idealisation but the "
         << config.describe_mesh() << " mesh is " << dim << "-D";
      throw ConfigError(os.str());
    }
    const ConfigNode integ = model.child("integration");
    config.integration.stiffness_points = integ.integer_or("stiffness_points", 2);
    config.integration.mass_points = integ.integer_or("mass_points", 3);
    // "face_points" (or the older "edge_points") set the face rule of the
    // traction loads SparLab no longer has; the keys are still accepted.
    (void)integ.integer_or("face_points", integ.integer_or("edge_points", 2));
    // A Hex8 sheet's rule through its thickness (0: the in-plane count) and
    // the element formulation; FemModel checks them against the mesh.
    config.integration.thickness_points = integ.integer_or("thickness_points", 0);
    if (config.integration.thickness_points < 0 ||
        config.integration.thickness_points > kMaxThicknessPoints) {
      std::ostringstream os;
      os << "'" << integ.path() << ".thickness_points' is "
         << config.integration.thickness_points << "; it must be 1 to " << kMaxThicknessPoints;
      throw ConfigError(os.str());
    }
    const std::string direction = integ.string_or("thickness_direction", "z");
    if (direction == "x" || direction == "y" || direction == "z") {
      config.integration.thickness_axis = direction[0] - 'x';
    } else {
      throw ConfigError("'" + integ.path() + ".thickness_direction' is \"" + direction +
                        "\"; expected \"x\", \"y\" or \"z\"");
    }
    config.integration.formulation =
        parse_element_formulation(model.string_or("element_formulation", "standard"));
  }

  // --- boundary conditions ------------------------------------------------
  {
    const std::vector<ConfigNode> bcs = root.array("boundary_conditions");
    // A forming deck may give every step its own constraints instead.
    if (bcs.empty() && !root.child("forming").exists()) {
      throw ConfigError(
          "'boundary_conditions' is missing or empty; an unconstrained model has a "
          "singular stiffness matrix");
    }
    int index = 0;
    for (const ConfigNode& bc : bcs) {
      config.constraints.push_back(
          parse_displacement_constraint(bc, "bc" + std::to_string(index++), dim));
    }
  }

  // --- load case ----------------------------------------------------------
  // The forming analysis applies no load case (its tools and prescribed
  // displacements drive it): the model gets one empty case so that it is
  // complete.
  {
    LoadCaseSpec spec;
    spec.name = "forming";
    config.load_cases.push_back(spec);
  }

  // --- forming ------------------------------------------------------------
  {
    const ConfigNode fm = root.child("forming");
    FormingConfig& c = config.forming;
    c.enabled = fm.exists();
    FormingOptions& o = c.options;
    if (c.enabled) {
      o.kinematics = parse_kinematics(fm.string_or("kinematics", "finite"));
      o.law = parse_hyperelastic_model(fm.string_or("material_model", "saint_venant_kirchhoff"));
      const ConfigNode md = fm.child("mean_dilatation");
      if (md.exists()) {
        o.mean_dilatation = md.raw()->is_bool()
                                ? (md.boolean() ? MeanDilatation::All : MeanDilatation::None)
                                : parse_mean_dilatation(md.string());
      }
      o.friction_tangent = parse_friction_tangent(fm.string_or("friction_tangent", "exact"));
      const std::string solver = fm.string_or("solver", "auto");
      if (solver != "auto" && solver != "eigen") {
        throw ConfigError("'" + fm.path() + ".solver' must be \"auto\" (SuiteSparse when built "
                          "in) or \"eigen\", got \"" + solver + "\"");
      }
      o.suitesparse = solver == "auto";

      // Tools.
      std::set<std::string> tool_names;
      int index = 0;
      for (const ConfigNode& t : fm.array("tools")) {
        RigidTool tool;
        tool.name = t.string_or("name", "tool" + std::to_string(index++));
        const std::string label = "'" + t.path() + "' (tool '" + tool.name + "')";
        if (!tool_names.insert(tool.name).second) {
          throw ConfigError("forming tool name '" + tool.name + "' is used twice");
        }
        tool.shape = parse_tool_shape(t.string_or("shape", "sphere"));
        if (tool.shape != RigidTool::Shape::Plane) {
          tool.radius = t.require("radius").number();
          if (!(tool.radius > 0.0)) {
            throw ConfigError(label + ": 'radius' must be positive");
          }
        }
        if (tool.shape == RigidTool::Shape::Plane) {
          tool.normal = t.require("normal").vector3(dim);
        }
        if (tool.shape == RigidTool::Shape::Cylinder && dim == 3) {
          tool.axis = t.vector3_or("axis", Vector3::UnitZ(), 3);
        }
        tool.surface = parse_region(t.require("surface"), tool.name + "_surface", dim);
        tool.friction = t.number_or("friction", 0.0);
        if (!(tool.friction >= 0.0)) throw ConfigError(label + ": 'friction' must be >= 0");
        tool.penalty = t.number_or("penalty", tool.penalty);
        tool.tangential_ratio = t.number_or("tangential_penalty", tool.tangential_ratio);
        const ConfigNode path = t.require("trajectory");
        const ConfigNode file = path.child("file");
        if (file.exists() == (path.child("times").exists() || path.child("points").exists())) {
          throw ConfigError("'" + path.path() + "' needs either a 'file' (CSV with the header "
                            "t,x,y,z) or 'times' and 'points', not both and not neither");
        }
        if (file.exists()) {
          std::filesystem::path resolved(file.string());
          if (resolved.is_relative() && !base_directory.empty()) {
            resolved = std::filesystem::path(base_directory) / resolved;
          }
          tool.trajectory = ToolTrajectory::from_csv(resolved.string());
        } else {
          tool.trajectory.times = path.require("times").number_list();
          for (const ConfigNode& point : path.require("points").items()) {
            tool.trajectory.points.push_back(point.vector3(dim));
          }
        }
        tool.validate(dim);
        o.tools.push_back(std::move(tool));
      }

      // Steps.
      std::set<std::string> step_names;
      std::set<std::string> previous_tools;
      Scalar previous_end = -std::numeric_limits<Scalar>::infinity();
      index = 0;
      const std::vector<ConfigNode> steps = fm.array("steps");
      if (steps.empty()) throw ConfigError("'" + fm.path() + ".steps' is missing or empty");
      for (const ConfigNode& st : steps) {
        FormingStep step;
        step.name = st.string_or("name", "step" + std::to_string(index + 1));
        ++index;
        const std::string label = "'" + st.path() + "' (step '" + step.name + "')";
        if (!step_names.insert(step.name).second) {
          throw ConfigError("forming step name '" + step.name + "' is used twice");
        }
        step.type = parse_step_type(st.string_or("type", "form"));
        const bool release = step.type == FormingStep::Type::Release;
        std::set<std::string> listed;
        for (const std::string& name : string_list(st, "tools")) {
          if (tool_names.count(name) == 0) {
            throw ConfigError(label + " names tool '" + name + "', which 'forming.tools' does "
                              "not define");
          }
          if (release && previous_tools.count(name) == 0) {
            throw ConfigError(label + " is a release that keeps tool '" + name + "', which the "
                              "step before it does not list; a release can only keep tools "
                              "already in place");
          }
          listed.insert(name);
          step.tools.push_back(name);
        }
        const ConfigNode window = st.child("time");
        if (window.exists()) {
          const std::vector<Scalar> w = window.number_list();
          if (w.size() != 2 || !(w[1] > w[0])) {
            throw ConfigError(label + ": 'time' must be [t_begin, t_end] with t_end > t_begin");
          }
          if (w[0] < previous_end) {
            std::ostringstream os;
            os << label << ": its time window starts at " << w[0] << " s, before the step "
               << "before it ended (" << previous_end << " s); the pseudo-time cannot run "
               << "backwards";
            throw ConfigError(os.str());
          }
          step.t_begin = w[0];
          step.t_end = w[1];
          previous_end = w[1];
        }
        const ConfigNode explicit_block = st.child("explicit");
        if (step.type == FormingStep::Type::FormExplicit) {
          // Explicit forming: central differences over the window's physical
          // time (ExplicitDynamics.hpp); its steps follow from the stable
          // time step, not from the implicit increments.
          if (st.child("max_tool_travel").exists() || st.child("increments").exists()) {
            throw ConfigError(label + ": 'max_tool_travel' and 'increments' set the increments "
                                      "of an implicit step; a form_explicit step takes its time "
                                      "steps from the stability limit (its 'explicit' block)");
          }
          if (!explicit_block.exists()) {
            throw ConfigError(label + " (form_explicit) needs an 'explicit' block with its "
                                      "'tool_speed' [m/s] or 'duration' [s]");
          }
          parse_explicit_step(explicit_block, step, label);
        } else {
          if (explicit_block.exists()) {
            throw ConfigError(label + ": an 'explicit' block belongs to a step of type "
                                      "\"form_explicit\"");
          }
          step.max_tool_travel = st.number_or("max_tool_travel", 0.0);
          step.increments = st.integer_or("increments", 0);
        }
        if (!(step.max_tool_travel >= 0.0) || step.increments < 0) {
          throw ConfigError(label + ": 'max_tool_travel' and 'increments' must not be negative");
        }
        int bc_index = 0;
        for (const ConfigNode& bc : st.array("boundary_conditions")) {
          StepConstraint sc;
          sc.constraint = parse_displacement_constraint(
              bc, step.name + "_bc" + std::to_string(bc_index++), dim);
          sc.name = sc.constraint.region.name;
          sc.mode = parse_constraint_mode(bc.string_or("mode", "hold"));
          // A held constraint keeps its DOFs where the step finds them: a
          // non-zero value there would be ignored, while the same entry at
          // the top level prescribes it. (Zero, "fixed", is accepted: decks
          // write it for clamps and supports that have not moved.)
          bool moved = false;
          for (int k = 0; k < dim; ++k) {
            moved = moved || (sc.constraint.fixes(k) && sc.constraint.value(k) != 0.0);
          }
          if (sc.mode == StepConstraint::Mode::Hold && moved) {
            throw ConfigError("'" + bc.path() + ".value' is not zero, but the constraint's mode "
                              "is \"hold\" (the default), which keeps its DOFs where the step "
                              "finds them; give \"mode\": \"absolute\" to move them to the "
                              "value over the step");
          }
          step.constraints.push_back(std::move(sc));
        }
        previous_tools = listed;
        o.steps.push_back(std::move(step));
      }

      const ConfigNode newton = fm.child("newton");
      o.max_iterations = newton.integer_or("max_iterations", o.max_iterations);
      o.residual_tolerance = newton.number_or("residual_tolerance", o.residual_tolerance);
      o.displacement_tolerance =
          newton.number_or("displacement_tolerance", o.displacement_tolerance);
      o.line_search = newton.boolean_or("line_search", o.line_search);
      o.max_cuts = newton.integer_or("max_cuts", o.max_cuts);
      o.max_increments = newton.integer_or("max_increments", o.max_increments);
      if (o.max_iterations < 1 || o.max_cuts < 0 || o.max_increments < 1 ||
          !(o.residual_tolerance > 0.0) || !(o.displacement_tolerance > 0.0)) {
        throw ConfigError("'forming.newton' needs max_iterations >= 1, max_cuts >= 0, "
                          "max_increments >= 1 and positive tolerances");
      }

      const ConfigNode out = fm.child("output");
      c.write_vtk = out.boolean_or("vtk", true);
      const ConfigNode snapshots = out.child("snapshots");
      if (snapshots.exists()) {
        if (snapshots.raw()->is_number()) {
          c.snapshots = FormingConfig::Snapshots::Stride;
          c.snapshot_stride = snapshots.integer();
          if (c.snapshot_stride < 1) {
            throw ConfigError("'" + snapshots.path() + "' must be \"steps\", \"none\" or an "
                              "increment stride >= 1");
          }
          o.snapshot_stride = c.snapshot_stride;
        } else {
          const std::string mode = snapshots.string();
          if (mode == "steps") {
            c.snapshots = FormingConfig::Snapshots::Steps;
          } else if (mode == "none") {
            c.snapshots = FormingConfig::Snapshots::None;
          } else {
            throw ConfigError("'" + snapshots.path() + "' must be \"steps\", \"none\" or an "
                              "increment stride >= 1, got \"" + mode + "\"");
          }
        }
      }
    }
  }

  // --- unknown keys -------------------------------------------------------
  const std::vector<std::string> unused = root.unused_keys();
  if (!unused.empty()) {
    std::ostringstream os;
    os << source << " contains " << unused.size()
       << " key(s) that SparLab does not recognise (a misspelled key silently takes "
          "its default, so this is reported):";
    for (const std::string& key : unused) os << "\n  - " << key;
    if (strict) throw ConfigError(os.str());
    log::warn(os.str());
  }

  return config;
}

Configuration load_configuration(const std::string& path, bool strict) {
  const json::Value document = json::parse_file(path);
  const std::string base = std::filesystem::path(path).parent_path().string();
  return parse_configuration(document, path, strict, base.empty() ? "." : base);
}

Mesh build_mesh(const Configuration& config) {
  switch (config.mesh_kind) {
    case MeshKind::StructuredHex: return make_structured_hex_mesh(config.mesh_spec);
    case MeshKind::File:
      if (config.file_mesh == nullptr) {
        throw ConfigError("the configuration names a mesh file that has not been read");
      }
      return *config.file_mesh;
  }
  throw ConfigError("unhandled mesh type");
}

FemModel build_model(const Configuration& config) {
  Mesh mesh = build_mesh(config);
  FemModel model(std::move(mesh), config.material(), config.thickness,
                 config.stress_state, config.integration);
  for (const MaterialRegion& mr : config.material_regions) {
    const std::vector<Index> elements = mr.region.select_elements(model.mesh());
    if (elements.empty()) {
      throw ConfigError("material region '" + mr.name + "' selected no element; check its "
                        "coordinates or group name");
    }
    model.assign_material(mr.material, elements);
    log::info("material region '", mr.name, "': ", elements.size(), " element(s) of '",
              mr.material.name(), "'");
  }
  model.constraints() = config.constraints;
  model.load_case_specs() = config.load_cases;
  model.finalize();
  return model;
}

}  // namespace sparlab
