// Where this fits: see include/sparlab/io/FormingWriter.hpp

#include "sparlab/io/FormingWriter.hpp"

#include "sparlab/io/CsvWriter.hpp"
#include "sparlab/io/VtkWriter.hpp"

#include <cstdio>

namespace sparlab {
namespace {

std::string number(Scalar v) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.17g", v);
  return buffer;
}

json::Value num(Scalar v) { return json::Value::make_number(v); }
json::Value str(const std::string& s) { return json::Value::make_string(s); }

json::Value vec(const Vector3& v, int dim) {
  json::Value out = json::Value::make_array();
  for (int k = 0; k < dim; ++k) out.push_back(num(v(k)));
  return out;
}

/// Nodes (reference coordinates and displacement) as the contract's CSV.
void write_nodes(const std::string& path, const Mesh& mesh, const Vector& u) {
  const int dim = mesh.dim();
  CsvWriter csv(path, {"node", "X", "Y", "Z", "ux", "uy", "uz"});
  for (Index n = 0; n < mesh.num_nodes(); ++n) {
    const Vector3 x = mesh.node(n);
    Vector3 d = Vector3::Zero();
    d.head(dim) = u.segment(n * dim, dim);
    csv.row(n, {x.x(), x.y(), dim == 3 ? x.z() : 0.0, d.x(), d.y(), dim == 3 ? d.z() : 0.0});
  }
  csv.close();
}

void write_vtk(const std::string& path, const Mesh& mesh, const Vector& u,
               const Vector* plastic, const Vector* von_mises) {
  const int dim = mesh.dim();
  VtkWriter vtk(mesh, "SparLab forming");
  vtk.add_point_vectors("displacement", u);
  Vector magnitude(mesh.num_nodes());
  for (Index n = 0; n < mesh.num_nodes(); ++n) magnitude(n) = u.segment(n * dim, dim).norm();
  vtk.add_point_scalars("displacement_magnitude", magnitude);
  if (plastic) vtk.add_cell_scalars("eq_plastic_strain", *plastic);
  if (von_mises) vtk.add_cell_scalars("von_mises", *von_mises);
  vtk.write(path);
}

}  // namespace

std::string forming_step_stem(std::size_t index, const std::string& name) {
  std::string clean;
  for (char c : name) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                    c == '-' || c == '_';
    clean.push_back(ok ? c : '_');
  }
  if (clean.empty()) clean = "step";
  return "step_" + std::to_string(index + 1) + "_" + clean;
}

std::vector<std::string> write_forming_step(const ResultWriter& writer, const FemModel& model,
                                            std::size_t index, const FormingStepResult& s,
                                            bool vtk, bool step_files) {
  const Mesh& mesh = model.mesh();
  std::vector<std::string> files;
  if (!s.completed || !step_files) return files;
  const std::string stem = forming_step_stem(index, s.name);
  write_nodes(writer.file(stem + "_nodes.csv"), mesh, s.displacement);
  files.push_back(stem + "_nodes.csv");
  {
    CsvWriter csv(writer.file(stem + "_elements.csv"),
                  {"element", "eq_plastic_strain", "von_mises_Pa"});
    for (Index e = 0; e < mesh.num_elements(); ++e) {
      csv.row(e, {s.element_plastic_strain(e), s.element_von_mises(e)});
    }
    csv.close();
    files.push_back(stem + "_elements.csv");
  }
  if (vtk) {
    write_vtk(writer.file(stem + ".vtk"), mesh, s.displacement, &s.element_plastic_strain,
              &s.element_von_mises);
    files.push_back(stem + ".vtk");
  }
  if (s.explicit_step) {
    // The energy history of an explicit step (ExplicitDynamics.hpp).
    CsvWriter csv(writer.file(stem + "_energy.csv"),
                  {"step", "t_s", "pseudo_t_s", "time_step_s", "kinetic_J", "internal_work_J",
                   "stored_J", "elastic_J", "plastic_dissipation_J", "contact_normal_work_J",
                   "contact_friction_work_J", "damping_J", "external_work_J",
                   "mass_scaling_work_J", "energy_error_J", "kinetic_ratio"});
    // The kinetic energy over the step's reference energy (its
    // kinetic_ratio_basis in summary.json), unfiltered.
    const bool elastic_basis = s.explicit_result.kinetic_ratio_basis == "elastic energy";
    for (const ExplicitRecord& r : s.explicit_result.records) {
      const Scalar reference = elastic_basis ? r.elastic : r.internal_energy;
      csv.raw_row({std::to_string(r.step), number(r.time), number(r.pseudo_time),
                   number(r.time_step), number(r.kinetic), number(r.internal), number(r.stored),
                   number(r.elastic), number(r.plastic), number(r.contact_normal),
                   number(r.contact_friction), number(r.damping), number(r.external),
                   number(r.mass_scaling), number(r.error),
                   number(reference > 0.0 ? r.kinetic / reference : 0.0)});
    }
    csv.close();
    files.push_back(stem + "_energy.csv");
  }
  for (const FormingSnapshot& snap : s.snapshots) {
    const std::string name = stem + "_inc_" + std::to_string(snap.increment);
    write_nodes(writer.file(name + "_nodes.csv"), mesh, snap.displacement);
    files.push_back(name + "_nodes.csv");
    if (vtk) {
      write_vtk(writer.file(name + ".vtk"), mesh, snap.displacement, nullptr, nullptr);
      files.push_back(name + ".vtk");
    }
  }
  return files;
}

void write_tool_forces(const ResultWriter& writer, const FemModel& model,
                       const FormingOptions& options, const FormingResult& result) {
  CsvWriter tools(writer.file("tool_forces.csv"),
                  {"step", "increment", "t", "tool", "cx", "cy", "cz", "fx", "fy", "fz",
                   "active_nodes", "max_penetration_m"});
  const int dim = model.mesh().dim();
  for (std::size_t k = 0; k < result.steps.size(); ++k) {
    for (const FormingIncrement& inc : result.steps[k].increments) {
      for (const ToolRecord& t : inc.tools) {
        const auto c = [&](const Vector3& v, int i) { return i < dim ? v(i) : 0.0; };
        tools.raw_row({std::to_string(k + 1), std::to_string(inc.index), number(inc.time),
                       options.tools[t.tool].name, number(c(t.centre, 0)), number(c(t.centre, 1)),
                       number(c(t.centre, 2)), number(c(t.force, 0)), number(c(t.force, 1)),
                       number(c(t.force, 2)), std::to_string(t.active_nodes),
                       number(t.max_penetration)});
      }
    }
  }
  tools.close();
}

json::Value forming_summary_json(const Configuration& config, const FemModel& model,
                                 const FormingOptions& options, const FormingResult& result,
                                 Scalar runtime_s, const std::string& version,
                                 const std::vector<std::string>& files) {
  const Mesh& mesh = model.mesh();
  const int dim = mesh.dim();
  json::Value out = json::Value::make_object();
  out.set("case", str(config.name));
  out.set("sparlab_version", str(version));
  out.set("completed", json::Value::make_bool(result.completed));
  out.set("termination", str(result.termination));
  out.set("runtime_s", num(runtime_s));

  json::Value timing = json::Value::make_object();
  for (const auto& [name, seconds] : result.timing.totals()) timing.set(name + "_s", num(seconds));
  timing.set("increments", num(result.total_increments));
  timing.set("iterations", num(result.total_iterations));
  timing.set("cuts", num(result.total_cuts));
  timing.set("explicit_steps", num(static_cast<Scalar>(result.total_explicit_steps)));
  timing.set("linear_solver", str(result.linear_solver));
  timing.set("failed_factorisations", str(result.failed_factorisations));
  timing.set("suitesparse", json::Value::make_bool(options.suitesparse &&
                                                   forming_suitesparse_available()));
  out.set("timing", timing);

  json::Value analysis = json::Value::make_object();
  analysis.set("kinematics", str(result.kinematics));
  analysis.set("plastic", json::Value::make_bool(result.plastic));
  analysis.set("mean_dilatation", json::Value::make_bool(result.mean_dilatation));
  analysis.set("friction_tangent", str(to_string(options.friction_tangent)));
  analysis.set("residual_tolerance", num(options.residual_tolerance));
  analysis.set("displacement_tolerance", num(options.displacement_tolerance));
  analysis.set("max_iterations", num(options.max_iterations));
  analysis.set("max_cuts", num(options.max_cuts));
  analysis.set("line_search", json::Value::make_bool(options.line_search));
  // The explicit steps' tolerances (their settings are under steps[k].explicit).
  {
    json::Value tolerances = json::Value::make_array();
    for (const FormingStep& st : options.steps) {
      if (st.type != FormingStep::Type::FormExplicit) continue;
      json::Value jt = json::Value::make_object();
      jt.set("step", str(st.name));
      jt.set("energy_tolerance", num(st.explicit_options.energy_tolerance));
      jt.set("energy_limit", num(st.explicit_options.energy_limit));
      jt.set("kinetic_ratio_warning", num(st.explicit_options.kinetic_ratio_warning));
      jt.set("penetration_warning", num(st.explicit_options.penetration_warning));
      jt.set("max_added_mass_fraction",
             num(st.explicit_options.mass_scaling.max_added_mass_fraction));
      jt.set("stable_step_safety", num(st.explicit_options.stable_step.safety));
      tolerances.push_back(jt);
    }
    analysis.set("explicit_tolerances", tolerances);
  }
  out.set("analysis", analysis);

  json::Value steps = json::Value::make_array();
  for (std::size_t k = 0; k < result.steps.size(); ++k) {
    const FormingStepResult& s = result.steps[k];
    json::Value js = json::Value::make_object();
    js.set("name", str(s.name));
    js.set("type", str(to_string(s.type)));
    js.set("completed", json::Value::make_bool(s.completed));
    js.set("increments", num(static_cast<Scalar>(s.increments.size())));
    js.set("iterations", num(s.iterations));
    js.set("cuts", num(s.cuts));
    js.set("max_plastic_strain", num(s.max_plastic_strain));
    js.set("reaction_norm_N", num(s.reaction_norm));
    json::Value warnings = json::Value::make_array();
    for (const std::string& w : s.warnings) warnings.push_back(str(w));
    js.set("warnings", warnings);
    js.set("termination", str(s.termination));
    js.set("files_stem", str(forming_step_stem(k, s.name)));
    js.set("t_begin_s", num(s.t_begin));
    js.set("t_end_s", num(s.t_end));
    js.set("tools", json::array_of(s.tools));
    js.set("constrained_dofs", num(s.constrained_dofs));
    js.set("start_imbalance_N", num(s.start_imbalance));
    js.set("reference_force_N", num(s.reference_force));
    js.set("max_displacement_change_m", num(s.max_displacement_change));
    // Every step the analysis records carries its displacement, a stopped
    // one included; a hand-made result without it reports 0.
    Scalar max_u = 0.0;
    if (s.displacement.size() == model.dofs().num_dofs()) {
      for (Index n = 0; n < mesh.num_nodes(); ++n) {
        max_u = std::max(max_u, s.displacement.segment(n * dim, dim).norm());
      }
    }
    js.set("max_displacement_m", num(max_u));
    if (s.explicit_step && k < options.steps.size()) {
      // The explicit integration: time steps, mass scaling, validity checks.
      const ExplicitResult& er = s.explicit_result;
      const ExplicitOptions& eo = options.steps[k].explicit_options;
      json::Value je = json::Value::make_object();
      je.set("steps", num(static_cast<Scalar>(er.steps)));
      je.set("physical_time_s", num(er.duration));
      je.set("tool_speed_m_s", num(s.explicit_tool_speed));
      je.set("duration_s", num(options.steps[k].duration));
      je.set("time_step_s", num(er.time_step));
      je.set("min_time_step_s", num(er.min_time_step));
      je.set("final_time_step_s", num(er.final_time_step));
      je.set("stable_time_step_s", num(er.stable_time_step));
      je.set("scaled_stable_time_step_s", num(er.scaled_stable_time_step));
      je.set("stable_step_method", str(to_string(eo.stable_step.method)));
      je.set("safety", num(eo.stable_step.safety));
      je.set("step_updates", num(er.step_updates));
      je.set("mass_scaling", str(to_string(eo.mass_scaling.mode)));
      je.set("target_time_step_s", num(eo.mass_scaling.target_time_step));
      je.set("dynamic_mass_scaling", json::Value::make_bool(eo.mass_scaling.dynamic));
      je.set("mass_updates", num(er.mass_updates));
      je.set("mass_scale_max", num(er.max_mass_scale));
      je.set("scaled_elements", num(er.scaled_elements));
      je.set("physical_mass_kg", num(er.physical_mass));
      je.set("scaled_mass_kg", num(er.scaled_mass));
      je.set("added_mass_fraction", num(er.added_mass_fraction));
      je.set("max_added_mass_fraction", num(eo.mass_scaling.max_added_mass_fraction));
      je.set("damping_per_s", num(eo.mass_damping));
      je.set("contact_stiffness", num(eo.contact_stiffness));
      je.set("contact", json::Value::make_bool(er.contact));
      je.set("max_kinetic_ratio", num(er.max_kinetic_ratio));
      je.set("peak_kinetic_ratio", num(er.peak_kinetic_ratio));
      je.set("kinetic_ratio_basis", str(er.kinetic_ratio_basis));
      je.set("kinetic_ratio_warning", num(eo.kinetic_ratio_warning));
      je.set("max_penetration_ratio", num(er.max_penetration_ratio));
      je.set("penetration_warning", num(eo.penetration_warning));
      je.set("max_energy_error", num(er.max_energy_error));
      je.set("energy_tolerance", num(eo.energy_tolerance));
      je.set("energy_limit", num(eo.energy_limit));
      je.set("history_every", num(eo.history_every));
      je.set("kernel", str(er.kernel));
      json::Value jt = json::Value::make_object();
      for (const auto& [name, seconds] : er.timing.totals()) jt.set(name + "_s", num(seconds));
      je.set("timing", jt);
      je.set("wall_s", num(er.timing.get("total")));
      je.set("energy_file", str(forming_step_stem(k, s.name) + "_energy.csv"));
      js.set("explicit", je);
    }
    steps.push_back(js);
  }
  out.set("steps", steps);

  json::Value tools = json::Value::make_array();
  for (std::size_t t = 0; t < options.tools.size(); ++t) {
    const RigidTool& tool = options.tools[t];
    json::Value jt = json::Value::make_object();
    jt.set("name", str(tool.name));
    jt.set("shape", str(to_string(tool.shape)));
    if (tool.shape != RigidTool::Shape::Plane) jt.set("radius_m", num(tool.radius));
    if (tool.shape == RigidTool::Shape::Plane) jt.set("normal", vec(tool.normal.normalized(), dim));
    if (tool.shape == RigidTool::Shape::Cylinder && dim == 3) {
      jt.set("axis", vec(tool.axis.normalized(), 3));
    }
    jt.set("friction", num(tool.friction));
    jt.set("penalty", num(tool.penalty));
    jt.set("tangential_penalty", num(tool.tangential_ratio));
    jt.set("trajectory_knots", num(static_cast<Scalar>(tool.trajectory.times.size())));
    jt.set("trajectory_t_start_s", num(tool.trajectory.start()));
    jt.set("trajectory_t_end_s", num(tool.trajectory.end()));
    Scalar length = 0.0;
    for (std::size_t i = 1; i < tool.trajectory.points.size(); ++i) {
      length += (tool.trajectory.points[i] - tool.trajectory.points[i - 1]).norm();
    }
    jt.set("trajectory_length_m", num(length));
    // The largest force the body exerted on the tool, and when.
    Scalar peak = 0.0;
    Vector3 peak_force = Vector3::Zero();
    Scalar peak_time = 0.0;
    int max_nodes = 0;
    Scalar max_penetration = 0.0;
    for (const FormingStepResult& s : result.steps) {
      for (const FormingIncrement& inc : s.increments) {
        for (const ToolRecord& r : inc.tools) {
          if (r.tool != t) continue;
          if (r.force.norm() > peak) {
            peak = r.force.norm();
            peak_force = r.force;
            peak_time = inc.time;
          }
          max_nodes = std::max(max_nodes, r.active_nodes);
          max_penetration = std::max(max_penetration, r.max_penetration);
        }
      }
    }
    jt.set("peak_force_N", vec(peak_force, dim));
    jt.set("peak_force_time_s", num(peak_time));
    jt.set("max_active_nodes", num(max_nodes));
    jt.set("max_penetration_m", num(max_penetration));
    tools.push_back(jt);
  }
  out.set("tools", tools);

  json::Value jm = json::Value::make_object();
  jm.set("source", str(config.describe_mesh()));
  jm.set("element_type", str(to_string(mesh.element_type())));
  jm.set("element_formulation", str(to_string(model.integration().formulation)));
  jm.set("stiffness_rule_points",
         num(static_cast<Scalar>(model.element().integration_rule(model.integration()).size())));
  jm.set("dim", num(dim));
  jm.set("nodes", num(mesh.num_nodes()));
  jm.set("elements", num(mesh.num_elements()));
  jm.set("dofs", num(model.dofs().num_dofs()));
  const BoundingBox box = mesh.bounding_box();
  jm.set("bounding_box_min_m", vec(box.lower, dim));
  jm.set("bounding_box_max_m", vec(box.upper, dim));
  out.set("mesh", jm);

  json::Value warnings = json::Value::make_array();
  for (const std::string& w : result.warnings) warnings.push_back(str(w));
  out.set("warnings", warnings);
  out.set("files", json::array_of(files));
  out.set("provenance", make_provenance(config));
  return out;
}

}  // namespace sparlab
