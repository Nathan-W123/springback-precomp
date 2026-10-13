/// \file FormingWriter.hpp
/// \brief The result files of an incremental-forming run (sparlab_form) -
///        an output contract other tools read, documented in
///        docs/forming.md. Names and columns are fixed:
/// \code
///   <out>/summary.json                 case, version, completion, runtime, timing,
///                                      per step and per tool results, mesh
///   <out>/mesh.json                    nodes, connectivity (ResultWriter::write_mesh)
///   <out>/step_<k>_<s>_nodes.csv       node,X,Y,Z,ux,uy,uz at the end of step k
///   <out>/step_<k>_<s>_elements.csv    element,eq_plastic_strain,von_mises_Pa
///   <out>/step_<k>_<s>.vtk             the same fields for ParaView
///   <out>/step_<k>_<s>_inc_<i>_nodes.csv  (and .vtk) snapshots every `stride` increments
///   <out>/step_<k>_<s>_energy.csv      form_explicit steps: the energy history
///                                      (step,t_s,pseudo_t_s,time_step_s,kinetic_J,
///                                      internal_work_J,stored_J,elastic_J,
///                                      plastic_dissipation_J,contact_normal_work_J,
///                                      contact_friction_work_J,damping_J,external_work_J,
///                                      mass_scaling_work_J,energy_error_J,kinetic_ratio)
///   <out>/tool_forces.csv              step,increment,t,tool,cx,cy,cz,fx,fy,fz,
///                                      active_nodes,max_penetration_m
/// \endcode
/// k is the 1-based step index and s its name (characters other than
/// letters, digits, '-' and '_' replaced by '_'). Reference coordinates X
/// and displacements u are in m (Z and uz are 0 on a 2-D model); the
/// element's equivalent plastic strain is the largest of its integration
/// points', its von Mises stress that of its point-averaged Cauchy stress
/// [Pa]; the tool force is the force the body exerts on the tool [N], and
/// (cx, cy, cz) the tool's reference point [m], at the end of each converged
/// increment - for an explicit step every `history_every` steps, the force
/// averaged over them, t the tools' pseudo-time. Only completed steps have
/// files. sparlab_form writes
/// config.json and mesh.json before the first step, each step's files and
/// tool_forces.csv as the step ends, and summary.json last.
#pragma once

#include "sparlab/fem/FemModel.hpp"
#include "sparlab/fem/Forming.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/Json.hpp"
#include "sparlab/io/ResultWriter.hpp"

#include <string>
#include <vector>

namespace sparlab {

/// "step_<k>_<name>" for the step at 0-based index `index`.
std::string forming_step_stem(std::size_t index, const std::string& name);

/// Write the files of the step at 0-based `index` - its node and element
/// CSVs, its VTK file (when `vtk`) and its snapshots - when it completed and
/// `step_files` is set. Returns the names of the files written.
std::vector<std::string> write_forming_step(const ResultWriter& writer, const FemModel& model,
                                            std::size_t index, const FormingStepResult& step,
                                            bool vtk, bool step_files);

/// Write tool_forces.csv with every increment of `result` (so far).
void write_tool_forces(const ResultWriter& writer, const FemModel& model,
                       const FormingOptions& options, const FormingResult& result);

/// The summary.json document of a forming run.
json::Value forming_summary_json(const Configuration& config, const FemModel& model,
                                 const FormingOptions& options, const FormingResult& result,
                                 Scalar runtime_s, const std::string& version,
                                 const std::vector<std::string>& files);

}  // namespace sparlab
