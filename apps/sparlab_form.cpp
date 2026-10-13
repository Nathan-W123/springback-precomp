/// \file sparlab_form.cpp
/// \brief Incremental-forming analysis of one configuration: the deck's
///        `forming` block - rigid tools travelling along their trajectories
///        over the model, then its release onto statically determinate
///        supports (springback) - with the result files of
///        FormingWriter.hpp (docs/forming.md).
///
/// It builds the model (which validates the mesh and the material), writes
/// the deck and the mesh, runs the steps in order - writing each completed
/// step's results and the tool forces so far as the step ends - and writes
/// summary.json last. The exit status is 0 when every step completed, 3 when
/// one stopped (its reason on stderr and in summary.json; the files of the
/// steps before it are written), 2 on a configuration error and 4 on an I/O
/// error (AppSupport.hpp, run_guarded).

#include "AppSupport.hpp"

#include "sparlab/core/Exceptions.hpp"
#include "sparlab/core/Timer.hpp"
#include "sparlab/core/Version.hpp"
#include "sparlab/fem/Assembler.hpp"
#include "sparlab/fem/Forming.hpp"
#include "sparlab/io/Config.hpp"
#include "sparlab/io/FormingWriter.hpp"
#include "sparlab/io/ResultWriter.hpp"

#include "sparlab_git_sha.h"

#ifdef SPARLAB_HAVE_OPENMP
#include <omp.h>
#endif

#include <iostream>

using namespace sparlab;

namespace {

/// "<version> (<git revision>)": what a result cache keys on.
std::string version_string() {
  return std::string(SPARLAB_VERSION_STRING) + " (" + SPARLAB_GIT_SHA + ")";
}

}  // namespace

int main(int argc, char** argv) {
  return app::run_guarded([&]() -> int {
    const std::vector<std::string> known = {"config",  "output", "threads", "verbosity",
                                            "strict-config", "no-vtk", "version", "help"};
    app::CommandLine cli(argc, argv, known);
    if (cli.has("version")) {
      std::cout << "sparlab " << version_string() << "\n";
      return 0;
    }
    if (cli.has("help") || argc == 1) {
      return app::print_usage(
          "sparlab_form", "--config <deck.json> [--output <dir>]",
          {{"--config <file>", "input deck with a 'forming' block (tools, steps)"},
           {"--output <dir>", "output directory (default results/<case>)"},
           {"--threads <n>", "OpenMP threads of the element loops (default: all)"},
           {"--no-vtk", "skip the per-step VTK files"},
           {"--strict-config", "treat unknown configuration keys as errors"},
           {"--verbosity <lvl>", "trace|debug|info|warn|error|silent"},
           {"--version", "print the SparLab version and git revision, and exit"},
           {"--help", "show this message"}});
    }
    app::apply_verbosity(cli);
    if (cli.has("threads")) {
      const int threads = cli.integer("threads", 1);
      if (threads < 1) throw ConfigError("--threads must be at least 1");
#ifdef SPARLAB_HAVE_OPENMP
      omp_set_num_threads(threads);
#else
      if (threads > 1) log::warn("built without OpenMP: --threads ", threads, " is ignored");
#endif
    }

    const Configuration config =
        load_configuration(cli.require("config"), cli.has("strict-config"));
    if (!config.forming.enabled) {
      throw ConfigError("'" + config.source_path + "' has no 'forming' block; sparlab_form "
                        "runs the incremental-forming analysis it describes (docs/forming.md)");
    }
    const bool vtk = config.forming.write_vtk && !cli.has("no-vtk");
    const std::string out_dir = cli.value("output", app::default_output_directory(config.name));

    Timer wall;
    FemModel model = build_model(config);
    model.mesh().validate();
    for (const LoadCaseSpec& lc : model.load_case_specs()) {
      if (lc.has_loads()) {
        log::warn("load case '", lc.name, "' carries loads, which the forming analysis does "
                  "not apply: its tools and prescribed displacements drive it");
      }
    }
    Assembler assembler(model);
    log::info("forming '", config.name, "': ", model.mesh().num_elements(), " ",
              to_string(model.mesh().element_type()), " elements, ", model.dofs().num_dofs(),
              " DOFs, ", config.forming.options.tools.size(), " tool(s), ",
              config.forming.options.steps.size(), " step(s); linear algebra ",
              config.forming.options.suitesparse && forming_suitesparse_available()
                  ? "SuiteSparse (CHOLMOD, UMFPACK)"
                  : "Eigen");

    FormingAnalysis analysis(model, assembler, config.forming.options);
    // The output directory, the deck and the mesh before the first step (an
    // unwritable directory fails now, not after hours), each step's files
    // and the tool forces so far as the step ends (a run that is killed
    // keeps the steps it completed), summary.json last.
    ResultWriter writer(out_dir, config);
    writer.write_config();
    writer.write_mesh(model);
    const bool step_files = config.forming.snapshots != FormingConfig::Snapshots::None;
    std::vector<std::string> files = {"summary.json", "config.json", "mesh.json"};
    analysis.set_step_observer([&](const FormingResult& so_far) {
      const std::size_t k = so_far.steps.size() - 1;
      for (std::string& f : write_forming_step(writer, model, k, so_far.steps[k], vtk, step_files)) {
        files.push_back(std::move(f));
      }
      write_tool_forces(writer, model, config.forming.options, so_far);
    });
    const FormingResult result = analysis.run();
    const Scalar runtime = wall.elapsed_seconds();
    files.push_back("tool_forces.csv");
    writer.write_json("summary.json",
                      forming_summary_json(config, model, config.forming.options, result,
                                           runtime, version_string(), files));

    std::cout << "case: " << config.name << "\n";
    std::cout << "  mesh:    " << model.mesh().num_elements() << " elements, "
              << model.dofs().num_dofs() << " DOFs\n";
    for (std::size_t k = 0; k < result.steps.size(); ++k) {
      const FormingStepResult& s = result.steps[k];
      std::cout << "  step " << k + 1 << " '" << s.name << "' (" << to_string(s.type)
                << "): " << (s.completed ? "completed" : "STOPPED") << ", "
                << s.increments.size() << " increment(s), " << s.iterations
                << " iteration(s), " << s.cuts << " cut(s), largest plastic strain "
                << app::format(s.max_plastic_strain) << ", largest displacement change "
                << app::format(s.max_displacement_change) << " m\n";
      if (!s.completed) std::cout << "      " << s.termination << "\n";
      for (const std::string& w : s.warnings) std::cout << "      warning: " << w << "\n";
    }
    std::cout << "  runtime: " << app::format(runtime, 4) << " s (" << result.total_increments
              << " increments, " << result.total_iterations << " Newton iterations; "
              << result.linear_solver << ")\n";
    if (!result.failed_factorisations.empty()) {
      std::cout << "  failed factorisations: " << result.failed_factorisations << "\n";
    }
    std::cout << "  output:  " << writer.directory() << "\n";
    if (!result.completed) {
      std::cerr << "[error] the forming analysis stopped: " << result.termination << "\n";
      return 3;
    }
    return 0;
  });
}
