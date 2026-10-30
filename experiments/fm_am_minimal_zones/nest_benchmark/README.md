# NEST module for paper calculations

The files here build the two-threshold PWA astrocyte used by the paper's
network scripts in `astrocyte_lab/experiments/paper_network` and
`astrocyte_lab/experiments/astro_ratio_network`. Use the paper workflow in
`paper/local_results/README.md` and `scripts/run_paper_local.sh`. The build
is candidate-specific, so the runner rebuilds the module for every selected
model. The upstream-derived NEST files and license are in `module/`.

The local build uses the NEST 3.10 Python wheel and its headers. If necessary,
`setup_build_tools.sh` downloads pinned CMake, Boost and libltdl headers under
`module/deps`, without modifying system directories. Generated code and build
tools are not committed. The runner checks the compiled candidate against
DOP853 before running each model.
