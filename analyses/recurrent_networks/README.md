# Trained flip-flop networks: scaling grid, escape ladder, known-barrier cells and the drive/gain plane

Eight trained 64-unit rate networks with two stable fixed points (`weights/`) are the systems of known dynamics
behind Fig. 1d,e, Fig. 2d,e, Fig. 3e, Supplementary Fig. 1 and Supplementary Table 14. The shared functions
(network training and loading, fixed-point enumeration, simulated recordings, the equilibrium counter, the
density barrier and the minimum-action tools) are in `network_lib.py`.

| Script | Output | Reported in |
|---|---|---|
| `network_scaling_grid.py --reps 2 --horizon 20000` | `out/network_parameter_grid.csv` (600 rows: 8 networks x 25 (g, s^2) cells x 3 constructions), `.json`, `.log`; the stored rows are the Source Data file `analysis_inputs/stored_results/network_parameter_grid.csv` | Fig. 1d,e; Supplementary Fig. 1 |
| `network_scaling_summary.py` | `out/network_parameter_grid_report.txt`, `out/network_parameter_grid_numbers.json`, `out/all_endpoint_ratios.csv`, `out/endpoint_summary.csv`, `out/supplementary_table_14.csv`, `out/network_scaling_endpoints.json` | Fig. 1d (joint scaling: density barrier x0.912 [0.811, 1.032], lag-1 rate x2.535 [2.017, 3.433], escape hazard x4.745 [3.698, 8.263] in the 5 networks whose escapes resolve at both ends); Fig. 1e (drift only x5.033, diffusion only x0.193; equilibrium count changes in 7/8 and 8/8 networks, joint 1/8); Supplementary Fig. 1 (600 combinations, 66 of 200 escape hazards censored); Supplementary Table 14 |
| `escape_ladder.py --qualify-only`, then workers `--indices 0,4`, `--indices 1,5`, `--indices 2,6`, `--indices 3,7`, then `--resume` | network cache: `cells/cell_<seed>_<gain>.json`, `selftest.json`, `rnn_escape_simulation.json`, `rnn_escape_simulation.csv`, `escape_ladder.log`; the stored output is the Source Data file `analysis_inputs/stored_results/rnn_escape_simulation.json` | escape simulation of the Methods; Fig. 2e right (through `arrhenius_summary.py`) |
| `arrhenius_summary.py` | `../../figure_reproduction/inputs/arrhenius_slopes.json`, `out/escape_ladder_trace.txt` | Fig. 2e right: Arrhenius slope 0.832 [0.796, 0.880] on the minimum action and 0.170 [0.104, 0.234] on the density barrier, 5 of the 7 networks with complete ladders |
| `known_barrier_cells.py --seeds 20260922,20260935,20260948,20260961 --reps 16` | `../../figure_reproduction/inputs/density_barrier_cells.csv` (768 rows), `out/density_barrier_cells.json`, `out/density_barrier_cells.log` | Fig. 2d (two-well cells; their slope is summarised in `density_barrier_summary.json` by the drift-replacement consolidation) and Fig. 2e left (one-well cells, density barrier zero) |
| `drive_gain_plane.py --reps 8` | network cache: `drive_gain_rows/rows_<seed>.json` | Fig. 3e (through `drive_gain_report.py`) |
| `drive_gain_report.py` | `../../figure_reproduction/inputs/drive_gain_separation.csv` (256 rows), `out/drive_gain_plane.csv`, `out/drive_gain_numbers.json`, `out/drive_gain_report.txt` | Fig. 3e: drive versus gain accuracy 0.898 for EEG sampling (chance 0.504, within-network ceiling 0.875); networks at or above 0.80: EEG 8, MEG 5, fMRI 2 of 8 |

## Run order

```bash
# escape ladder (cells on the GPU, workers in parallel), then its summary
python analyses/recurrent_networks/escape_ladder.py --qualify-only
python analyses/recurrent_networks/escape_ladder.py --indices 0,4      # one process per index set
python analyses/recurrent_networks/escape_ladder.py --indices 1,5
python analyses/recurrent_networks/escape_ladder.py --indices 2,6
python analyses/recurrent_networks/escape_ladder.py --indices 3,7
python analyses/recurrent_networks/escape_ladder.py --resume
python analyses/recurrent_networks/arrhenius_summary.py
# known-barrier cells (reads rnn_escape_simulation.csv of the escape ladder)
python analyses/recurrent_networks/known_barrier_cells.py
# drift/diffusion scaling grid
python analyses/recurrent_networks/network_scaling_grid.py
python analyses/recurrent_networks/network_scaling_summary.py
# drive versus gain
python analyses/recurrent_networks/drive_gain_plane.py
python analyses/recurrent_networks/drive_gain_report.py
```

The defaults of every script are the settings of the stored run. The summary steps also run on the stored
outputs: `network_scaling_summary.py --rows <Source Data network_parameter_grid.csv>` and
`arrhenius_summary.py --ladder <Source Data rnn_escape_simulation.json>` reproduce the stored report, endpoint
tables, Supplementary Table 14 and `arrhenius_slopes.json` exactly.

## Inputs and locations

| Variable | Default | Content |
|---|---|---|
| `NDIF_RNN_WEIGHTS` | `analyses/recurrent_networks/weights` | trained weights |
| `NDIF_RNN_CACHE` | `analyses/_work/rnn` | escape-ladder cells and outputs, drive/gain rows |
| `NDIF_DEVICE` | `cuda` if available, else `cpu` | torch device |
| `NDIF_OUT` | (unset) | if set, outputs are written below this directory instead of their stored locations |

No source data are needed: every recording analysed here is simulated from the trained networks.

## Trained weights

`weights/network_<seed>.pt` are the state dictionaries (W 64 x 64, W_in 64 x 4, b 64, W_out 1 x 64, float32) of
the nine candidate networks, seeds 20260915 + 7 + 13k for k = 0, ..., 8; `network_<seed>.json` records the
training score and recipe, and `qualification_<seed>.json` the enumerated structure at u = 0 (number of fixed
points, number of stable fixed points, completeness of the enumeration). Each network was trained by
`network_lib.train_original_recipe(network_lib.flipflop_batch, "flipflop-<seed>", 1200, seed)`: 1,200 Adam steps
(learning rate 5e-3, batches of 128 trials of 40 steps at alpha = 0.2, gradient norm clipped at 1) on a flip-flop
task in which a +-1 pulse on input 0 during the first five steps must be held. A network is used when the
enumeration at u = 0 finds exactly two stable fixed points with completeness >= 0.98: eight of the nine
candidates qualify; seed 20261013 (a single fixed point, not stable) is not used. `escape_ladder.py` trains and qualifies
any candidate whose weight file is missing; the stored analyses used the weights shipped here.

## Runtimes and reproducibility

On one desktop with a CUDA GPU (stored runs: Python 3.13.9, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0,
torch 2.9.0+cu130, NVIDIA GeForce RTX 3050): `network_scaling_grid.py` about 4 min; `escape_ladder.py` 40 cells of
28-131 s each (about 16 min wall with parallel workers, about 1 h serial) and seconds for `--resume`;
`known_barrier_cells.py` about 13.5 min; `drive_gain_plane.py` about 12.5 min; the summary and report steps
seconds to about one minute. The simulations draw their noise from seeded torch generators on the selected
device (CUDA for the stored runs).
