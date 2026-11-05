# Experiments behind VERIFICATION.md and the README tables

Scripts only; every run writes its logs and results under `runs/` (gitignored).

| script | what | output |
|---|---|---|
| `repro.sh` / `repro.py` | published numbers in the papers' own settings (NCM, RanPAC, CODA-P, DualPrompt, L2P++) | `runs/repro/<name>_s<seed>/` |
| `fecam_own_transform.py` | FeCAM with the authors' exact transform | `runs/repro/fecam_debug/` |
| `preprocess_ablation.py` | effect of CIFAR preprocessing on NCM | `runs/repro/preproc/` |
| `reference.sh <gpu>` | the README reference table (exact CLI commands, 3 seeds) | `runs/reference/` |
| `merging.sh <gpu>` / `merging_magmax.py` | MagMax setting; every merging method on one sequential fine-tuning trajectory | `runs/merging/` |
| `clip_zeroshot_reference.py` | independent CLIP zero-shot check (no toolkit code) | stdout |
| `interop_grid.py` | every method x scenario x model kind on tiny CPU models (`--epochs 5 --learn-check` also checks learning) | `runs/interop/grid.{md,json}` |
| `summarize.py <folder>` | mean ± std over seeds | stdout |

Paths to CIFAR-100 (`/scratch/eric/data/cifar100`) are set at the top of each script.
