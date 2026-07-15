# Workflow Files

Two official workflow entry points are provided. Both run the **same** YBI_Ranking_Framework pipeline; they are not separate pipelines.

1. **Python entry point (cross-platform; recommended on Windows)**:
   - Script: `run_full_pipeline.py`
   - Usage: `python scripts/workflow/run_full_pipeline.py`
   - Supported platforms: Windows / macOS / Linux

2. **Shell entry point (macOS / Linux wrapper)**:
   - Script: `run_full_pipeline.sh`
   - Usage: `bash scripts/workflow/run_full_pipeline.sh`
   - Supported platforms: macOS / Linux
   - Implementation: forwards all arguments unchanged to the Python entry point so that artifact paths and resume checks have a single implementation

Workflow configuration file:

- `full_pipeline.yaml`

Responsibilities:

- Runs the three core YBI_Ranking_Framework stages in sequence: `make-folds` -> `sweep-lambda` -> `select-lambda --train`
- Supports `--skip-existing` for safe resume behavior and automatic reuse of completed work
- Supports command-line overrides such as `--batch-size` and `--rank-loss-mode`
- When `--rank-loss-mode disabled` is used, the workflow trains the classification baseline with a fixed `lambda_rank=0`, skips adaptive S2/S1 lambda selection, and writes the selected five-fold ensemble to the standard `disabled/selection` artifact directory

Sweep cache semantics:

- `--no-skip-existing` passes `--no-reuse-existing` to `sweep-lambda`, causing every requested route/lambda/fold run to be retrained
- `--skip-existing` still executes `sweep-lambda`, but reuses only runs whose resolved configuration, input data, fold manifest, training-code fingerprint, and required artifacts are complete and compatible; all other runs are retrained automatically
- Legacy sweep runs without `run_completion.json` cannot be verified and are therefore retrained once for safety

Notes:

- Both entry points use the same Python workflow implementation, so they produce the same artifacts and follow the same default directory rules and stage order
- Choose an entry point based only on the operating system and personal preference; the entry points do not represent different experimental workflows
- Before running either entry point, activate a suitable Python environment and ensure that the `cvrank` dependencies are installed
