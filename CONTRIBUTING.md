# Contributing

This project is an experiment log and protocol for legacy Fermi inference on a
current Arch Linux system. Changes should make the experiment easier to repeat,
safer to run, or easier to interpret.

## Ground Rules

- Keep the Intel/Wayland desktop path recoverable.
- Keep NVIDIA testing isolated to the dedicated boot entry.
- Prefer commands that inspect state over commands that mutate state.
- Document system changes before making them.
- Keep generated artifacts out of git.
- Record failures as first-class results.

## Documentation Style

- Use concrete commands and observed output.
- Include dates for hardware/software observations.
- Separate proven facts from hypotheses and likely outcomes.
- Call out host-specific assumptions, especially package versions and kernel versions.

## Results

Place run summaries under `results/runs/YYYY-MM-DD-HHMM/` locally. The run
directories are ignored by git by default because they can contain large logs.
Summarize important findings in a tracked Markdown note when a run produces a
decision or reusable lesson.
