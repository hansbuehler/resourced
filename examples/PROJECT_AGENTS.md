<!--
  Per-project agent instructions template, meant to live as this repository's
  own AGENTS.md / CLAUDE.md. It assumes the generic, machine-wide guardrails
  in examples/HOST_AGENTS.md are already merged into a global instructions
  file (e.g. ~/.codex/AGENTS.md); do not repeat them here, only what this
  project adds or narrows on top of that baseline. Replace the <PLACEHOLDER>
  values and trim sections that do not apply.
-->

# Agent baseline

Read and follow `STYLE_GUIDE.md` before making code changes in this repository.
More specific instructions in subdirectories supplement or override this file.
Host-wide guardrails (resource monitoring, progress logging, detached-job
hygiene, treating Git trees as source) are in the global instructions file;
this file only adds what is specific to this project.

## Project environment

- Set `<PROJECT>_ROOT` (and `PYTHONPATH`) explicitly in any detached job or
  service definition; do not rely on values set in an interactive shell.
- Project outputs, logs, and checkpoints live under
  `<PROJECT_ROOT>/runs/<job-name>/`; do not write them elsewhere (see the
  host guardrails' directory rule).
- Resource limits (RAM, and disk paths this project actually writes to) are
  configured once in `resourced`'s `config.toml`, not here. If this project
  writes to a path resourced does not yet monitor, add a `[[check]]` for it
  there instead of inventing a separate threshold in this file.

## GPU usage

<!-- Paste the output of `gpuctl agent-instructions` below, and regenerate it
     after upgrading resourced rather than hand-editing it. -->

## Known gotchas

<!-- Project-specific pitfalls an agent would otherwise rediscover the hard
     way, for example: data formats, flaky external services, slow tests to
     avoid running by default, or directories that look empty but are not. -->
