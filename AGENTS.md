# AGENTS.md

Use the devcontainer for development, builds, tests, smoke runs, and future toolchain work. The canonical runtime signal is `PROJECT_RUNTIME=devcontainer`.

Keep this file short. Use these documents for the actual project record:

- [README.md](C:\PRJ\hopfield-energy\README.md)
  Project entry point: setup, workflows, validation commands, runtime status, and release/deploy notes.
- [Architecture.md](C:\PRJ\hopfield-energy\Architecture.md)
  System design source of truth: APIs, module boundaries, backend strategy, and migration plan.
- [docs/models/*](C:\PRJ\hopfield-energy\docs\models)
  Per-model source of truth: model behavior, equations/energy definitions, training/retrieval details, and model-specific caveats.

Working rules:

- React pages are presentation only.
- Workers and CLI are runtime adapters only.
- Shared model/data logic belongs in reusable core modules.
- Prefer one clean implementation per model family.
- Always add or update terminal-verifiable checks when behavior changes.
- Always run validation from terminal, preferably inside the devcontainer.
- Keep README, Architecture, and docs/models synchronized after meaningful changes.
