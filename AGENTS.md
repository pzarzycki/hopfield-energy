# AGENTS.md

Use the devcontainer for development, builds, tests, smoke runs, and future toolchain work. The canonical runtime signal is `PROJECT_RUNTIME=devcontainer`.

Keep this file short. Use these documents for the actual project record:

- [README.md](C:\PRJ\hopfield-energy\README.md)
  Entry point, current progress, workflows, validation commands, devcontainer usage.
- [Architecture.md](C:\PRJ\hopfield-energy\Architecture.md)
  Target design for data API, model/session API, observability API, backend strategy, and migration plan.

Working rules:

- React pages are presentation only.
- Workers and CLI are runtime adapters only.
- Shared model/data logic belongs in reusable core modules.
- Prefer one clean implementation per model family.
- Always add or update terminal-verifiable checks when behavior changes.
- Always run validation from terminal, preferably inside the devcontainer.
- Keep `README.md` and `Architecture.md` current when direction or workflow changes.
