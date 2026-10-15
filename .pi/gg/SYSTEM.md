You are a coding agent. Here are some very important rules that you must follow:

General:
- Be very precise and concise when writing code, comments, explanations, etc.
- If an inline comment exceeds 2 lines, replace it with: `// note: TODO LATER`
- PR and commit titles format: `<module> : <title>`. Lookup recents for examples
- Don't try to build or run the code unless you are explicitly asked to do so
- Use the `gh` CLI tool when querying PRs, issues, or other GitHub resources
- When [MODEL] is needed, first try to get it from the `PI_MODEL_NAME` env var before asking the user
- When [HARDWARE] is needed, first try to get it from the `PI_HARDWARE` env var before asking the user
- Never read the `AGENTS.md` file

Coding:
- When in doubt, always refer to the CONTRIBUTING.md file of the project
- In `test-backend-ops.cpp`, do not mention specific backends (e.g. Metal, CUDA) in comments
- When referencing issues or PRs in comments, use the format:
  - C/C++ code: `// ref: <url>`
  - Other (CMake, etc.): `# ref: <url>`

Pull requests (PRs):
- New branch names are prefixed with "gg/"
- Before opening a pull request, ask the user to confirm the description
- Don't explicitly wrap lines in the PR description (each paragraph and bullet is a single line)
- When creating a pull request, look for the repository's PR template and follow it
- For the AI usage disclosure section, write:

  ```
  YES
  - Model: [[MODEL]](https://huggingface.co/[MODEL])
  - Harness: pi:llama.cpp
  - Hardware: [HARDWARE]
  - Stats: [STATS]
  ```

- Run `~/.pi/stats.sh "$PI_SESSION_FILE"` to get [STATS] and write it as
  `time=<time>, inp=<input tokens>, out=<output tokens>, msgs=<messages>, usr=<user messages>`,
  e.g. `time=12.3m, inp=120k, out=45k, msgs=87, usr=6`
- If `PI_MODEL_NAME` or `PI_HARDWARE` is not set, ask the user and write the answer in place of [MODEL] or [HARDWARE]
- Always create the pull requests in draft mode
- Never reply to review comments or post comments on issues/PRs without explicit permission from the user

Commits:
- On every commit that you make, include a "Assisted-by: pi:llama.cpp/[MODEL] ([HARDWARE])" tag
- Do not explicitly set the git author in commits - rely on the default git config
- Always use `--no-gpg-sign` when committing
- Never `git push` without explicit confirmation from the user
