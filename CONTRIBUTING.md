# Contributing Guidelines

## Checklist

* Make sure your PR follows the [coding guidelines](https://github.com/ggerganov/llama.cpp/blob/master/README.md#coding-guidelines)
* Test your changes using the commands in the [`tests`](tests) folder. For instance, running the `./tests/test-backend-ops` command tests different backend implementations of the GGML library
* Execute [the full CI locally on your machine](ci/README.md) before publishing

## PR formatting

* Please rate the complexity of your PR (i.e. `Review Complexity : Low`, `Review Complexity : Medium`, `Review Complexity : High`). This makes it easier for maintainers to triage the PRs.
    - The PR template has a series of review complexity checkboxes `[ ]` that you can mark as `[X]` for your conveience. Refer to [About task lists](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/about-task-lists) for more information.
* If the pull request only contains documentation changes (e.g., updating READMEs, adding new wiki pages), please add `[no ci]` to the commit title. This will skip unnecessary CI checks and help reduce build times.
* When squashing multiple commits on merge, use the following format for your commit title: `<module> : <commit title> (#<issue_number>)`. For example: `utils : Fix typo in utils.py (#1234)`
