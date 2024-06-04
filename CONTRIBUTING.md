# Contributing Guidelines

## Checklist

* Make sure your PR follows the [coding guidelines](https://github.com/ggerganov/llama.cpp/blob/master/README.md#coding-guidelines)
* Test your changes using the commands in the [`tests`](tests) folder. For instance, running the `./tests/test-backend-ops` command tests different backend implementations of the GGML library
* Execute [the full CI locally on your machine](ci/README.md) before publishing

## PR formatting

* Please rate the complexity of your PR (i.e. `easy`, `medium`, `hard`). This makes it easier for maintainers to triage the PRs.
* If the pull request only contains documentation changes (e.g., updating
READMEs, adding new wiki pages), please add `[no ci]` to the commit title. This will skip unnecessary CI checks and help reduce build times.
* When squashing multiple commits on merge, use the following format for your commit title: `<module>:<commit title> (#<issue_number>)`. For example: `utils: Fix typo in utils.py (#1234)`
