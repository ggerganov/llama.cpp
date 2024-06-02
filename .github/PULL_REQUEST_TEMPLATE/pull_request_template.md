# Pull Request Template

## Summary

* [Briefly describe the changes made in this PR.]
* [Include any relevant context, such as the issue or feature being addressed.]

## Changes Made

* [List specific files or directories affected by these changes]
* [Describe any significant updates, rewrites, or new code added]
* [Mention any removed or deleted files or code]

## Relevant Details

* **Affected Code**: [List specific code paths, functions, or classes changed or updated]
* **Impact**: [Explain how the changes affect the project's functionality, performance, or security]
* **New Features/Changes**: [Describe new features or significant changes added in this PR]
* **Fixed Issues**: [List specific issues or bugs fixed by these changes]

## Verification

To verify this PR, you can:

* Run automated tests or scripts to ensure the changes do not introduce errors
* Review code for style, security, and best practices
* Verify that all changed files are properly formatted and consistent

## Additional Information (Optional)

[Add any additional context, explanations, or requests that are relevant to this PR.]

**Important Notes**

* If this pull request only contains documentation changes (e.g., updating
READMEs, adding new wiki pages), please add `[no ci]` to the commit title.
This will skip unnecessary CI checks and help reduce build times.
* When squashing multiple commits on merge, use the following format for
your commit title: `<module>:<commit title> (#<issue_number>)`. For example: `utils: Fix typo in utils.py (#1234)`
* Please ensure that this PR follows our contributing guidelines, available at [](README.md). This includes formatting code according to our style guide and ensuring that all changes are thoroughly tested.
