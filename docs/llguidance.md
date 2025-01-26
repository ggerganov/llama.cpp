# LLGuidance Support in llama.cpp

[LLGuidance](https://github.com/guidance-ai/llguidance) is a library for constrained decoding (also called constrained sampling or structured outputs) for Large Language Models (LLMs). Initially developed as the backend for the [Guidance](https://github.com/guidance-ai/guidance) library, it can also be used independently.

LLGuidance supports JSON Schemas and arbitrary context-free grammars (CFGs) written in a [variant](https://github.com/guidance-ai/llguidance/blob/main/parser/src/lark/README.md) of Lark syntax. It is [very fast](https://github.com/guidance-ai/jsonschemabench/tree/main/maskbench) and has [excellent](https://github.com/guidance-ai/llguidance/blob/main/parser/src/json/README.md) JSON Schema coverage but requires the Rust compiler, which complicates the llama.cpp build process.

## Building

To enable LLGuidance support, build llama.cpp with the `LLAMA_LLGUIDANCE` option:

```sh
cmake -B build -DLLAMA_LLGUIDANCE=ON
make -C build -j
```

This requires the Rust compiler and the `cargo` tool to be [installed](https://www.rust-lang.org/tools/install).

## Interface

There are no new command-line arguments or modifications to `common_params`. When enabled, grammars starting with `%llguidance` are passed to LLGuidance instead of the [current](../grammars/README.md) llama.cpp grammars. Additionally, JSON Schema requests (e.g., using the `-j` argument in `llama-cli`) are also passed to LLGuidance.

## Performance

Computing a "token mask" (i.e., the set of allowed tokens) for a llama3 tokenizer with 128k tokens takes, on average, 50μs of single-core CPU time for the [JSON Schema Bench](https://github.com/guidance-ai/jsonschemabench). The p99 time is 0.5ms, and the p100 time is 20ms. These results are due to the lexer/parser split and several [optimizations](https://github.com/guidance-ai/llguidance/blob/main/docs/optimizations.md).

## JSON Schema

LLGuidance adheres closely to the JSON Schema specification. For example:

- `additionalProperties` defaults to `true`, unlike current grammars, though you can set `"additionalProperties": false` if needed.
- any whitespace is allowed.
- The definition order in the `"properties": {}` object is maintained, regardless of whether properties are required (current grammars always puts required properties first).

Unsupported schemas result in an error message—no keywords are silently ignored.

## Why Not Reuse GBNF Format?

GBNF lacks the concept of a lexer.

Most programming languages, including JSON, use a two-step process: a lexer (built with regular expressions) converts a byte stream into lexemes, which are then processed by a CFG parser. This approach is faster because lexers are cheaper to evaluate, and there is ~10x fewer lexemes than bytes.

LLM tokens often align with lexemes, so the parser is engaged in under 0.5% of tokens, with the lexer handling the rest.

However, the user has to provide the distinction between lexemes and CFG symbols. In [Lark](https://github.com/lark-parser/lark), lexeme names are uppercase, while CFG symbols are lowercase.

For example, a simplified C grammar in Lark:

```lark
%llguidance {}

start: program

program: (function_definition | declaration)*

function_definition: type ID "(" parameter_list? ")" "{" statement* "}"
parameter_list: parameter ("," parameter)*
parameter: type ID

declaration: type variable_list ";"
variable_list: ID ("," ID)*

type: "int" | "float" | "char" | "void"

statement: declaration
         | assignment ";"
         | "return" expr ";"
         | if_statement
         | while_statement
         | expr ";"

assignment: ID "=" expr
expr: term (("+" | "-") term)*
term: factor (("*" | "/") factor)*
factor: ID | NUMBER | "(" expr ")"

if_statement: "if" "(" expr ")" "{" statement* "}" ("else" "{" statement* "}")?
while_statement: "while" "(" expr ")" "{" statement* "}"

ID: /[a-zA-Z_][a-zA-Z0-9_]*/
NUMBER: /[0-9]+/

%ignore /[ \t\f\r\n]+/
```

In GBNF, lexemes like `ID` and `NUMBER` are typically lowercase and converted to CFG rules instead of remaining regular expressions. Ignoring whitespace would need to be explicitly specified everywhere.

Writing grammars without lexemes would be slower and might result in "single-byte lexeme" errors in LLGuidance, fixable by renaming symbols to uppercase.

## Error Handling

Errors are currently printed to `stderr`, and generation continues. Improved error handling may be added in the future.
