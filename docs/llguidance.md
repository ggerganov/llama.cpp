# LLGuidance support in llama.cpp

[LLGuidance](https://github.com/guidance-ai/llguidance) is a library for constrained decoding (also called constrained sampling or structured outputs) for Large Langauge Models (LLMs).
It was developed as the backend for [Guidance](https://github.com/guidance-ai/guidance) library, but can be also used standalone.

LLGuidance supports JSON Schemas or arbitrary context-free grammars (CFGs) in
a [variant](https://github.com/guidance-ai/llguidance/blob/main/parser/src/lark/README.md) of Lark syntax.
It is [very fast](https://github.com/guidance-ai/jsonschemabench/tree/main/maskbench)
and has [excellent](https://github.com/guidance-ai/llguidance/blob/main/parser/src/json/README.md) JSON Schema coverage.
It does, however, complicate llama.cpp build process, as it requires Rust compiler.

## Building

To enable LLGuidance support, build llama.cpp with the `LLAMA_LLGUIDANCE` option:

```sh
cmake -B build -DLLAMA_LLGUIDANCE=ON
make -C build -j
```

This requires the Rust compiler and `cargo` tool to be [installed](https://www.rust-lang.org/tools/install).

## Interface

There are no new command line arguments or `common_params`.
When enabled, any grammar starting with `%llguidance` is passed to LLGuidance, not the [current](../grammars/README.md) llama.cpp Grammars.
Additionally, when JSON Schema is requested (eg., with `-j` argument to `llama-cli`), it's also passed to LLGuidance.

## Performance

Computing "token mask" (ie., set of all allowed tokens), for a llama3 tokenizer (with 128k tokens),
for [JSON Schema Bench](https://github.com/guidance-ai/jsonschemabench) takes on avarage 50μs of single-core CPU time. The p99 time is 0.5ms, and p100 is 20ms.

This is due to lexer/parser split and a bunch of [optimizations](https://github.com/guidance-ai/llguidance/blob/main/docs/optimizations.md).

## JSON Schema

LLGuidance tries to be faithful to the JSON Schema specification where possible.
In particular, unlike in current Grammars, `additionalProperties` defaults to `true`, and any whitespace is allowed.
You can of course set `"additionalProperties": false` yourself.
LLGuidance will also follow definition order of properties in the `"properties": {}` object,
regardless if they are required or not (current Grammars always put required properties first).

If a schema is not fully supported by LLGuidance, it will error out with a message.
That is, no JSON Schema keywords are silently ignored.

## Why not re-use GBNF format?

GBNF has no concept of a lexer.

For virtually all programming languages (including JSON), lexers, typically built using regular expressions, are used to convert a stream of bytes into a stream of lexemes (also called tokens, but that name conflicts with LLM tokens).
Then, the context-free grammar (CFG) parser can operate on lexemes, and there is way fewer of them than bytes.
Because regular expressions are cheaper to evaluate than context-free grammars, this two-step process is faster than parsing the whole input with a CFG.

Typically the LLM tokens are somewhat aligned with lexemes, meaning that when executing the grammar against all tokens, the parser needs to be involved in 0.5% or less of cases, leaving the rest to the lexer.

However, the user has to specify the distinction between lexemes and CFG symbols.
In [Lark](https://github.com/lark-parser/lark) this is done by making the lexemes names all uppercase,
while CFG symbols are all lowercase.

For example, this is a very simplified grammar for the C programming language:

```lark
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

The GBNF grammar would be very similar, but `ID` and `NUMBER` would typically be
lowercase, and would be internally translated to a CFG, instead of being kept as regular expressions.
Also, in the last line we define that all whitespace should be ignored.
This would have to specified explicitly everywhere in the GBNF format.

While it is possible to write a grammar with only lowercase symbols, it will be much slower than a grammar with lexemes.
You will also eventually get an error about 'single-byte lexemes' from LLGuidance.
Typically, renaming some symbols to uppercase will fix this.

## Error handling

Currently, errors are just printed to stderr, and the generation continues.
This can hopefully be improved in the future.
