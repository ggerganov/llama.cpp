# GBNF Guide

GBNF (GGML BNF) is a format for defining [formal grammars](https://en.wikipedia.org/wiki/Formal_grammar) to constrain model outputs in `llama.cpp`. For example, you can use it to force the model to generate valid JSON, or speak only in emojis. GBNF grammars are supported in various ways in `examples/main` and `examples/server`.

## Background

[Bakus-Naur Form (BNF)](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form) is a notation for describing the syntax of formal languages like programming languages, file formats, and protocols. GBNF is an extension of BNF that primarily adds a few modern regex-like features.

## Basics

In GBNF, we define *production rules* that specify how a *non-terminal* (rule name) can be replaced with sequences of *terminals* (characters, specifically Unicode [code points](https://en.wikipedia.org/wiki/Code_point)) and other non-terminals. The basic format of a production rule is `nonterminal ::= sequence...`.

## Examples

Here are some examples and what they mean:

- `root ::= "1. " move " " move "\n"...`: In this rule from a chess game grammar, `root` requires the characters `"1. "`, followed by a sequence of characters that match the `move` rule, followed by a space, and so on
- `move ::= (pawn | nonpawn | castle) [+#]?...`: Here, `move` is an abstract representation, which can be a pawn, nonpawn, or castle. The `[+#]?` denotes the possibility of checking or mate signs after moves.
- `ws ::= [ \t\n]*`: In this context, `ws` is an abstract representation of whitespace, which could be any combination of space, tab or newline characters, or nothing (`*` denotes one or more).

## Non-Terminals and Terminals

Non-terminal symbols (rule names) stand for a pattern of terminals and other non-terminals. They are required to be a dashed lowercase word, like `move`, `castle`, or `check-mate`.

Terminals are actual characters ([code points](https://en.wikipedia.org/wiki/Code_point)). They can be specified as a sequence like `"1"` or `"O-O"` or as ranges like `[1-9]` or `[NBKQR]`.

## Characters and character ranges

Terminals support the full range of Unicode. Unicode characters can be specified directly in the grammar, for example `hiragana ::= [ぁ-ゟ]`, or with escapes: 8-bit (`\xXX`), 16-bit (`\uXXXX`) or 32-bit (`\UXXXXXXXX`).

Character ranges can be negated with `^`:
```
single-line ::= [^\n]+ "\n"`
```

## Sequences and Alternatives

The order of symbols in a sequence matter. For example, in `"1. " move " " move "\n"`, the `"1. "` must come before the first `move`, etc.

Alternatives, denoted by `|`, give different sequences that are acceptable. For example, in `move ::= pawn | nonpawn | castle`, `move` can be a `pawn` move, a `nonpawn` move, or a `castle`.

Parentheses `()` can be used to group sequences, which allows for embedding alternatives in a larger rule or applying repetition and optptional symbols (below) to a sequence.

## Repetition and Optional Symbols

- `*` after a symbol or sequence means that it can be repeated zero or more times.
- `+` denotes that the symbol or sequence should appear one or more times.
- `?` makes the preceding symbol or sequence optional.

## Comments and newlines

Comments can be specified with `#`:
```
# defines optional whitspace
ws ::= [ \t\n]+
```

Newlines are allowed between rules and between symbols or sequences nested inside parentheses. Additionally, a newline after an alternate marker `|` will continue the current rule, even outside of parentheses.

## The root rule

In a full grammar, the `root` rule always defines the starting point of the grammar. In other words, it specifies what the entire output must match.

```
# a grammar for lists
root ::= ("- " item)+
item ::= [^\n]+ "\n"
```

## Next steps

This guide provides a brief overview. Check out the GBNF files in this directory (`grammars/`) for examples of full grammars. You can try them out with:
```
./main -m <model> --grammar-file grammars/some-grammar.gbnf -p 'Some prompt'
```
