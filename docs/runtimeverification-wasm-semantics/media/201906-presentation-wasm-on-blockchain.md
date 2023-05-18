---
title: 'KWasm'
subtitle: 'Semantics of WebAssembly in the K framework'
author:
-   Rikard Hjort
-   Everett Hildenbrandt
-   Qianyang Peng
date: June 8, 2019
institute:
-   Runtime Verification, Inc.
theme: metropolis
fontsize: 8pt
header-includes:
-   \usepackage{color}
-   \usepackage{fancyvrb}
-   \fvset{commandchars=\\\{\}}
-   \newcommand{\instr}{instr}
-   \newcommand{\STORE}{\textit{S}}
-   \newcommand{\FRAME}{\textit{F}}
-   \newcommand{\CONST}{\texttt{const~}}
-   \newcommand{\LOCALGET}{\texttt{local.get~}}
-   \newcommand{\LOCALSET}{\texttt{local.set~}}
-   \newcommand{\DATA}{\texttt{data}}
-   \newcommand{\FUNCS}{\texttt{funcs}}
-   \newcommand{\GLOBALS}{\texttt{globals}}
-   \newcommand{\GROW}{\texttt{grow}}
-   \newcommand{\ITHREETWO}{\texttt{i32}}
-   \newcommand{\LOCALS}{\texttt{locals}}
-   \newcommand{\MEMADDRS}{\texttt{memaddrs}}
-   \newcommand{\MEMORY}{\texttt{memory}}
-   \newcommand{\MEMS}{\texttt{mems}}
-   \newcommand{\MODULE}{\texttt{module}}
-   \newcommand{\SIZE}{\texttt{size}}
-   \newcommand{\TABLES}{\texttt{tables}}
-   \newcommand{\with}{\textit{~with~}}
-   \newcommand{\stepto}{~\hookrightarrow~}
-   \newcommand{\wif}[1]{\text{if}~#1}
-   \newcommand{\diminish}[1]{\begin{footnotesize}#1\end{footnotesize}}
---


Overview
--------

> 1. KWasm: Intro and roadmap
> 2. Introduction to K & KEVM
> 3. Deepdive: What the semantics look like
> 4. Demo: Proving things

. . .

\vspace{1em}

Please ask questions as we go.

KWasm: Intro and roadmap
========================

KWasm
-----

> * KWasm is the project name for specifying Wasm in K.
> * K is a framework for creating **runnable specifications** of programming languages.
> * K uses rewrite based semantics, just like those Wasm is defined with [@rossberg-web-up-to-speed].
> * The goal is to use the runnable spec to **formally verify** aspects of blockchain runtimes and smart contracts.
> * There is already a specification of the EVM, called KEVM [@hildenbrandt-saxena-zhu-rosu-k-evm], which we use for formal verification. \newline ![](media/img/kevm-paper.png)

Status
------

![](media/img/github-top-screenshot.png)

* Bulk of the semantics are done.
* Tables and indirect calls in progress.

A few big todos:

- Defining and instantiating several modules.
- Parsing more textual versions of commands (syntactic sugar).
- Add floating point numbers (not top priority).

Design
------

* A very faithful translation of Wasm spec (K and Wasm both use rewrite semantics). Some differences:
  - Two stacks: one for operands, one for instructions and control flow.
  - We are *more permissive*; allow running instructions directly:
\newline `(i32.add (i32.const 1337) (i32.const 42))` is a full KWasm program.
* Execution focused, we assume validation is done beforehand.

Goals
-----

- "Make KEVM for Ethereum 2.0".
- Create Ewasm semantics, KEwasm, by importing and embedding KWasm.
- We would like to build a repository of verified code using KEwasm.
There is such a repository for KEVM:

[![](media/img/github-verified-contracts-screenshot.png)](https://github.com/runtimeverification/verified-smart-contracts)

Introduction to K
=================

The Vision: Language Independence
---------------------------------

![K Tooling Overview](media/img/k-overview.png)


K Tooling/Languages
-------------------

### Tools

-   Parser
-   Interpreter
-   Debugger
-   Reachability Logic Prover [@stefanescu-park-yuwen-li-rosu-reachability-prover]
-   ...

. . .

### Languages

-   Java 1.4 - 2015 [@bogdanas-rosu-k-java]
-   C11 - 2015 [@hathhorn-ellison-rosu-k-c]
-   KJS - 2015 [@park-stefanescu-rosu-k-js]
-   KEVM - 2018 [@hildenbrandt-saxena-zhu-rosu-k-evm]
-   KLLVM <https://github.com/kframework/llvm-semantics>
-   KX86-64 <https://github.com/kframework/X86-64-semantics>
- In progress (external groups):
   - Solidity <https://github.com/kframework/solidity-semantics>
   - Rust

Parts of a K specification
--------------------------

A language spec in K consists of 3 things

* Syntax
* Configuration ("state")
* Operational semantics as **rewrite rules**

K Specifications: Syntax
------------------------

Concrete syntax built using EBNF style:

```k
    syntax IValType ::= "i32" | "i64"
    syntax Instr    ::= "(" IType "." "const" Int ")"
    syntax Instr    ::= "(" "local.get" Int ")" | "(" "local.set" Int ")"
    syntax Instr    ::= "(" IValType "." IBinOp ")"    // Concrete syntax
                      | IValType "." IBinOp Int Int    // Abstract syntax
    syntax IBinOp   ::= "div_u"
    syntax Instrs   ::= List{Instr, ""} // Builtin helper for cons lists.
```

Note: we tend to mix abstract and concrete syntax.

. . .

This would allow parsing a program like this:

```scheme
    (local.get 1)
    (local.get 0)
    (i32.div_u)
    (local.set 0)
```

K Specifications: Configuration
-------------------------------

Tell K about the structure of your execution state.

```k
    configuration <k>        $PGM:Instrs </k>
                  <valstack> .ValStack   </valstack>
                  <locals>   .Map        </locals>
```

. . .

> - `<k>` will contain the initial parsed program.
> - `<valstack>` operand stack of `Val` items.
> - `<locals>` a mapping `Int -> Val`

. . .

```k
    syntax Val ::= "<" IValType ">" Int
```

K Specifications: Transition Rules
----------------------------------

Using the above grammar and configuration:

. . .

### Push to ValStack

\begin{Verbatim}[]
    rule <k> ( ITYPE . const I ) \textcolor{blue}{=>} #chop(< ITYPE > I) \textcolor{blue}{...} </k>
\end{Verbatim}

. . .

> - `=>` is the rewrite arrow.
> - Words in all caps are variables.
> - We match on and rewrite the front of the cell contents, and `...` matches the rest of the cell.
> - We don't need to mention the cells we don't use or modify.

\vfill{}

. . .

\begin{Verbatim}[]
    rule <k> \textcolor{blue}{V:Val => \textbf{.}} ... </k>
         <valstack> \textcolor{blue}{VALSTACK => V : VALSTACK} </valstack>
\end{Verbatim}

. . .

> - `.` is like $\epsilon$, so rewriting to `.` is erasing.
> - We can rewrite several cells at once.
> - In `<valstack>`, we match on the entire cell.

K Specifications: Transition Rules
----------------------------------

### Helper functions:

\begin{Verbatim}[]
    syntax Val ::= #chop ( Val ) \textcolor{blue}{[function]}
 // ----------------------------------------
    rule #chop(< ITYPE > N) => < ITYPE > (N modInt #pow(ITYPE))

   syntax Int ::= #pow  ( IValType ) \textcolor{blue}{[function]}
 // -------------------------------------------
    rule #pow (i32) => 4294967296
\end{Verbatim}

. . .

\vspace{1em}

> - The `[function]` annotation means the rule applies regardless of context.

K Specifications: Transition Rules
----------------------------------

### Binary operators

\begin{Verbatim}[]
    rule <k> ( ITYPE . BOP:IBinOp ) => ITYPE . BOP C1 C2 ... </k>
         <valstack>
           < ITYPE > C2 : < ITYPE > C1 : VALSTACK => VALSTACK
         </valstack>

    rule <k> ITYPE . div_u I1 I2 => < ITYPE > (I1 /Int I2)  ... </k>
      \textcolor{blue}{requires I2 =/=Int 0}
    rule <k> ITYPE . div_u I1 I2 => undefined ... </k>
      \textcolor{blue}{requires I2  ==Int 0}
\end{Verbatim}

\vspace{1em}

- `requires` specifies side conditions.
- We often use K operators specialized by type, e.g. `==Int`

K Specifications: Transition Rules
----------------------------------

### Get local variable

$$
\FRAME ; (\LOCALGET x) \stepto \FRAME ; val
\qquad \textcolor{blue}{(\wif \FRAME.\LOCALS[x] = val)}
$$

. . .

\begin{Verbatim}[]
    rule <k> ( local.get INDEX ) => . ... </k>
         <valstack> VALSTACK => VALUE : VALSTACK </valstack>
         <locals> ... \textcolor{blue}{INDEX |-> VALUE} ... </locals>
\end{Verbatim}

. . .

- `<locals>` is a `Map` (builtin data structure), which is an associative-commutative pair of values. We can put `...` on both sides indictating we are matching *somewhere* in the `Map`.


K Specifications: Transition Rules
----------------------------------

### Set local variable

$$
\FRAME ; (\LOCALSET x) \stepto \FRAME' ; \epsilon
\qquad (\wif \FRAME' = \FRAME \textcolor{blue}{\with \LOCALS[x] = val})
$$

. . .

\begin{Verbatim}[]
    rule <k> ( local.set INDEX ) => . ... </k>
         <valstack> VALUE : VALSTACK => VALSTACK </valstack>
         <locals> ... INDEX |-> \textcolor{blue}{(_ => VALUE)} ... </locals>
\end{Verbatim}

. . .

- `_` is a wildcard (matches any value).
- We can use parentheses to isolate the part of a term we are rewriting, like updating the value part of a map entry.


## Example execution

### We can use KLab to explore execution of our example program.

```scheme
  (local.get 1)
  (local.get 0)
  (i32.div_u)
  (local.set 0)
```

with intial configuration

```k
  <locals>
    0 |-> <i32> 4
    1 |-> <i32> 24
  </locals>
```

. . .

\vfill

\center\huge DEMO!

Repo tour
=========

Repo layout
---------

* Literate programming style in markdown with K code blocks.
* `wasm.md`: The main part of the semantics
* `data.md`: Some helper data structures.
* `test.md`: Some useful assertion functions.
* `kwasm-lemmas.md`: A trusted base for proving.

Proving
=======

Verifying Wasm programs
----------------------------

> 1. From the KWasm semantics, K generates a parser and a deductive program verifier.
> 2. A verification claim is written like a rewrite rule. `rule A => B` should be read as "`A` will eventually always evaluate to `B`".
> 3. The automatic prover tries to construct a proof (with the help of Z3 to check constraint satisfiability) that every possible execution path starting in `A` eventually rewrites to `B`.
> 4. KLab offers an interactive view of execution, both the successful and the failed paths.

. . .

\vfill

\center\huge DEMO!

RV specializes in formal verification
-------------------------------------

- If you're interested in verification of Wasm programs, talk to us!
- [rikard.hjort@runtimeverification.com](rikard.hjort@runtimeverification.com)
- https://riot.im/app/#/room/#k:matrix.org

Conclusion/Questions?
=====================

Thanks!
-------

-   Thanks for listening!

References
----------

\tiny

