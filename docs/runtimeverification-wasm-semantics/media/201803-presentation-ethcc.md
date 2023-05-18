---
title: 'KWASM: Overview and Path to KeWASM'
author: Everett Hildenbrandt
date: '\today'
theme: metropolis
header-includes:
-   \newcommand{\K}{\ensuremath{\mathbb{K}}}
-   \newcommand{\instr}{instr}
-   \newcommand{\LOOP}{\texttt{loop}}
-   \newcommand{\LABEL}{\texttt{label}}
-   \newcommand{\END}{\texttt{end}}
-   \newcommand{\stepto}{\hookrightarrow}
---

Overview
--------

1.  Introduction to \K{}
2.  KWASM Design
3.  Using KWASM (Psuedo-Demo)
4.  Future Directions

Introduction to \K{}
====================

The Vision: Language Independence
---------------------------------

Separate development of PL software into two tasks:

. . .

### The Programming Language

PL expert builds rigorous and formal spec of the language in a high-level human-readable semantic framework.

. . .

### The Tooling

Build each tool *once*, and apply it to every language, eg:

-   Parser
-   Interpreter
-   Debugger
-   Compiler
-   Model Checker
-   Program Verifier

The Vision: Language Independence
---------------------------------

![K Tooling Overview](media/img/k-overview.png)

Current Semantics
-----------------

Many languages have full or partial \K{} semantics, this lists some notable ones (and their primary usage).

-   [C](https://github.com/kframework/c-semantics): detecting undefined behavior
-   [Java](https://github.com/kframework/java-semantics): detecting racy code
-   [EVM](https://github.com/kframework/evm-semantics): verifying smart contracts
-   [LLVM](https://github.com/kframework/llvm-semantics): compiler validation (to x86)
-   [JavaScript](https://github.com/kframework/javascript-semantics): finding disagreements between JS engines
-   [P4](https://github.com/kframework/p4-semantics): SDN data-layer verification
-   many others ...

\K{} Specifications: Syntax
---------------------------

Concrete syntax built using EBNF style:

```k
    syntax Exp ::= Int | Id | "(" Exp ")" [bracket]
                 | Exp "*" Exp
                 > Exp "+" Exp // looser binding

    syntax Stmt ::= Id ":=" Exp
                  | Stmt ";" Stmt
                  | "return" Exp
```

. . .

This would allow correctly parsing programs like:

```imp
    a := 3 * 2;
    b := 2 * a + 5;
    return b
```

\K{} Specifications: Configuration
----------------------------------

Tell \K{} about the structure of your execution state.
For example, a simple imperative language might have:

```k
    configuration <k>     $PGM:Program </k>
                  <env>   .Map         </env>
                  <store> .Map         </store>
```

. . .

> -   `<k>` will contain the initial parsed program
> -   `<env>` contains bindings of variable names to store locations
> -   `<store>` conaints bindings of store locations to integers

\K{} Specifications: Transition Rules
-------------------------------------

Using the above grammar and configuration:

. . .

### Variable lookup

```k
    rule <k> X:Id => V ... </k>
         <env>   ...  X |-> SX ... </env>
         <store> ... SX |-> V  ... </store>
```

. . .

### Variable assignment

```k
    rule <k> X := I:Int => . ... </k>
         <env>   ...  X |-> SX       ... </env>
         <store> ... SX |-> (V => I) ... </store>
```

Example Execution
-----------------

### Program

```imp
    a := 3 * 2;
    b := 2 * a + 5;
    return b
```

### Initial Configuration

```k
    <k>     a := 3 * 2 ; b := 2 * a + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 0    1 |-> 0 </store>
```

Example Execution (cont.)
-------------------------

### Variable assignment

```k
    rule <k> X := I:Int => . ... </k>
         <env>   ...  X |-> SX       ... </env>
         <store> ... SX |-> (V => I) ... </store>
```

### Next Configuration

```k
    <k>     a := 6 ~> b := 2 * a + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 0    1 |-> 0 </store>
```

Example Execution (cont.)
-------------------------

### Variable assignment

```k
    rule <k> X := I:Int => . ... </k>
         <env>   ...  X |-> SX       ... </env>
         <store> ... SX |-> (V => I) ... </store>
```

### Next Configuration

```k
    <k>               b := 2 * a + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

Example Execution (cont.)
-------------------------

### Variable lookup

```k
    rule <k> X:Id => V ... </k>
         <env>   ...  X |-> SX ... </env>
         <store> ... SX |-> V  ... </store>
```

### Next Configuration

```k
    <k>     a ~> b := 2 * [] + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

Example Execution (cont.)
-------------------------

### Variable lookup

```k
    rule <k> X:Id => V ... </k>
         <env>   ...  X |-> SX ... </env>
         <store> ... SX |-> V  ... </store>
```

### Next Configuration

```k
    <k>     6 ~> b := 2 * [] + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

Example Execution (cont.)
-------------------------

### Variable lookup

```k
    rule <k> X:Id => V ... </k>
         <env>   ...  X |-> SX ... </env>
         <store> ... SX |-> V  ... </store>
```

### Next Configuration

```k
    <k>          b := 2 * 6 + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

Example Execution (cont.)
-------------------------

### Variable assignment

```k
    rule <k> X := I:Int => . ... </k>
         <env>   ...  X |-> SX       ... </env>
         <store> ... SX |-> (V => I) ... </store>
```

### Next Configuration

```k
    <k>     b := 17 ~> return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

Example Execution (cont.)
-------------------------

### Variable assignment

```k
    rule <k> X := I:Int => . ... </k>
         <env>   ...  X |-> SX       ... </env>
         <store> ... SX |-> (V => I) ... </store>
```

### Next Configuration

```k
    <k>                return b </k>
    <env>   a |-> 0    b |-> 1  </env>
    <store> 0 |-> 6    1 |-> 17 </store>
```

KWASM Design
============

WASM Specification
------------------

Available at <https://github.com/WebAssembly/spec>.

-   Fairly unambiguous[^betterThanEVM].
-   Well written with procedural description of execution accompanied by small-step semantic rules.

. . .

Example rule:

1. Let $L$ be the label whose arity is 0 and whose continuation is the start of the loop.
2. `Enter` the block $\instr^\ast$ with label $L$.

. . .

\vspace{-2em}
$$
    \LOOP~[t^?]~\instr^\ast~\END
    \quad \stepto \quad
    \LABEL_0\{\LOOP~[t^?]~\instr^\ast~\END\}~\instr^\ast~\END
$$

[^betterThanEVM]: At least, better than the [YellowPaper](https://github.com/ethereum/yellowpaper).

Translation to \K{}
-------------------

### WASM Spec

\vspace{-1em}
$$
    \LOOP~[t^?]~\instr^\ast~\END
    \quad \stepto \quad
    \LABEL_0\{\LOOP~[t^?]~\instr^\ast~\END\}~\instr^\ast~\END
$$

. . .

### In \K{}

```k
    syntax Instr ::= "loop" Type Instrs "end"
 // -----------------------------------------
    rule <k> loop TYPE IS end
          => IS
          ~> label [ .ValTypes ] {
                loop TYPE IS end
             } STACK
          ...
         </k>
         <stack> STACK </stack>
```

Design Difference: 1 or 2 Stacks?
---------------------------------

. . .

### WASM Specification

One stack mixing values and instructions.

-   Confusing control-flow semantics (with `label`s).
-   Use meta-level context operator to describe semantics of `br`.
-   Correct label is buried in the value stack (which we must also access the top of).
-   Section 4.4.5 of the WASM spec.

. . .

### KWASM

Uses two stacks, values in `<stack>` cell and instructions in `<k>` cell.

-   Can access both cells simultaneously, without backtracking/remembering one stack.
-   Cleaner semantics, no meta-level context operator needed.

Design Choice: Incremental Semantics
------------------------------------

-   KWASM semantics are given incrementally.
-   Makes it possible to execute program fragments.
-   Allows users to quickly experiment with WASM using KWASM.

. . .

For example, KWASM will happily execute the following fragment (without an enclosing `module`):

```wast
    (i32.const 4)
    (i32.const 5)
    (i32.add)
```

Early Stage Benefits of KWASM
-----------------------------

> -   Discovered bugs in spec (one filed on Github, some to come).
> -   Suggests clearer presentation of control-flow semantics.
> -   Interpreter allowing direct experimentation with program fragments.

Using KWASM (Psuedo-Demo)
=========================

Getting/Building
----------------

Clone the repository:

```sh
git clone 'https://github.com/kframework/wasm-semantics'
cd wasm-semantics
```

Build the dependencies, then the KWASM semantics:

```sh
make deps
make build
```

`kwasm` Script
--------------

The file `./kwasm` is the main runner for KWASM.

### Running `./kwasm help`

```sh
usage: ./kwasm <cmd> <file> <K args>*

    # Running
    # -------
    ./kwasm run   <pgm>   Run a single WASM program
    ./kwasm debug <pgm>   Run a single WASM program in the debugger
    ...
```

Running a Program
-----------------

### WASM Program `pgm1.wast`

```wasm
(i32.const 4)
(i32.const 5)
(i32.add)
```

### Result of `./kwasm run pgm1.wast`

```k
<generatedTop>
  <k>
    .
  </k>
  <stack>
    < i32 > 9 : .Stack
  </stack>
</generatedTop>
```

Debugging a Program
-------------------

### Run `./kwasm debug pgm1.wast`

```k
== debugging: pgm1.wast
KDebug> s
1 Step(s) Taken.
KDebug> p
<generatedTop>
  <k>
    i32 . const 4 ~> i32 . const 5  i32 . add  .Instrs
  </k>
  <stack>
    .Stack
  </stack>
</generatedTop>
```

Debugging a Program (cont.)
---------------------------

### Take a `s`tep then `p`eek at state

```k
KDebug> s
1 Step(s) Taken.
KDebug> p
<generatedTop>
  <k>
    i32 . const 5  i32 . add  .Instrs
  </k>
  <stack>
    < i32 > 4 : .Stack
  </stack>
</generatedTop>
```

Debugging a Program (cont.)
---------------------------

### Take a `s`tep then `p`eek at state

```k
KDebug> s
1 Step(s) Taken.
KDebug> p
<generatedTop>
  <k>
    i32 . const 5 ~> i32 . add  .Instrs
  </k>
  <stack>
    < i32 > 4 : .Stack
  </stack>
</generatedTop>
```

Debugging a Program (cont.)
---------------------------

### Take 10 `s`teps then `p`eek at state

```k
KDebug> s 10
Attempted 10 step(s). Took 4 steps(s).
Final State Reached
KDebug> p
<generatedTop>
  <k>
    .
  </k>
  <stack>
    < i32 > 9 : .Stack
  </stack>
</generatedTop>
```

Future Directions
=================

Finish KWASM
------------

The semantics are fairly early-stage.

### In progress (1 week out)

-   Frame/locals semantics, `call*` and `return` opcodes.

### To be done (1-2 months out)

-   Some bitwise operators.
-   Everything floating point.
-   Tables.
-   Memories.
-   Modules.

Fork KeWASM
-----------

-   eWASM adds the gas metering contract to WASM, but otherwise largely leaves the semantics alone.
-   Could we give a direct semantics to gas metering?
-   Possibly, perhaps then we could verify that the gas metering contract and the direct gas metering agree.

Verify eWASM Programs
---------------------

-   KEVM currently has many verified smart contracts at <https://github.com/runtimeverification/verified-smart-contracts>.
-   We similarly would like to build a repository of verified code using KeWASM.

Conclusion
==========

Benefits of \K{} Approach
-------------------------

-   Many tools that a PL designer/user would want already exist.
-   Powerful (and generic) verification engine "for free".
-   Executable specification clears up issues in the existing WASM specification[^similarKEVM].

[^similarKEVM]: Similarly for KEVM, we found several issues/ambiguities in the [YellowPaper](https://github.com/ethereum/yellowpaper).

Questions?
----------

Thanks for listening!

References
----------

\tiny

