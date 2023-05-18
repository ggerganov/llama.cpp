---
title: 'Intro to K, KEVM, and KWasm'
author:
-   Everett Hildenbrandt
-   Rikard Hjort
date: '\today'
institute:
-   Runtime Verification, Inc.
-   Chalmers University of Technology
theme: metropolis
fontsize: 8pt
header-includes:
-   \newcommand{\instr}{instr}
-   \newcommand{\LOOP}{\texttt{loop}}
-   \newcommand{\LABEL}{\texttt{label}}
-   \newcommand{\END}{\texttt{end}}
-   \newcommand{\stepto}{\hookrightarrow}
---

Overview
--------

1.  Introduction to K
2.  KEVM
2.  KWasm
2.  Reachability Logic Prover
4.  Future Directions

(Brief) Introduction to K
=========================

K Vision
--------

![K Overview](media/img/k-overview.png)

K Tooling/Languages
-------------------

### Tools

-   Parser
-   Interpreter
-   Debugger
-   Reachability Logic Prover [@stefanescu-park-yuwen-li-rosu-reachability-prover]

. . .

### Languages

-   Java 1.4 - 2015 [@bogdanas-rosu-k-java]
-   C11 - 2015 [@hathhorn-ellison-rosu-k-c]
-   KJS - 2015 [@park-stefanescu-rosu-k-js]
-   KEVM - 2018 [@hildenbrandt-saxena-zhu-rosu-k-evm]
-   P4K - 2018 [@kheradmand-rosu-k-p4]
-   KIELE - 2018 [@kasampalis-guth-moore-rosu-johnson-k-iele]
-   KLLVM <https://github.com/kframework/llvm-semantics>
-   KX86-64 <https://github.com/kframework/X86-64-semantics>

K Specification: The Components
-------------------------------

-   **Syntax** of your language (term algebra of programs).
-   **Configuration** of your language (term algebra of program states).
-   **Rules** describing small-step operational semantics of your language.

K Specifications: Syntax
------------------------

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

K Specifications: Configuration
-------------------------------

Tell K about the structure of your execution state.
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

K Specifications: Transition Rules
----------------------------------

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

KEVM
====

What is KEVM?
-------------

-   Implementation of EVM in K.
-   Passes all of the VMTests and the BlockchainTests/GeneralStateTests.
-   Derived interpreter nearly as performant as cpp-ethereum.
-   The Jello Paper is derived from KEVM <https://jellopaper.org>.

. . .

-   Used for commercial verification by Runtime Verification, Inc.
-   Used by DappHub to verify the MKR SCD and MCD core contracts.
-   Check out repo of verified smart contracts at <https://github.com/runtimeverification/verified-smart-contracts>.

KWasm Design
============

Wasm Specification
------------------

Available at <https://github.com/WebAssembly/spec>.

-   Fairly unambiguous[^betterThanEVM].
-   Well written with procedural description of execution accompanied by small-step semantic rules.

\vfill{}

. . .

Example rule:

1. Let $L$ be the label whose arity is 0 and whose continuation is the start of the loop.
2. `Enter` the block $\instr^\ast$ with label $L$.

\vfill{}

. . .

$$
    \LOOP~[t^?]~\instr^\ast~\END
    \quad \stepto \quad
    \LABEL_0\{\LOOP~[t^?]~\instr^\ast~\END\}~\instr^\ast~\END
$$

[^betterThanEVM]: Better than the [YellowPaper](https://github.com/ethereum/yellowpaper).

Translation to K
----------------

### Wasm Spec

\vspace{-1em}
$$
    \LOOP~[t^?]~\instr^\ast~\END
    \quad \stepto \quad
    \LABEL_0\{\LOOP~[t^?]~\instr^\ast~\END\}~\instr^\ast~\END
$$

. . .

### In K

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

### Wasm Specification

One stack mixing values and instructions.

-   Confusing control-flow semantics (with `label`s).
-   Use meta-level context operator to describe semantics of `br`.
-   Section 4.4.5 of the Wasm spec.

\vfill{}

. . .

### KWasm

Uses two stacks, values in `<stack>` cell and instructions in `<k>` cell.

-   Can access both cells simultaneously, without backtracking/remembering one stack.
-   Cleaner semantics, no meta-level context operator needed.

Design Choice: Incremental Semantics
------------------------------------

-   KWasm semantics are given incrementally.
-   Makes it possible to execute program fragments.
-   Allows users to quickly experiment with Wasm using KWasm.

\vfill{}

. . .

For example, KWasm will happily execute the following fragment (without an enclosing `module`):

```wast
    (i32.const 4)
    (i32.const 5)
    (i32.add)
```

Future Steps for KWasm
----------------------

### To be done

-   Everything floating point.
-   Tables.
-   Modules.

### KeWasm

-   eWasm adds gas metering to Wasm, but otherwise leaves the semantics alone.

Reachability Logic Prover
=========================

Inference System
----------------

![Reachability Logic Inference System](media/img/reachability-logic-inference-system.png)

-   Sound and relatively complete.
-   Interesting rules are Circularity/Transitivity, allows coinductive reasoning.

Tool
----

-   K Reachability Logic prover accepts proof claims in same format as operational semantics axioms[@stefanescu-park-yuwen-li-rosu-reachability-prover].
-   On success will print `#True`, on failure will print symbolic counterexample end-states.
-   Added instrumentation allows KLab to provide more useful interface to K Prover <https://github.com/dapphub/klab>.
-   Proof search is fully automated, only write the theorem (specification), no manual control over how to discharge it.

Example KWasm Proof
-------------------

Non-overflowing addition operation:

```k
    rule <k> ( ITYPE:IValType . const X:Int )
             ( ITYPE          . const Y:Int )
             ( ITYPE . add )
          => .
          ...
         </k>
         <stack> S:Stack => < ITYPE > (X +Int Y) : S </stack>
      requires 0 <=Int X andBool 0 <=Int Y
       andBool (X +Int Y) <Int #pow(ITYPE)
```

-   Program which adds two symbolic numbers `X` and `Y`.
-   Don't care about bitwidth (`ITYPE` can be either `i32` or `i64`).
-   Add pre-condition that overflow doesn't happen: `(X +Int Y) <Int #pow(ITYPE)`.

Conclusion/Questions?
=====================

References
----------

-   Thanks for listening!

\tiny

