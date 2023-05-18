---
title: 'Semantics of WebAssembly in the K framework'
subtitle: 'What is KWasm?'
author:
-   Rikard Hjort \tiny supervised by Magnus Myreen \normalsize
-   Everett Hildenbrandt
abstract:
  WebAssembly is a low-ish-level language designed to run efficiently on all modern platforms. The Ethereum blockchain currently runs on its own virtual machine (the EVM) but is expected to move to use WebAssembly in the future. The K framework definition of the EVM has become the de facto standard for verifying properties of smart contracts, most notably the Dai coin contracts, currently handling over 100 million USD of value. Since we want to verify Ethereum contracts compiled to WebAssembly -- as well as other WebAssembly programs -- we need to have a K formalization of WebAssembly. That is what I've been working on.

  I will be giving a hands-on talk on WebAssembly, how to define languages in K, and how we have been translating the official WebAssembly specification into K."
date: March 25, 2019
institute:
-   Chalmers University of Technology
-   Runtime Verification, Inc.
theme: metropolis
fontsize: 8pt
header-includes:
-   \newcommand{\instr}{instr}
-   \newcommand{\STORE}{\textit{S}}
-   \newcommand{\FRAME}{\textit{F}}
-   \newcommand{\CONST}{\texttt{const~}}
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

Video
-----

This talk was recoreded, and is available in full at <https://www.youtube.com/watch?v=V6tOYuneMqo>

Overview
--------

1.  Background
2.  Introduction to K
3.  Introduction to WebAssembly (Wasm)
4.  Demo: implement a Wasm subset
5.  (Proving things)

Background
==========

Smart contracts and formal methods
------------

![](media/img/ethereum.png){ width=65%}

- Blockchain technology, **smart contracts** in particular, caught my interest.
- Public, immutable code handling lots of money? Great area of application for formal methods!

Existing projects
-------

![](media/img/maker.png){ width=20% align=center style="margin-bottom:40px"}

- Contacted friends at MakerDAO.
- They have verified the core contracts of their "stablecoin", Dai.

. . .

![](media/img/dapphub.png){ width=40% align=center style="margin-bottom:40px"}

- The verification was largely done by a related organization, DappHub ...

. . .

![](media/img/k.png){ height=15% hspace=30px } &nbsp;&nbsp;&nbsp;
![](media/img/RV-logo-blue.eps){ height=15% hspace=30px } &nbsp;&nbsp;&nbsp;
<!-- ![](media/img/rv.png){ height=15%} -->

- ... using the K framework.

<!--

Verifying Ethereum contracts
---------

1. Contracts are compiled to Ethereum virtual machine (EVM) bytecode.
2. Some property or invariant is specified as a rewrite rule.
3. K tries to construct a proof (using the SMT solver Z3) that every possible execution path eventually rewrites to the correct thing
4. The tool KLab (by DappHub) offers an interactive view of execution paths, great for seeing where and why the prover failed.

-->

The KEVM
--------

![](media/img/kevm-paper.png)

- Verification made possible by KEVM [@hildenbrandt-saxena-zhu-rosu-k-evm], modelling the EVM.
- The EVM is a stack machine with $\approx$ 120 opcodes.
- Everett had begun work on a K specification of another low-level languge: Wasm.

ewasm
----

**Rationale**

How would ewasm be better than EVM?

. . .

- Speed
- Size
- Security
- Write contracts in C/C++, go, or rust
- Static analysis
- Optional metering
- Portability: ewasm contracts will be compatibile with any standard Wasm environment, including IoT and mobile devices


(Brief) Introduction to K
==========================

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
> -   `<store>` contains bindings of store locations to integers

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

Example Execution (cont.)
-------------------------

### Final configuration

```k
    <k>     return 17           </k>
    <env>   a |-> 0    b |-> 1  </env>
    <store> 0 |-> 6    1 |-> 17 </store>
```

(Brief) Introduction to WebAssembly
========

Wasm: your new favorite compile target
----

- It's not web, and it's not assembly, just a great low-ish level language.
  - Fast on hardware, but platform agnostic.
  - Allows stream compiling.
  - Efficient byte format, readable text format.

. . .

- Safety features
  - Blocks, loops and breaks, but not arbitrary jumps.
  - Allows static validation.
  - No implicit casts.

. . .

- Organized in modules *(example coming up)*.
  - Can declare functions, allocate and modify their own linear memory, global variables etc.
  - Cay export some of its contents.
  - Can have `start` functions, which are run when the module is loaded.

. . .

- Wasm is stack-based, but the syntax allows S-expression "folding" *(example coming up)*.


Code fold/unfold
----------------

```scheme
(memory.size)      ;; Nullary -- push memory size (i32).
(i64.extend_i32_u) ;; Unary   -- i32 ==> i64.
```

. . .

```scheme
(local.get $tmp)   ;; Nullary -- push local variable $tmp (i32).
(i64.load8_u)      ;; Unary   -- load 1 byte from argument address, push.
```

. . .

```scheme
(i64.add)          ;; Binary
```

. . .

\vfill{}

becomes

. . .

\vfill{}

```scheme
(i64.add
    (i64.extend_i32_u (memory.size))
    (i64.load8_u      (local.get $tmp)))
```

\vfill{}

. . .

Mix freely! Also OK:

```scheme
(i64.extend_i32_u (memory.size))
(i64.load8_u      (local.get $tmp)))
(i64.add)
```



Code example (folded)
---------------------

```scheme
(module
    (memory 1)
    (func   ;; Function descriptors.
            (export "myGrowAndStoreFunction")
            (param $by i32) (param $put i64) ;; Identifiers: $by and $put.
            (result i64)
            (local $tmp i32)

        ;; Body of the function.
        (local.set $tmp
            (i32.mul (memory.grow (local.get $by)) (i32.const 65536)))
        (i64.store (local.get $tmp) (local.get $put))
        (i64.add
            (i64.extend_i32_u (memory.size))
            (i64.load8_u      (local.get $tmp)))
    ) ;; End func.
) ;; End module.
```

Running the example
-------------------
```
$ ./wasm -i myProgram.wast -
wasm 1.0 reference interpreter
> (invoke "myGrowAndStoreFunction" (i32.const 2) (i64.const 0))
3 : i64
```

. . .

```
> (invoke "myGrowAndStoreFunction" (i32.const 2) (i64.const 0))
5 : i64
```

. . .

```
> (invoke "myGrowAndStoreFunction" (i32.const 1) (i64.const -1))
261 : i64
```

Wasm Specification
------------------

Available at <https://github.com/WebAssembly/spec>.

-   Fairly unambiguous[^betterThanEVM].
-   Well written with procedural description of execution accompanied by small-step semantic rules.

[^betterThanEVM]: Better than the [YellowPaper](https://github.com/ethereum/yellowpaper).

Wasm Specification
------------------

Semantic rule:

$$
\STORE; \FRAME; \MEMORY.\SIZE \stepto \STORE; \FRAME; (\ITHREETWO.\CONST sz)
$$
$$
(\wif{|\STORE.\MEMS[\FRAME.\MODULE.\MEMADDRS[0]].\DATA| = sz * 64 Ki)}
$$

. . .

*store* ($\STORE$) and a *current frame* ($\FRAME$):

\begin{alignat*}{5}
%&store      &::=~&\{ & \quad &\FUNCS    ~&\quad &funcinst^*         &     \\
&store      &::=~&\{ & \quad &\dots     ~&\quad &                   &     \\
%&           &    &   & \quad &\TABLES   ~&\quad &tableinst^*        &     \\
&           &    &   & \quad &\MEMS     ~&\quad &meminst^*          &     \\
&           &    &   & \quad &\dots     ~&\quad &                   &\}   \\
%&           &    &   & \quad &\GLOBALS  ~&\quad &globalinst^* \quad &\}   \\
&frame      &::=~&\{ & \quad &\LOCALS   ~&\quad &val^*              &     \\
&           &    &   & \quad &\MODULE   ~&\quad &moduleinst   \quad &\}   \\
&moduleinst~&::=~&\{~& \quad &\dots     ~&\quad &                   &     \\
&           &    &   & \quad &\MEMADDRS ~&\quad &memaddr^*          &     \\
&           &    &   & \quad &\dots     ~&\quad &                   &\}
\end{alignat*}

Wasm Specification (cont.)
--------------------------

**Example rule:** `(memory.size)`

. . .

Execution description:

1. Let $\FRAME$ be the current frame.
2. \diminish{Assert: due to validation, $\FRAME.\MODULE.\MEMADDRS[0]$ exists.}
3. Let $a$ be the memory address $\FRAME.\MODULE.\MEMADDRS[0]$.[^memIdxZero]
4. \diminish{Assert: due to validation, $\STORE.\MEMS[a]$ exists.}
5. Let $mem$ be the memory instance $\STORE.\MEMS[a]$.
6. Let $sz$ be the length of $mem.\DATA$ divided by the page size.
7. Push the value $\ITHREETWO.\CONST sz$ to the stack.

[^memIdxZero]: Every module in Wasm has a single memory for now, so we always implicitly work on `memaddrs[0]`.


Demo: implement a Wasm subset
=============================

`(memory.size)`
---------------

$$
\STORE ; \FRAME ; \MEMORY.\SIZE \stepto \STORE ; \FRAME ; (\ITHREETWO.\CONST sz)
$$
$$
\wif {
|\STORE.\MEMS[\FRAME.\MODULE.\MEMADDRS[0]].\DATA| = sz \cdot 64 Ki
}
$$

\vfill{}

\begin{alignat*}{5}
%&store      &::=~&\{ & \quad &\FUNCS    ~&\quad &funcinst^*         &     \\
&store      &::=~&\{ & \quad &\dots     ~&\quad &                   &     \\
%&           &    &   & \quad &\TABLES   ~&\quad &tableinst^*        &     \\
&           &    &   & \quad &\MEMS     ~&\quad &meminst^*          &     \\
&           &    &   & \quad &\dots     ~&\quad &                   &\}   \\
%&           &    &   & \quad &\GLOBALS  ~&\quad &globalinst^* \quad &\}   \\
&frame      &::=~&\{ & \quad &\LOCALS   ~&\quad &val^*              &     \\
&           &    &   & \quad &\MODULE   ~&\quad &moduleinst   \quad &\}   \\
&moduleinst~&::=~&\{~& \quad &\dots     ~&\quad &                   &     \\
&           &    &   & \quad &\MEMADDRS ~&\quad &memaddr^*          &     \\
&           &    &   & \quad &\dots     ~&\quad &                   &\}
\end{alignat*}

`(memory.grow)`
---------------

\begin{align*}
\STORE; \FRAME ; &(\ITHREETWO.\CONST n)~(\MEMORY.\GROW) \stepto \STORE ' ; \FRAME ; (\ITHREETWO.\CONST sz ) \\
(&\wif{\FRAME.\MODULE.\MEMADDRS[0] = a \\
&\land sz = |\STORE.\MEMS[a].\DATA|/64 Ki \\
&\land \STORE ' = \STORE \with \MEMS[a] = growmem(\STORE.\MEMS[a], n))} \\
\\
\STORE; \FRAME ; &(\ITHREETWO.\CONST n)~(\MEMORY.\GROW) \stepto \STORE ; \FRAME ; (\ITHREETWO.\CONST {-1} ) \\
\end{align*}


Future Directions
=================

Finish KWasm
------------

The semantics are fairly early-stage.

### In progress

-   Memories.

### To be done

-   Everything floating point.
-   Tables.
-   Modules.

KeWasm
------

-   eWasm adds gas metering to Wasm, but otherwise leaves the semantics alone.

\vfill{}

. . .

-   KEVM currently has many verified smart contracts at <https://github.com/runtimeverification/verified-smart-contracts>.
-   We similarly would like to build a repository of verified code using KeWasm.

Conclusion/Questions?
=====================

References
----------

-   Thanks for listening!

\tiny



<!--
SCRATCH
=======

* Explain semantics of imp, that Id's always refer to unique identifiers.
* Replace `loop` example with own example (smth you wrote)





KWasm Design
============

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

Using KWasm (Psuedo-Demo)
=========================

Getting/Building
----------------

Clone the repository:

```sh
git clone 'https://github.com/kframework/wasm-semantics'
cd wasm-semantics
```

Build the dependencies, then the KWasm semantics:

```sh
make deps
make build
```

`kwasm` Script
--------------

The file `./kwasm` is the main runner for KWasm.

### Running `./kwasm help`

```sh
usage: ./kwasm (run|test) [--backend (ocaml|java|haskell)] <pgm>  <K args>*
       ./kwasm prove      [--backend (java|haskell)]       <spec> <K args>*
       ./kwasm klab-(run|prove)                            <spec> <K args>*

    ./kwasm run   : Run a single WebAssembly program
    ./kwasm test  : Run a single WebAssembly program like it's a test
    ./kwasm prove : Run a WebAssembly K proof

    Note: <pgm> is a path to a file containing a WebAssembly program.
          <spec> is a K specification to be proved.
          <K args> are any arguments you want to pass to K when executing/proving.
```

Running a Program
-----------------

### Wasm Program `pgm1.wast`

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

Conclusion/Questions?
=====================


-->
