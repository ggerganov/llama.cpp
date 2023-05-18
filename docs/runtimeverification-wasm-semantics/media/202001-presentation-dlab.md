---
title: 'KWasm'
subtitle: 'Semantics of WebAssembly in the \K framework'
author:
-   Rikard Hjort (presenting)
-   Everett Hildenbrandt
-   Qianyang Peng
-   Stephen Skeirik
date: June 8, 2019
institute:
-   Runtime Verification, Inc.
theme: metropolis
fontsize: 8pt
header-includes:
-   \usepackage{color}
-   \usepackage{fancyvrb}
-   \usepackage{listings}
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
-   \newcommand{\K}{$\mathbb{K}$~}
-   \newcommand{\lK}{$\mathbb{K}$}
---


Overview
--------

> 1. KWasm: Intro and roadmap
> 2. \K recap
> 3. Proving things

KWasm: Intro and roadmap
========================

KWasm
-----

> * The goal is to use the runnable spec to **formally verify** aspects of blockchain runtimes and smart contracts.
> * The current targets are **Ewasm** (Ethereum smart contract VM) an **Pwasm** (Polkadot client bytecode).

Status
------

![](media/img/github-top-screenshot.png)

* 98 % complete for text format (missing a few floating point instructions).
* We have a full Wasm semantics, and prototype Ewasm and Pwasm **embeddings**.
* No bytecode support (yet), so bytecode is piped through `wasm2wat` tool.
* We have verified a few simple properties, but not yet any "real" programs (in progress with WRC20).

Design
------

* A very faithful translation of Wasm spec (\K and Wasm both use rewrite semantics). Some differences:
  - Two stacks: one for operands, one for instructions and control flow.
  - We are *more permissive*; allow running instructions directly:
\newline `(i32.add (i32.const 1337) (i32.const 42))` is a full KWasm program.
* Execution focused, we assume validation is done beforehand. No type checking or similar.
* We sometimes need to be more explicit to make rules computational.

Working with \K
=============

The Vision: Language Independence
---------------------------------

\center
\includegraphics[height=0.9\paperheight]{media/img/k-overview}


\K Tooling/Languages
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

Parts of a \K specification
--------------------------

A language spec in \K consists of 3 things

* Syntax
* Configuration ("state")
* Operational semantics as **rewrite rules**

<!---
\K Specifications: Syntax
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

. . .

This would allow parsing a program like this:

```scheme
    (local.get 1)
    (local.get 0)
    (i32.div_u)
    (local.set 0)
```

\K Specifications: Configuration
-------------------------------

Tell \K about the structure of your execution state.

```k
    configuration <k>        $PGM:Instrs </k>
                  <valstack> .ValStack   </valstack>
                  <locals>   .Map        </locals>
```

. . .

> - `<k>` will contain the initial parsed program.
> - `<valstack>` operand stack of `Val` items.
> - `<locals>` a mapping `Int -> Val`

<!---

. . .

```k
    syntax Val ::= "<" IValType ">" Int
```

-->

\K Specifications: Transition Rules
----------------------------------

We have configuration with a few cells.

Among them a `<k>` cell and a `<valstack>` cell.

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


\K Specifications: Transition Rules
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
> - This is also how we write lemmas. A lemma is basically a function rule.

Proving
=======

Verifying Wasm programs
----------------------------

> 1. From the KWasm semantics, \K generates a parser and a deductive program verifier.
> 2. A verification claim is written like a rewrite rule. `rule A => B` should be read as "`A` will eventually always evaluate to `B`".
> 3. The automatic prover tries to construct a proof (with the help of Z3 to check constraint satisfiability) that every possible execution path starting in `A` eventually rewrites to `B`.
> 4. KLab offers an interactive view of execution, both the successful and the failed paths.

The program (pseudo-code)
-----------

```
(func $i64.reverse_bytes(i64 input)
  i64 local[1] = 0
  i64 local[2] = 0
  while True:
    if local[1] >= 8:
      break
    bits = 56 - (local[1] * 8)
    local[2] = local[2] + (((input << bits) >> 56) << bits)
    local[1] = local[1] + 1
  return local[2]
)
```

Specification
-------------

```
rule <k> #wrc20ReverseBytes
      ~> (i64.load (i32.const ADDR))
          (i64.store (i32.const ADDR) (call $i64.reverse_bytes))
      => . ...
    </k>
// ...
      <mdata> BM => BM' </mdata>
  requires #inUnsignedRange(i32, ADDR)
// ...
  ensures  #get(BM, ADDR +Int 0) ==Int #get(BM', ADDR +Int 7 )
    andBool #get(BM, ADDR +Int 1) ==Int #get(BM', ADDR +Int 6 )
    andBool #get(BM, ADDR +Int 2) ==Int #get(BM', ADDR +Int 5 )
    andBool #get(BM, ADDR +Int 3) ==Int #get(BM', ADDR +Int 4 )
    andBool #get(BM, ADDR +Int 4) ==Int #get(BM', ADDR +Int 3 )
    andBool #get(BM, ADDR +Int 5) ==Int #get(BM', ADDR +Int 2 )
    andBool #get(BM, ADDR +Int 6) ==Int #get(BM', ADDR +Int 1 )
    andBool #get(BM, ADDR +Int 7) ==Int #get(BM', ADDR +Int 0 )
```

<!--- TODO continue from here -->


First proof attempt
-------------------

\tiny

```
( ( ( #getRange ( BM , ADDR , 8 ) <<Int 0  ) modInt (2 ^Int 64) >>Int 56 <<Int 0  ) modInt (2 ^Int 64) +Int
( ( ( #getRange ( BM , ADDR , 8 ) <<Int 8  ) modInt (2 ^Int 64) >>Int 56 <<Int 8  ) modInt (2 ^Int 64) +Int
( ( ( #getRange ( BM , ADDR , 8 ) <<Int 16 ) modInt (2 ^Int 64) >>Int 56 <<Int 16 ) modInt (2 ^Int 64) +Int
( ( ( #getRange ( BM , ADDR , 8 ) <<Int 24 ) modInt (2 ^Int 64) >>Int 56 <<Int 24 ) modInt (2 ^Int 64) +Int
( ( ( #getRange ( BM , ADDR , 8 ) <<Int 32 ) modInt (2 ^Int 64) >>Int 56 <<Int 32 ) modInt (2 ^Int 64) +Int
( ( ( #getRange ( BM , ADDR , 8 ) <<Int 40 ) modInt (2 ^Int 64) >>Int 56 <<Int 40 ) modInt (2 ^Int 64) +Int
( ( ( #getRange ( BM , ADDR , 8 ) <<Int 48 ) modInt (2 ^Int 64) >>Int 56 <<Int 48 ) modInt (2 ^Int 64) +Int
( ( ( #getRange ( BM , ADDR , 8 ) <<Int 56 ) modInt (2 ^Int 64) >>Int 56 <<Int 56 ) modInt (2 ^Int 64) +Int 0 )

modInt (2 ^Int 64) )
modInt (2 ^Int 64) )
modInt (2 ^Int 64) )
modInt (2 ^Int 64) )
modInt (2 ^Int 64) )
modInt (2 ^Int 64) )
modInt (2 ^Int 64) )
modInt (2 ^Int 64)
```

\normalsize

> * Each line in the top part represents a byte.
> * The expression is essentially a sum, where the implicit invariant that they can only have `1` bits in separate locations (no addition carry).

First proof attempt
-------------------

Least significant byte in original number, modified the first loop iteration:

```
(((((( #getRange(BM, ADDR, 8) <<Int 56) modInt (2 ^Int 64)) >>Int 56) <<Int 56)
       modInt (2 ^Int 64)) +Int 0) modInt (2 ^Int 64)
```

First proof attempt
-------------------

\begin{minipage}[l]{0.40\textwidth}
\begin{tikzpicture}[scale=0.56]
  \node (a) {modInt}
  child {
    node (b) {+Int}
    child {
      node (d) {modInt}
      child {
        node (e) {<<Int}
        child {
          node (g) {>>Int}
          child {
            node (k) {modInt}
            child {
              node (m) {<<Int}
              child {
                node (n) {\#getRange}
                child {
                  node (p) {BM}
                }
                child {
                  node (q) {ADDR}
                }
                child {
                  node (r) {8}
                }
              }
              child {
                node (o) {\quad 56}
              }
            }
            child {
              node (l) {$2^{64}$}
            }
          }
          child {
            node (j) {56}
          }
        }
        child {
          node (h) {56}
        }
      }
      child {
        node (f) {$2^{64}$}
      }
    }
    child {node  (i) {0}}}
  child {node (c) {$2^{64}$}};
\end{tikzpicture}
\pause
\end{minipage}
\begin{minipage}[r]{0.58\textwidth}

\vspace*{60pt}

\quad\texttt{rule X +Int 0 => X}
\pause

\vspace*{16pt}

\texttt{rule (X modInt M) modInt N => X modInt M}

\texttt{~~requires M >Int 0}

\texttt{~~~andBool M <=Int N}
 \pause

\vspace*{16pt}

\texttt{rule (X <<Int N) modInt POW}

\texttt{~ => (X modInt (POW /Int (2 \^{}Int N)))}

\texttt{~~~~ <<Int N}

\texttt{~~ requires N >=Int 0}

\texttt{~~~ andBool POW >Int 0}

\texttt{~~~ andBool POW modInt (2 \^{}Int N) ==Int 0}

\end{minipage}

New state
---------

\begin{minipage}[l]{0.40\textwidth}
\begin{tikzpicture}[scale=0.56]
  \node (a) {modInt}
  child {
    node (b) {+Int}
    child {
      node (d) {modInt}
      child {
        node (e) {<<Int}
        child {
          node (g) {>>Int}
          child {
            node (k) {modInt}
            child {
              node (m) {<<Int}
              child {
                node (n) {\#getRange}
                child {
                  node (p) {BM}
                }
                child {
                  node (q) {ADDR}
                }
                child {
                  node (r) {8}
                }
              }
              child {
                node (o) {\quad 56}
              }
            }
            child {
              node (l) {$2^{64}$}
            }
          }
          child {
            node (j) {56}
          }
        }
        child {
          node (h) {56}
        }
      }
      child {
        node (f) {$2^{64}$}
      }
    }
    child {node  (i) {0}}}
  child {node (c) {$2^{64}$}};
\end{tikzpicture}
\end{minipage}
\hfill
\begin{minipage}[c]{0pt}
\vfill
\begin{tikzpicture}
\node (a) { \Huge $\Rightarrow$ };
\end{tikzpicture}
\vfill
\end{minipage}
\hfill
\begin{minipage}[r]{0.4\textwidth}
\begin{tikzpicture}[scale=0.6]
  \node (a) {<<Int}
      child {
        node (e) {modInt}
        child {
          node (g) {>>Int}
          child {
            node (m) {<<Int}
            child {
              node (k) {modInt}
              child {
                node (n) {\#getRange}
                child {
                  node (p) {BM}
                }
                child {
                  node (q) {ADDR}
                }
                child {
                  node (r) {8}
                }
              }
              child {
                node (l) {\quad $2^{8}$}
              }
            }
            child {
              node (o) {56}
            }
          }
          child {
            node (j) {56}
          }
        }
        child {
          node (h) {$2^{8}$}
        }
      }
  child {node (c) {56}};
\end{tikzpicture}
\end{minipage}

Second proof attempt
---------

\uncover<2->{\begin{minipage}[r]{0.52\textwidth}

\vspace*{60pt}

\texttt{rule (X <<Int N) >>Int M}

\texttt{~ => X <<Int (N -Int M)}

\texttt{~~ requires N >=Int M}

\vspace*{16pt}

\uncover<3->{

\texttt{rule \#getRange(BM, ADDR, WIDTH) modInt 256}

\texttt{~ => \#get(BM, ADDR)}

\texttt{~ requires WIDTH =/=Int 0}

\texttt{~~ andBool \#isByteMap(BM)}

\texttt{~ ensures 0 <=Int \#get(BM, ADDR)}

\texttt{~~ andBool \#get(BM, ADDR) <Int 256}
}
\end{minipage}
}
\hfill
\only<1->{\begin{minipage}[r]{0.39\textwidth}
\begin{tikzpicture}[scale=0.6]
  \node (a) {<<Int}
      child {
        node (e) {modInt}
        child {
          node (g) {>>Int}
          child {
            node (m) {<<Int}
            child {
              node (k) {modInt}
              child {
                node (n) {\#getRange}
                child {
                  node (p) {BM}
                }
                child {
                  node (q) {ADDR}
                }
                child {
                  node (r) {8}
                }
              }
              child {
                node (l) {\quad$2^{8}$}
              }
            }
            child {
              node (o) {56}
            }
          }
          child {
            node (j) {56}
          }
        }
        child {
          node (h) {$2^{8}$}
        }
      }
  child {node (c) {56}};
\end{tikzpicture}

\end{minipage}
}

New state
---------

\begin{minipage}[r]{0.4\textwidth}
\begin{tikzpicture}
  \node (a) {<<Int}
  child {
    node (n) {\#get}
    child {
      node (p) {BM}
    }
    child {
      node (q) {ADDR}
    }
  }
  child {node (c) {56}};
\end{tikzpicture}
\end{minipage}
\hfill
\begin{minipage}[c]{30pt}
\vfill
\begin{tikzpicture}
\node (a) { \Huge $\Leftarrow$ };
\end{tikzpicture}
\vfill
\end{minipage}
\hfill
\begin{minipage}[r]{0.4\textwidth}
\begin{tikzpicture}[scale=0.6]
  \node (a) {<<Int}
      child {
        node (e) {modInt}
        child {
          node (g) {>>Int}
          child {
            node (m) {<<Int}
            child {
              node (k) {modInt}
              child {
                node (n) {\#getRange}
                child {
                  node (p) {BM}
                }
                child {
                  node (q) {ADDR}
                }
                child {
                  node (r) {8}
                }
              }
              child {
                node (l) {\quad $2^{8}$}
              }
            }
            child {
              node (o) {56}
            }
          }
          child {
            node (j) {56}
          }
        }
        child {
          node (h) {$2^{8}$}
        }
      }
  child {node (c) {56}};
\end{tikzpicture}
\end{minipage}
Conclusion/Questions?
=====================

Thanks!
-------

-   Thanks for listening!
- [rikard.hjort@runtimeverification.com](rikard.hjort@runtimeverification.com)
- https://riot.im/app/#/room/#k:matrix.org

References
----------

\tiny

