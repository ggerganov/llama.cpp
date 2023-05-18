---
title: 'Formally Verifying WebAssembly with KWasm'
subtitle: 'Towards an Automated Prover for Wasm Smart Contracts'
author:
-   Rikard Hjort
-   \tiny Supervisor$\colon$Thomas Sewell
-   \tiny Examiner$\colon$Wolfgang Ahrendt
-   \tiny Opponent$\colon$Jakob Larsson
date: January 30, 2020
institute:
-   M.Sc. Thesis at Chalmers University of Technology, Gothenburg, Sweden\newline Department of Computer Science and Engineering
-   In collaboration with Runtime Verification, Inc., Urbana, IL, United States
abstract: A smart contract is immutable, public bytecode which handles valuable assets. This makes it a prime target for formal methods. WebAssembly (Wasm) is emerging as bytecode format for smart contracts. KWasm is a mechanization of Wasm in the K framework which can be used for formal verification. The current project aims to verify a single smart contract in Wasm by completing unfinished parts of KWasm, making an Ethereum flavored embedding for the Wasm semantics, and incremententally using KWasm for more complex verification, with a focus on heap reasoning. In the process many arithmetic axioms are added to the reasoning capabilities of the K prover.
theme: metropolis
fontsize: 9pt
header-includes:
-   \titlegraphic{\begin{flushright}\includegraphics[width=60pt]{media/img/thesis-logo}\end{flushright}}
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
-   \newcommand{\diminish}[1]{\begin{footnotesize}#1\end{footnotesize}}
-   \newcommand{\K}{$\mathbb{K}$~}
-   \newcommand{\lK}{$\mathbb{K}$}
---

Goals and Motivation
====================

Smart contracts and formal methods
------------

![](media/img/ethereum.png){ width=50%}

- Smart contracts are code that handles money on blockchains.
- Public, immutable, Turing-complete code handling lots of money? Formal verification is worth the effort!
- Ethereum is the most significant platform, with $700+ million locked in financial smart contracts.[^1] 
- Currently execute contracts on their own virtual machine, the *EVM*.

WebAssmebly and Ewasm
----

Planned transition to Ewasm as part of Ethereum 2.0 roadmap.

. . .

- WebAssembly (Wasm) is a stack based low-ish level language.
- Separated into modules with their own functions, global variables, linear memory.

. . .

- Fast
- Small byte code
- Safe memory model
- Compile target of C/C++, Go, Rust, etc.
- Portable

. . .

Wasm is designed to be *embedded*: environment interactions specified by each embedding.

. . .

Ewasm is coming, we want to be able to verify the contracts.

. . .

\flushright\footnotesize (... and Wasm has more applications than just smart contracts.)



WRC20
-----

> * WRC20 is a token contract written in Wasm.
> * The capstone of this project would have been to fully verify it.

. . .

* It has two public functions:
    - `balanceOf ::  address              -> i256`
    - `transfer  :: (address, i256 value) -> i1`

\$i64.reverse\_bytes
--------------------

WRC20 has one helper function:

`reverse_bytes :: i64 -> i64`

. . .

Ethereum sends parameter big-endian encoded, but Wasm uses little-endian.
Parameters and return values need to be converted accordingly.

. . .

In pseudo-code:

```
func $i64.reverse_bytes(i64 input) (result i64) {
  i64 i = 0
  i64 res = 0
  while i < 8 {
    bits = 56 - (i * 8)
    res = res + (((input << bits) >> 56) << bits)
    i++
  }
  return res
}
```

. . .

A pure Wasm function!

Overview
--------


1. The tool: KWasm
2. Finishing KWasm
    * Completing KWasm
    * Ewasm embedding
3. Proofs and axioms
     * Example: Verifying the helper function
     * Axiom engineering
4. Discussion

KWasm as a tool
===============

Why \lK?
--------

* Give \K an operational semantics, and it produces a powerful and automatic reachability logic prover. [@stefanescu-park-yuwen-li-rosu-reachability-prover]

. . .

* Mature
    - Complex languages:
      -   KEVM - 2018 [@hildenbrandt-saxena-zhu-rosu-k-evm]
      -   Java 1.4 - 2015 [@bogdanas-rosu-k-java]
      -   C11 - 2015 [@hathhorn-ellison-rosu-k-c]
      -   KJS - 2015 [@park-stefanescu-rosu-k-js]
      -   KLLVM <https://github.com/kframework/llvm-semantics>
      -   KX86-64 <https://github.com/kframework/X86-64-semantics>

. . .

* Uses non-deterministic rewriting, same formalism as Wasm.

. . .

* The forerunner to KWasm, KEVM, has seen significant adoption, and we want to build on the success.

. . .

* KWasm was a mature prototype when we started---no need to start from scratch.

Steps Towards Finishing KWasm
===========================

Completing KWasm
----------------

![](media/img/github-top-screenshot.png)

* Before the project, Wasm lacked support for more than one module.
* Ad-hoc tests.

. . .

We 

* added module support.
* wrote a test harness to test KWasm against the official conformance tests.

. . .

Each are a topic of their own, skipping over them in the interest of time.

Ewasm embedding
---------------

\center
![](media/img/ewasm-contract.png){ width=90%}

\flushleft

> * An Ewasm contract exposes a `"main"` function that is invoked when the contract is called.
> * Ethereum state is accessed through imported host functions, 256-bit results are returned in the Wasm memory.
> * The Wasm state is cleared between each invocation, persistent data lives in the blockchain client.
> * Design: Ethereum client (EEI) and Wasm semantics composed into one semantics with a thin boundary between them.

Proofs and Axioms
=====================

Verifying Wasm programs
----------------------------

> 1. From the KWasm semantics, \K generates a parser and a deductive program verifier.
> 2. A verification claim is written like a rewrite rule. `rule A => B` should be read as "`A` will eventually always evaluate to `B`".
> 3. A proof is (hopefully) constructed for the entire rewrite by composing rules in the semantics. Some equalities and side conditions are proved by Z3, the SMT solver.
> 3. The automatic prover tries to construct a proof (with the help of Z3 to check constraint satisfiability) that every possible execution path starting in `A` eventually rewrites to `B`.
> 4. We can help the prover by adding lemmas (which must be proven), or axioms (which are taken for true). These must be expressed as context-free rewrite rules, i.e. they can rewrite expressions but not step the machine.

Verification Example: `reverse_bytes` Program Again
-----------------

```
func $i64.reverse_bytes(i64 input) (result i64) {
  i64 i = 0
  i64 res = 0
  while i < 8 {
    bits = 56 - (i * 8)
    res = res + (((input << bits) >> 56) << bits)
    i++
  }
  return res
}
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
      <mdata> BM => ?BM' </mdata>
  requires #inUnsignedRange(i32, ADDR)
// ...
  ensures  #get(BM, ADDR +Int 0) ==Int #get(?BM', ADDR +Int 7 )
   andBool #get(BM, ADDR +Int 1) ==Int #get(?BM', ADDR +Int 6 )
   andBool #get(BM, ADDR +Int 2) ==Int #get(?BM', ADDR +Int 5 )
   andBool #get(BM, ADDR +Int 3) ==Int #get(?BM', ADDR +Int 4 )
// ...
```

. . .

> * To test reversing bytes, we load from memory (linear byte array), call the function, store the result memory, and compare memory locations.
> * Expressed as a side condition (`ensures`).
> * Uses a simplifying sidecondition (`requires`).

First proof attempt
-------------------

The prover runs the full program, but ends up storing the following symbolic expression back to the memory location.

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

> * Our approach will be to extend \lK's ability to reason about these kinds of arithmetic expressions, and about the `#getRange` function
> * Axioms will be added as context-free rewrite rules.
> * Each is simple and has been hand-verified, but not yet machine-verified.
<!--
> * We have chosen to add reasoning capabilities to \K over telling Z3 about the structure of memory. 
-->

. . .

Least significant byte in original number, modified the first loop iteration:

```
(((((( #getRange(BM, ADDR, 8)
       <<Int 56) modInt (2 ^Int 64))
       >>Int 56)
       <<Int 56) modInt (2 ^Int 64))
       +Int 0) modInt (2 ^Int 64)
```

First proof attempt
-------------------
\small

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
\small

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
\small

\uncover<2->{\begin{minipage}[r]{0.52\textwidth}

\vspace*{60pt}

\texttt{rule (X <<Int N) >>Int M}

\texttt{~ => X <<Int (N -Int M)}

\texttt{~~ requires N >=Int M}

\vspace*{10pt}

\uncover<3->{
\texttt{rule X <<Int 0 => X}
}

\vspace*{10pt}

\uncover<4->{

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
\only<1->{\begin{minipage}[r]{0.391\textwidth}
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
\small

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

Result
-----------------

We finished the proof of the `reverse_bytes` function.

. . .


We ended up adding 32 axioms to KWasm.

* 25 axioms that can be upstreamed into \lK's reasoning capabilities.
* 7 relate to the `#get` and `#set` operations of KWasm, and can be used in any KWasm verification with memory access.

Axiom Engineering
-------

> * \K supports adding new axioms (as rewrite rules), or new statements to Z3, to help the prover.
> * Whenever we can, we choose to add lemmas to KWasm and upstream them into the \K tool when they are general enough.
> * Axioms in \K apply unconditionally, so there is a risk of infinite rewrites. (No commutativity or associativity axioms!)
> * We found a lexicographic product which the axioms always decrease.

. . .

$$
(b, e, n)
$$

- $b$: Number and height in parse tree of certain operations, (for now `mod`, `<<`, and `>>`.)
- $e$: Expression size.
- $n$: Sum of absolute values of integers.

Discussion & Conclusion
=======================

Discussion & Conclusion
-----------------------

> * We are missing the capstone: verifying a smart contract ...
> * ... but pieces are in place: completed KWasm, embedding, memory lemmas, an example of pure Wasm verification.
> * \K needs more arithmetic reasoning, and KWasm is spearheading the process.

Future Work
-----------

> * Verifying the full WRC20 would tie it all together. Started work, but still exploratory.
> * Committee with other \K projects on what axioms we want in general.
> * Proving lemmas sound, possible even using the \K prover itself.
> * More control over the prover, proof assistant style.
> * A DSL for the prover---expressing properties over blockchain state is verbose and error prone without it.

. . .

Open source project[^2], currently funded, development is ongoing.

Thanks!
-------

-   Thanks for listening!
- [rikard.hjort@runtimeverification.com](rikard.hjort@runtimeverification.com)
- https://riot.im/app/#/room/#k:matrix.org

\vfill

### References

\tiny

Cover image by Bogdan Stanciu.


[^1]: <https://defipulse.com/>, as of 2020-01-25

[^2]: <https://github.com/kframework/wasm-semantics>

