---
title: 'Semantics of WebAssembly in the \K framework'
author:
-   Rikard Hjort[^supervised] \quad \href{mailto:hjort@hjorthjort.xyz}{hjort@hjorthjort.xyz}
abstract:
  WebAssembly is a low-ish-level language designed to run efficiently on all modern platforms. The Ethereum blockchain currently runs on its own virtual machine (the EVM) but is expected to move to use WebAssembly in the future. The \K framework definition of the EVM has become the *de facto* standard for verifying properties of smart contracts, most notably the Dai coin contracts, currently handling over 100 million USD of value. Since we want to verify Ethereum contracts compiled to WebAssembly -- as well as other WebAssembly programs -- we need to have a \K formalization of WebAssembly. That is what I've been working on.

  I will introduce WebAssembly, how to define languages in \K, and how we have been translating the official WebAssembly specification into \K.\newline

  \indent This report is based on a talk, available in full at \newline <https://www.youtube.com/watch?v=V6tOYuneMqo> \newline
date: \today
institute:
-   Chalmers University of Technology
-   Runtime Verification, Inc.
theme: metropolis
fontsize: 8pt
header-includes:
-   \usepackage{amssymb}
-   \newcommand{\K}{$\mathbb{K}$~}
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
-   \newcommand{\MAX}{\texttt{max}}
-   \newcommand{\MEMADDRS}{\texttt{memaddrs}}
-   \newcommand{\MEMORY}{\texttt{memory}}
-   \newcommand{\MEMS}{\texttt{mems}}
-   \newcommand{\MEMINST}{\textit{meminst}}
-   \newcommand{\MODULE}{\texttt{module}}
-   \newcommand{\SIZE}{\texttt{size}}
-   \newcommand{\TABLES}{\texttt{tables}}
-   \newcommand{\with}{\text{~with~}}
-   \newcommand{\stepto}{~\hookrightarrow~}
-   \newcommand{\wif}[1]{\text{if}~#1}
-   \newcommand{\diminish}[1]{\begin{footnotesize}#1\end{footnotesize}}
---

[^supervised]: supervised by Magnus Myreen. Parts of the report, especially the
    description of \K, is due to Everett Hildenbrandt. \normalsize

\newpage
Background
==========

The motivation for this project came from wanting work on formal verification on
smart contracts. The combination of immutable, public code and processing high
value assets calls for high assurance software engineering. Looking at what has
been done in that space led me to the company MakerDAO[^maker], which famously
have verified the core contracts or their distrubuted application for the Dai
cryptocurrency[^dai-verified]. Much of this verification effort has been carried
out by, or in collaboration with, the organization DappHub[^dapphub]. When I
reached out to them in turn I was told that the *de facto* standard toolset for
verifying smart contracts is the \K framework[^kframework].

[^maker]: <https://makerdao.com/en/>
[^dai-verified]: <https://medium.com/makerdao/the-code-is-ready-2aee2aa62e73>
[^dapphub]: <https://dapphub.com/>
[^kframework]: <http://www.kframework.org/index.php/Main_Page>

The current state-of-the-art method of verifying Ethereum smart contracts goes
something as follows:

1. Contracts are compiled to Ethereum virtual machine (EVM) bytecode.
2. Some property or invariant is specified as a rewrite rule.
3. \K tries to construct a proof (using the SMT solver Z3) that every possible
   execution path eventually rewrites to the correct thing
4. The tool KLab (by DappHub) offers an interactive view of execution paths,
   great for seeing where and why the prover failed.

The reason \K can be used for verifying smart contracts is that there is a full
formalization of the Ethereum Virtual Machine (EVM) in \K, called KEVM
[@hildenbrandt-saxena-zhu-rosu-k-evm]. The EVM is a stack machine of
approximately 120 opcodes, many of which are very similar to eachother. The EVM
is currently the only virtual machine available to run Ethereum contracts.
However, an ongoing project in the Ethereum community is trying to migrate to
"ewasm"[^ewasm]: an Ethereum virtual machine built on top of WebAssembly.

[^ewasm]: <https://github.com/ewasm/design>

There are several reasons listed in the design documents:

- Speed
- Size
- Security
- Write contracts in C/C++, go, or rust
- Static analysis
- Optional metering
- Portability: ewasm contracts will be compatibile with any standard Wasm
  environment, including IoT and mobile devices

Seeing how Ethereum might migrate to ewasm in the future, there is reason to
start looking into formalizing ewasm in \K. Since ewasm mostly extends
WebAssembly (and only alters WebAssembly slightly to allow for gas metering),
the first step to a formalization of ewasm in \K is a formalization of
WebAssembly. As it turns out, the main author of []KEVM, Everett Hildenbrandt,
has been working on a prototype semantics for WebAssembly. Development was put
on hold due to other projects, so I reached out to Everett and asked if I could
pick up the baton, which he happily let me do. Since January 2019, I have been
adding more of the WebAssembly semantics to the project, with help and guidance
from Everett.

This report will serve as a brief introduction to the \K framework, WebAssembly
and KWasm, our formalization of WebAssembly in \K. A reader familiar with \K
and/or WebAssembly may want to skip ahead.

------

\newpage
# Brief introduction to \K #



The \K framework lets one define programming languages by giving their syntax
and semantics. The vision of \K is to provide a language-independent framework
for obtaining not only parsers, interpreters and compilers, but also more
non-standard tools like documentation, program verifiers, and model
checkers. Once a language is definied, all these tools should follow.
Perhaps the most interesting tool that comes out of a \K specification is the
Reachability Logic Prover [@stefanescu-park-yuwen-li-rosu-reachability-prover].

![\K Tooling Overview](media/img/k-overview.png)

Several large and popular programming languages have been specified in \K, among
them Java 1.4 [@bogdanas-rosu-k-java], C11 [@hathhorn-ellison-rosu-k-c], and
JavaScript [@park-stefanescu-rosu-k-js]. There are ongoing projects to specfy
LLVM[^llvm] and X86 assemby[^x86].

[^llvm]: <https://github.com/kframework/llvm-semantics>
[^x86]: <https://github.com/kframework/X86-64-semantics>

A language specification in \K consists of 3 things:

* the concrete syntax,
* a configuration, which is the exectuion state,
* operational semantics as rewrite rules on the configuration.

## \K Specifications: Syntax ##

The concrete syntax is built using EBNF style. As a running example, we use a
small, imperative language.

```k
    syntax Exp ::= Int | Id | "(" Exp ")" [bracket]
                 | Exp "*" Exp
                 > Exp "+" Exp // looser binding

    syntax Stmt ::= Id ":=" Exp
                  | Stmt ";" Stmt
                  | "return" Exp
```

This would allow correctly parsing programs like:

```imp
    a := 3 * 2;
    b := 2 * a + 5;
    return b
```

## \K Specifications: Configuration ##

The configuration gives the structure of the execution state.
For example, a simple imperative language might have the following
configuration.

```k
    configuration <k>     $PGM:Program </k>
                  <env>   .Map         </env>
                  <store> .Map         </store>
```

-   `<k>` will contain the initial parsed program
-   `<env>` contains bindings of variable names to store locations
-   `<store>` contains bindings of store locations to integers


Note that the parsed program gets put into the `<k>` cell. Thus, the program is
part of the execution state, and can be reduced or extended as part of rewrites.

## \K Specifications: Transition Rules ##

The actual operational semantics of a \K program is written as rewrite rules.
For example, we might define the following semantic rules for our imperative
language.

A transition rule takes the form

```k
rule <cellA> X => Y </cellA>
     <cellB> Z => W </cellB>
```

where the arrow represents a rewrite: the value to the left is rewritten to the
value on the right. The above rule says that "if `<cellA>` contains `X` and
`<cellB>` contains `Z`, change the contents of `<cellA>` to `Y` and the contents
of `<cellB>` to `W`." A rule is an atomic transition, so both cells are
rewritten simultaneously. The rewrite arrows are given inside each cell to make
the rules easier to read and write.

Another feature of the rules is that they only need to mention the parts of the
configuration they match on. If the configuration contains a cell `<cellUnused>`
whose contents we do not want to rewrite, and that we do not want to extract any
information from, it does not have to be mentioned in the rule. Actually,
mentioning unsused cells in a rule is bad practice and makes the configuration
less modular: if we at some point were to remove `<cellUnused>` from the
configuration we would have to also edit rules in which it is not used in any
meaningful way.

Perhaps equally convenient, but less intuitive, is that you can leave out
surrounding cells when writing rules. The rewrite

```k
rule <cellNesting>
       <cellInside> X => Y </cellInside>
     </cellNesting>
```

can be equivalently written as

```k
rule <cellInside> X => Y </cellInside>
```

This freedom is convenient, since configuration cell can contain many
nestings, but it also increases modularity, as it allows us to nest the cell
`<cellInside>` with more or less context, without having to edit rules for which
the context is irrelevant.

### Syntactic sugar for matching parts of a cell ###

Idiomatically, the `<k>` cell contains an associative
sequencing operator, `~>`, so the `<k>` cell may look something like

```k
<k> STMT1 ~> STMT2 ~> STMT3 ~> MORE_STATEMENTS </k>
```

A rewrite in the `<k>` cell which only cares about the first of these statements
and replaces it by `NEW` (which could be anything) may then look something like
the following, inside a `rule` declaration:

```k
<k> (STMT1 ~> THE_OTHER_STATEMENTS)
    => (NEW ~> THE_OTHER_STATEMENTS)
</k>
```

Equivalently, we can apply the rewrite in place:

```k
<k> (STMT1 => NEW) ~> THE_OTHER_STATEMENTS </k>
```

To avoid the trouble of explicitly mentioning the entire contents of the cell,
there is syntacit sugar that lets us write a rule like this is with ellipses,
which mean "and something else after `~>`".

```k
<k> STMT1 => NEW ... </k>
```

Finally, when the configuration contains a Map, we may look up a key-value pair
with the key `X` using the following syntax:

```k
<cell> MAP_LEFT X |-> SX MAP_RIGHT </env>
```

meaning there is a bunch of map entries to the left, and a bunch of map entries
to the right. `Map`s in \K are defined as either the empty map, `.Map`, the map
of a single entry `X |-> SX`, or the concatenation of two `Maps`, i.e. `syntax
Map ::= Map Map`, the concatenation operator just being a single space in this
case.

Since the map concatenation is commutative and associative, we can match on any
part of the map, wheras when the operator is just associative, like `~>`, it is
only possible to match on the leftmost operand[^assoc-comm].

[^assoc-comm]: The annotations `assoc` and `comm` marks opeartions associative
    and commutative.

We can write map lookup in a nicer, sugared form:

```k
<cell> ... X |-> SX ... </env>
```

Rewrites on a `Map` take the form

`Map` rewrites take the following forms.

Adding entries:

```k
<cell> MAP_LEFT (.Map => X |-> SX) MAP_RIGHT </env>
// or
<cell> ... (.Map => X |-> SX) ... </env>
```

Changing values:

```k
<cell> MAP_LEFT X |-> (SX => SX') MAP_RIGHT </env>
// or
<cell> ... X |-> (SX => SX') ... </env>
```

There are other, more idiomatic syntaxes for doing lookups, additions, updates
and deletions, but this syntax is the closest to the underlying definition of
`Map`. The interested reader may see the
[`domains.k`](https://github.com/kframework/k/blob/master/k-distribution/include/builtin/domains.k)
that follows with the K framework.

### Variable lookup ###

When an identifier is the first element in the `<k>` cell, rewrite it to its
corresponding value, by first looking up what store location it is bound to.

```k
    rule <k> X:Id => V ... </k>
         <env>   ...  X |-> SX ... </env>
         <store> ... SX |-> V  ... </store>
```

### Variable assignment ###

Similarly, variable assigment is done by looking up the store location a
variable points to and replacing what is there by the right-hand side of the
assignment.

```k
    rule <k> X := I:Int => . ... </k>
         <env>   ...  X |-> SX       ... </env>
         <store> ... SX |-> (V => I) ... </store>
```

Note that, for this rule to apply, the right-hand side must be an integer
literal, i.e., a fully evaluated expression.

### Example Execution ###

To show how to run the \K semantics, we take the program from before and supply
an initial configuration.

```imp
    a := 3 * 2;
    b := 2 * a + 5;
    return b
```

The configuration already has the variables and store locations initialized. In
a more realistic example of an imperative language, we would need to write
syntax and rules for how to declare and allocate variables, but for this
example, it suffices to assume that the configuration looks as follows.

```k
    <k>     a := 3 * 2 ; b := 2 * a + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 0    1 |-> 0 </store>
```

First, \K (using some auxilliary rules of a language like this) is going to
break out the first statement in the initial list of statements, then break out
the right-hand side of the first assignment, and put it back fully evaluated
(let's not bother with how that is done for now: it's just something some other
rules do). Then, \K sees the following configuration ...

```k
    <k>     a := 6 ~> b := 2 * a + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 0    1 |-> 0 </store>
```

and \K realizes that the following rule matches that configuration ...

```k
    rule <k> X := I:Int => . ... </k>
         <env>   ...  X |-> SX       ... </env>
         <store> ... SX |-> (V => I) ... </store>
```

and applies the rule, giving the next configuration:


```k
    <k>               b := 2 * a + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

Again, some other rules apply and \K works its magic, until it gets to this
configuration:

```k
    <k>     a ~> b := 2 * [] + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

What happened was that \K encountered a variable as it tried to evaluate the
right-hand side of the assignment to `b`. So, before it could proceed with the
arithmetic, it needed to get the value of `a`. So \K "heated" `a` and put it
first at the top of the stack, leaving a "hole" in the expression where the
result should be plugged back in.

Now, with a solitary identifier at the head of the `<k>` cell, the following
rule applies ...

```k
    rule <k> X:Id => V ... </k>
         <env>   ...  X |-> SX ... </env>
         <store> ... SX |-> V  ... </store>
```

... leading to this configuration ...

```k
    <k>     6 ~> b := 2 * [] + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

... and after plugging back in the result `6` ...

```k
    <k>          b := 2 * 6 + 5 ; return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

... and again, evaluating the right-hand side to an integer:

```k
    <k>     b := 17 ~> return b </k>
    <env>   a |-> 0    b |-> 1 </env>
    <store> 0 |-> 6    1 |-> 0 </store>
```

After those steps, our assigment rule matches again ...

```k
    rule <k> X := I:Int => . ... </k>
         <env>   ...  X |-> SX       ... </env>
         <store> ... SX |-> (V => I) ... </store>
```

... and we und up with this final configuration

```k
    <k>     return 17           </k>
    <env>   a |-> 0    b |-> 1  </env>
    <store> 0 |-> 6    1 |-> 17 </store>
```

We take the final configuration to be the result of executing the above program.
If we wanted, we could of course define a further rule that deals with return
statements.

In principle, \K executes programs by taking an initial configuration and
producing a resulting configuration, i.e., the configuration we are left with
when no more rules can be applied. In practice, we often start with an empty
configuration, containing only the very mininal initial values, and end when the
`<k>` cell is completely empty.

-------

\newpage Brief introduction to WebAssembly
========

WebAssembly[@rossberg-web-up-to-speed], commonly known as "Wasm", is a low-ish
level language that is neither web-specific, nor really an assembly language in
the normal sense. Some of the boasted features are

- Fast on modern hardware architectures, but platform agnostic.
- Allows stream compiling.
- Efficient byte format, readable text format.
- Safety features
  - Blocks, loops and breaks, but not arbitrary jumps.
  - Static validation.
  - No implicit casts.

Code fold/unfold
----------------

Wasm comes in both a bytecode format and a text format. I will use the text
format for all my examples.

While Wasm is a stack-based, low-level language, with many of the usual
operations one would expect from a stack based assembly language, Wasm code in
the wild often takes a "folded" structure for increased readbility. The folding
uses S-expression syntax to group operations with their operands.

In a regular stack-based fashion, one might write the following piece of code,
which gets the current memory size (in 64 Kb pages), loads a byte from memory,
and adds the size to the loaded byte.


```scheme
(memory.size)      ;; Nullary -- push memory size (i32).
(i64.extend_i32_u) ;; Unary   -- i32 ==> i64.
(local.get $tmp)   ;; Nullary -- push local variable $tmp (i32).
(i64.load8_u)      ;; Unary   -- load 1 byte from argument address, push.
(i64.add)          ;; Binary
```

Wasm allows us to write the exact same piece of code in the following way.

```scheme
(i64.add
    (i64.extend_i32_u (memory.size))
    (i64.load8_u      (local.get $tmp)))
```

Note how each $n$-ary operation is followed by $n$ (folded) instructions.

We can also mix freely between folded and unfolded syntax. The following is yet
another equivalent way to write the code above:

```scheme
(i64.extend_i32_u (memory.size))
(i64.load8_u      (local.get $tmp)))
(i64.add)
```

We are not allowed, however, to fold paritally, i.e., giving the arguemnts to
`add` in this way:

```scheme
(i64.load8_u      (local.get $tmp)))
(i64.add (i64.extend_i32_u (memory.size))) ;; Not valid!
```

We either must give all $n$ arguments in the folded way, or none.

Modules
---

All Wasm code is organized in modules, much like some languages organize all
code into classes. A module may declare functions, allocate and modify their own
linear memory, maintain module-global variables, etc. All these assets are
internal to the module by default, but the module may explicitly export them
with a name, which allows other modules to import, and treat as their own, the
assests of other modules.

The following is a simple module showcasing some of the functionality of Wasm by
declaring a memory and two functions that modify memory.

```scheme

(module
    (memory 1)
    ;; The function $init will get run when the module is loaded.
    (start $init)
    (func $init
        ;; Put 4 bytes of 1's at the start of the memory.
        (i32.store (i32.const 0) (i32.const -1))
    )
    (func   ;; Function descriptors.
            (export "myGrowAndStoreFunction")
            (param $by i32) (param $put i64)
            (result i64)
            (local $tmp i32)

        ;; Body of the function.

        ;; Grow memory, discard result (old memory size, or -1)
        (drop (memory.grow (local.get $by)))
        ;; Set this variable to the first address
        ;; in the last page of memory.
        (local.set $tmp
            (i32.mul
                (i32.sub (memory.size) (i32.const 1))
                (i32.const 65536)))
        ;; Store a value in little-endian form at the location $tmp.
        (i64.store (local.get $tmp) (local.get $put))
        ;; Add the
        (i64.add
            (i64.extend_i32_u (local.get $tmp))
            (i64.load8_u      (local.get $tmp)))
    ) ;; End function.
) ;; End module.
```

We can load this module into a reference interpreter[^ref-interpreter] and
invoke the exported function a few times:

[^ref-interpreter]: <https://github.com/WebAssembly/spec/tree/master/interpreter>


```
$ ./wasm -i ../../my_programs/modules.wast -
wasm 1.0 reference interpreter
> (invoke "myGrowAndStoreFunction" (i32.const 2) (i64.const 0))
131072 : i64
> (invoke "myGrowAndStoreFunction" (i32.const 2) (i64.const 0))
262144 : i64
> (invoke "myGrowAndStoreFunction" (i32.const 0) (i64.const -1))
262399 : i64
```

Wasm Specification
------------------

The offical specification is available at <https://github.com/WebAssembly/spec>.
It is well written and contains both a procedural, prose desription of how to
execute every operation, as well as a rewrite-based small-step semantic rules.

For example, the `(memory.size)` operation has the following execution specification:

1. Let $\FRAME$ be the current frame.
2. \diminish{Assert: due to validation, $\FRAME.\MODULE.\MEMADDRS[0]$
   exists.}[^validation]
3. Let $a$ be the memory address $\FRAME.\MODULE.\MEMADDRS[0]$.[^memIdxZero]
4. \diminish{Assert: due to validation, $\STORE.\MEMS[a]$ exists.}
5. Let $mem$ be the memory instance $\STORE.\MEMS[a]$.
6. Let $sz$ be the length of $mem.\DATA$ divided by the page size.
7. Push the value $\ITHREETWO.\CONST sz$ to the stack.

[^validation]: Validation happens before execution. In KWasm, we always assume
    that code we execute has been validated.
[^memIdxZero]: Every module in Wasm has a single memory for now, so we always
    implicitly work on `memaddrs[0]`.

The semantic rewrite rule is as follows.

$$
\STORE; \FRAME; \MEMORY.\SIZE \stepto \STORE; \FRAME; (\ITHREETWO.\CONST sz)
$$
$$
(\wif{|\STORE.\MEMS[\FRAME.\MODULE.\MEMADDRS[0]].\DATA| = sz \cdot 64 Ki)}
$$

Notice the struct-like access to the members of $\STORE$ and $\FRAME$.
The execution state contains a store of all declared modules, their functions,
memories, etc. It also contains a current exectuion frame with all local
variables and the module that the current call happens in. A memory is a byte
vector and an optional, unsigned 32-bit integer defining the maximum size of the
memory. Here's the relevant part of the execution state:

\begin{alignat*}{5}
%&store      &::=~&\{ & \quad &\FUNCS    ~&\quad &funcinst^*         &     \\
&store      &::=~&\{ & \quad &\dots     ~&\quad &                   &     \\
%&           &    &   & \quad &\TABLES   ~&\quad &tableinst^*        &     \\
&           &    &   & \quad &\MEMS     ~&\quad &meminst^*          &     \\
&           &    &   & \quad &\dots     ~&\quad &                   &\}   \\
%&           &    &   & \quad &\GLOBALS  ~&\quad &globalinst^* \quad &\}   \\
&meminst    &::=~&\{ & \quad &\DATA     ~&\quad &vec(byte)          &     \\
&           &    &   & \quad &\MAX      ~&\quad &u32^?              &\}   \\
&frame      &::=~&\{ & \quad &\LOCALS   ~&\quad &val^*              &     \\
&           &    &   & \quad &\MODULE   ~&\quad &moduleinst   \quad &\}   \\
&moduleinst~&::=~&\{~& \quad &\dots     ~&\quad &                   &     \\
&           &    &   & \quad &\MEMADDRS ~&\quad &memaddr^*          &     \\
&           &    &   & \quad &\dots     ~&\quad &                   &\}
\end{alignat*}

Asterisks indicate an array of 0 or more elements, and a question mark declares
a field as optional.

------

\newpage
KWasm
=====

KWasm is being developed at <https://github.com/kframework/wasm-semantics>.

The high-level goals of KWasm are to give a complete execution semantics of
Wasm, and to serve as a basis for KeWasm.

A complete semantics of Wasm
----------------------------

The goal is to correctly run all valid Wasm programs, i.e., providing semantics
for a superset of Wasm. KWasm may parse programs that are not strictly
syntactically correct Wasm programs, and allows executing standalone script
programs outside of a module.

This approach allows KWasm to run programs outside of modules. For example KWasm
can run the program `(i32.add (i32.const 1) (i32.const 2))` on its own,
producing a value on top of the stack. We consider this to be good from a
pedagogical perspective, and allows us to build the semantics from the bottom
up, starting with simple instructions and moving to modules.

We tend to assume, as a design principle, that any real Wasm program that is
passed to KWasm will first have been validated. From that premise, we can skip
validation steps to keep our semantics simpler.

A basis for KeWasm
------------------

KeWasm should (ideally) only extend KWasm, not modify it. The focus on KeWasm
means that in practice, we may alter the first goal by ignoring floating point
opertions. Implementing correct semantics for floating point arithmetic
according to the Wasm specification is a large undertaking and is not part of
the eWasm semantics, which uses only integer arithmetic, using fixed-point
interpretation to represent fractional numbers.

Future direction
----------------

The semantics are fairly early-stage. Handling memory declaration,
instantiation, load and store are in progress. Everything floating point remains
to be done (but may be pushed until some verification effort requires it). So
does implementing *tables*, a mechanism for having function pointers. Finally,
KWasm does not currently support modules -- as mentioned, we are building the
semantics bottom-up, and modules are the highest-up construct.

KEVM currently has many verified smart contracts at
<https://github.com/runtimeverification/verified-smart-contracts>.
We similarly would like to build a repository of verified code using KeWasm.

Example of a translation from the WebAssembly specification into \K
------------------------------------------------------------------

As a case study, let's look at the `(memory.grow)` operation. It gives an
example of modifying a configuration in `K`, how we diverge from the literal
letter of the Wasm specification for performance reasons, how we use
abstractions and functions to help us develop the semantics, and how we deal
with the few cases of non-determinism in WebAssembly.


### The specification ###

Recall the *store*, *meminst*, *frame* and *modulinst* parts of the runtime
structure:

\begin{alignat*}{5}
%&store      &::=~&\{ & \quad &\FUNCS    ~&\quad &funcinst^*         &     \\
&store      &::=~&\{ & \quad &\dots     ~&\quad &                   &     \\
%&           &    &   & \quad &\TABLES   ~&\quad &tableinst^*        &     \\
&           &    &   & \quad &\MEMS     ~&\quad &meminst^*          &     \\
&           &    &   & \quad &\dots     ~&\quad &                   &\}   \\
%&           &    &   & \quad &\GLOBALS  ~&\quad &globalinst^* \quad &\}   \\
&meminst    &::=~&\{ & \quad &\DATA     ~&\quad &vec(byte)          &     \\
&           &    &   & \quad &\MAX      ~&\quad &u32^?              &\}   \\
&frame      &::=~&\{ & \quad &\LOCALS   ~&\quad &val^*              &     \\
&           &    &   & \quad &\MODULE   ~&\quad &moduleinst   \quad &\}   \\
&moduleinst~&::=~&\{~& \quad &\dots     ~&\quad &                   &     \\
&           &    &   & \quad &\MEMADDRS ~&\quad &memaddr^*          &     \\
&           &    &   & \quad &\dots     ~&\quad &                   &\}
\end{alignat*}

The following is the semantic rules for `(memory.grow)`. Note that there are two
rules which are not mutually exclusive. This is because growing memory may fail
for reasons unknown to Wasm, i.e., the embdedder has no resources to allocate.
In this situation, the operation returns `-1`. Note also that the other rule is
specified in terms of an auxilliary function, $growmem$. This functon may fail,
in which case that rule isn't appied, and failure happens instead.

\begin{align*}
\STORE; \FRAME ; &(\ITHREETWO.\CONST n)~(\MEMORY.\GROW) \stepto \STORE ' ; \FRAME ; (\ITHREETWO.\CONST sz ) \\
(&\wif{\FRAME.\MODULE.\MEMADDRS[0] = a \\
&\land sz = |\STORE.\MEMS[a].\DATA|/64 Ki \\
&\land \STORE ' = \STORE \with \MEMS[a] = growmem(\STORE.\MEMS[a], n))} \\
\\
\STORE; \FRAME ; &(\ITHREETWO.\CONST n)~(\MEMORY.\GROW) \stepto \STORE ; \FRAME ; (\ITHREETWO.\CONST {-1} ) \\
\end{align*}

\begin{align*}
growmem(meminst, n)\quad =\quad &\MEMINST \with \DATA = \MEMINST.\DATA~(\texttt{0x00})^{n \cdot 64 Ki} \\
&(\wif{len = n + |\MEMINST.\DATA|/64 Ki                       \\
&\land len \leq 2^{16}                                        \\
&\land (\MEMINST.\MAX = \epsilon \lor len \leq \MEMINST.\MAX)})
\end{align*}

The $\with$ keyword indicates modifying a single field of a structure, in this
case the $\DATA$ field of a $\MEMINST$.

The `(memory.grow)` operation should thus either succeed with the old size of the
memory pushed to the stack, or fail by pushing a `-1` to the stack. Since the
32-bit unsigned interpretation of `-1` is larger than the max allowed memory
size, `2^{16}`, there is no ambiguity here.


In \K
----

This is the full definition of the `(memory.grow)` operation:

```k
    syntax Instr ::= "(" "memory.grow"       ")"
                   | "(" "memory.grow" Instr ")"
                   | "grow" Int
 // --------------------------------------------
    rule <k>
           ( memory.grow I:Instr ) =>
           I ~> ( memory.grow ) ...
         </k>
    rule <k> ( memory.grow ) => grow N ... </k>
         <stack> < i32 > N : STACK => STACK </stack>

    rule <k> grow N
      => < i32 > #if #growthAllowed(SIZE +Int N, MAX)
                   #then SIZE
                   #else -1
                 #fi ... </k>
         <memAddrs> 0 |-> ADDR </memAddrs>
         <memInst>
           <memAddr> ADDR  </memAddr>
           <mmax>    MAX  </mmax>
           <msize>   SIZE => #if #growthAllowed(SIZE +Int N, MAX)
                               #then SIZE +Int N
                               #else SIZE
                             #fi </msize>
           ...
         </memInst>

    rule <k> grow N => < i32 > -1 </k>
          <deterministicMemoryGrowth>
            false
          </deterministicMemoryGrowth>

    syntax Bool ::= #growthAllowed(Int, MemBound) [function]
 // --------------------------------------------------------
    rule #growthAllowed(SIZE, .MemBound)
      => SIZE <=Int #maxMemorySize()
    rule #growthAllowed(SIZE, I:Int)
      => #growthAllowed(SIZE, .MemBound) andBool SIZE <=Int I
```

The first rule simply unfolds the folded case, and the second rule pops an
operand (which must be of type `i32`, due to validation), and passes it to an
instructon in abstract syntax, `grow`. This instruction is not in Wasm, but
since all syntax defined in \K becomes concrete, it would be possbile to use
this syntax when writing a program for KWasm.

The third and fourth rule implement the semantic rules from the Wasm
specification somewhat faithfully. There is a check made with the help of a
function, `#growthAllowed`, that checks that we don't violate the size
conditions.

A deviation from the official specification is that we store the size of memory
explicitly. The spec, read literally, asks an implementation to allocate full
pages of zereo-bytes when growing (or allocating) memory. For space efficiency,
we store only the size, and represent data as a map from indices to non-zero
bytes.

Finally, we add a flag to the configuration to tell KWasm when to allow
non-deterministic fails. Since we may want to conduct proofs on programs under
the assumption that memory growth does not fail spuriously, we allow a toggle
for this in the configuration, so that proofs can be conducted under different
assumptions. By setting the toggle `<deterministicMemoryGrowth>` to `true`, we
can conduct proofs under the assumption that the foruth rule is never applied.
In the future, we may make this more advanced, by maintaining instead a counter
of available memory pages, decrementing it when memory is allocated or grown,
and stating only allowing the fourth rule when we would run out of memory.

Hopefully this serves as an illustration of how we generally follow the Wasm
specification, but make a few design choices to suit our needs, when we can do
so without changing the semantics of an actual Wasm program.

----

\newpage
References
==========
