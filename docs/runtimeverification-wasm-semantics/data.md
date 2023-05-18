WebAssembly Data
================

```k
module WASM-DATA-SYNTAX
    imports WASM-DATA-COMMON-SYNTAX
endmodule
```

Common Syntax
-------------

```k
module WASM-DATA-COMMON-SYNTAX
    imports INT-SYNTAX
    imports FLOAT-SYNTAX
```

### WASM Token Sorts

```k
    syntax IdentifierToken [token]
    syntax WasmIntToken    [token]
    syntax #Layout         [token]
    syntax WasmStringToken [token]
 // ------------------------------
```

### Identifiers

And we use `OptionalId` to handle the case where an identifier could be omitted.

```k
    syntax Identifier ::= IdentifierToken
    syntax OptionalId ::= "" [klabel(.Identifier), symbol]
                        | Identifier
 // --------------------------------
```

### Strings

Wasm binary data can sometimes be specified as a string.
The string considered to represent the sequence of UTF-8 bytes that encode it.
The exception is for characters that are explicitly escaped which can represent bytes in hexadecimal form.

```k
    syntax WasmString ::= ".WasmString"
                        | WasmStringToken
 // -------------------------------------

    syntax String ::= #parseWasmString   ( WasmStringToken ) [function, total, hook(STRING.token2string)]
 // ----------------------------------------------------------------------------------------------------------

    syntax DataString ::= List{WasmString, ""} [klabel(listWasmString)]
 // -------------------------------------------------------------------
```

### Indices

Many constructions in Wasm, such a functions, labels, locals, types, etc., are referred to by their index.
In the abstract syntax of Wasm, indices are 32 bit unsigned integers.
However, we extend the `Index` sort with another subsort, `Identifier`, in the text format.

```k
    syntax Index ::= WasmInt
 // ------------------------
```

### ElemSegment

Element Segment is a list of indices.
It is used when initializing a WebAssembly table, or used as the parameter of the `br_table` function.

```k
    syntax ElemSegment ::= List{Index, ""} [klabel(listIndex)]
 // ----------------------------------------------------------
```

### WebAssembly Types

#### Base Types

WebAssembly has four basic types, for 32 and 64 bit integers and floats.

```k
    syntax IValType ::= "i32" [klabel(i32), symbol] | "i64" [klabel(i64), symbol]
    syntax FValType ::= "f32" [klabel(f32), symbol] | "f64" [klabel(f64), symbol]
    syntax ValType  ::= IValType | FValType
 // ---------------------------------------
```

#### Type Constructors

There are two basic type-constructors: sequencing (`[_]`) and function spaces (`_->_`).

```k
    syntax ValTypes ::= List{ValType, ""} [klabel(listValTypes), symbol]
 // --------------------------------------------------------------------
```

### Integers

Here we define the rules about what integer formats can be used in Wasm.
For the core language, only regular integers are allowed.

```k
    syntax WasmInt ::= Int
 // ----------------------
```
### Type Mutability

```k
    syntax Mut ::= "const" [klabel(mutConst), symbol]
                 | "var"   [klabel(mutVar), symbol]
 // -----------------------------------------------
```


### Basic Values

WebAssembly values are either integers or floating-point numbers, of 32- or 64-bit widths.

```k
    syntax Number ::= Int | Float
 // -----------------------------
```

### External Values

An `external value` is the runtime representation of an entity that can be `imported` or `exported`.
It is an `address` denoting either a `function instance`, `table instance`, `memory instance`, or `global instances` in the shared store.

```k
    syntax AllocatedKind ::= "func" | "table" | "memory" | "global"
    syntax Externval     ::= AllocatedKind Index
 // --------------------------------------------
```

```k
endmodule
```

`WASM-DATA` module
------------------

```k
module WASM-DATA
    imports WASM-DATA-COMMON-SYNTAX

    imports INT
    imports BOOL
    imports STRING
    imports LIST
    imports MAP
    imports FLOAT
    imports BYTES
    imports K-EQUAL
```

### Identifiers

In `KWASM` rules, we use `#freshId ( Int )` to generate a fresh identifier based on the element index in the current module.
```k
    syntax Identifier ::= #freshId ( Int )
 // --------------------------------------
```

In KWasm we store identifiers in maps from `Identifier` to `Int`, the `Int` being an index.
This rule handles adding an `OptionalId` as a map key, but only when it is a proper identifier.

```k
    syntax Map ::= #saveId (Map, OptionalId, Int) [function]
 // -------------------------------------------------------
    rule #saveId (MAP, ID:OptionalId, _)   => MAP             requires notBool isIdentifier(ID)
    rule #saveId (MAP, ID:Identifier, IDX) => MAP [ID <- IDX]
```

`Int` is the basic form of index, and indices always need to resolve to integers.
In the text format, `Index` is extended with the `Identifier` subsort, and there needs to be a way to resolve it to an `Int`.
In Wasm, the current "context" contains a mapping from identifiers to indices.
The `#ContextLookup` function provides an extensible way to get an `Int` from an index.
Any extension of the `Index` type requires that the function `#ContextLookup` is suitably extended.
For `Int`, however, a the context is irrelevant and the index always just resolves to the integer.

```k
    syntax Int ::= #ContextLookup ( Map , Index ) [function]
 // --------------------------------------------------------
    rule #ContextLookup(_IDS:Map, I:Int) => I
```

### ElemSegment

```k
    syntax Ints ::= List{Int, ""} [klabel(listInt), symbol]
 // -------------------------------------------------------

    syntax Int   ::= #lenElemSegment (ElemSegment)      [function]
    syntax Index ::= #getElemSegment (ElemSegment, Int) [function]
    syntax Int   ::= #lenInts        (Ints)             [function]
    syntax Int   ::= #getInts        (Ints,        Int) [function]
 // --------------------------------------------------------------
    rule #lenElemSegment(.ElemSegment) => 0
    rule #lenElemSegment(_TFIDX    ES) => 1 +Int #lenElemSegment(ES)

    rule #getElemSegment(E _ES, 0) => E
    rule #getElemSegment(_E ES, I) => #getElemSegment(ES, I -Int 1) requires I >Int 0

    rule #lenInts(.Ints) => 0
    rule #lenInts(_TFIDX    ES) => 1 +Int #lenInts(ES)

    rule #getInts(E _ES, 0) => E
    rule #getInts(_E ES, I) => #getInts(ES, I -Int 1) requires I >Int 0

    syntax Ints ::= elemSegment2Ints ( ElemSegment ) [function]
 // -----------------------------------------------------------
    rule elemSegment2Ints(.ElemSegment) => .Ints
    rule elemSegment2Ints(E:Int ES)     => E elemSegment2Ints(ES)
```

### OptionalInt

In some cases, an integer is optional, such as when either giving or omitting the max bound when defining a table or memory.
The sort `OptionalInt` provides this potentially "undefined" `Int`.

```k
    syntax OptionalInt ::= Int | ".Int"
 // -----------------------------------
```

### Limits

Tables and memories have limits, defined as either a single `Int` or two `Int`s, representing min and max bounds.

```k
    syntax Limits ::= #limitsMin(Int)   [klabel(limitsMin),    symbol]
                    | #limits(Int, Int) [klabel(limitsMinMax), symbol]
 // ------------------------------------------------------------------
```

### Type Constructors

There are two basic type-constructors: sequencing (`[_]`) and function spaces (`_->_`).

```k
    syntax VecType  ::= "[" ValTypes "]" [klabel(aVecType), symbol]
 // ---------------------------------------------------------------

    syntax FuncType ::= VecType "->" VecType [klabel(aFuncType), symbol]
 // --------------------------------------------------------------------

    syntax Int ::= lengthValTypes ( ValTypes ) [function, total]
 // -----------------------------------------------------------------
    rule lengthValTypes(.ValTypes) => 0
    rule lengthValTypes(_V VS)     => 1 +Int lengthValTypes(VS)
```

All told, a `Type` can be a value type, vector of types, or function type.

```k
    syntax Type ::= ".Type" | ValType | VecType | FuncType
 // ------------------------------------------------------
```

We can append two `ValTypes`s with the `_+_` operator.

```k
    syntax ValTypes ::= ValTypes "+" ValTypes [function, total]
 // ----------------------------------------------------------------
    rule .ValTypes   + VTYPES' => VTYPES'
    rule (VT VTYPES) + VTYPES' => VT (VTYPES + VTYPES')
```

Also we can reverse a `ValTypes` with `#revt`

```k
    syntax ValTypes ::= #revt ( ValTypes )            [function, total]
                      | #revt ( ValTypes , ValTypes ) [function, total, klabel(#revtAux)]
 // ------------------------------------------------------------------------------------------
    rule #revt(VT) => #revt(VT, .ValTypes)

    rule #revt(.ValTypes, VT') => VT'
    rule #revt(V VT     , VT') => #revt(VT, V VT')
```

### Type Information

The `#width` function returns the bit-width of a given `IValType`.

```k
    syntax Int ::= #width    ( IValType ) [function, total]
    syntax Int ::= #numBytes ( IValType ) [function, total, smtlib(numBytes)]
 // ------------------------------------------------------------------------------
    rule #width(i32) => 32
    rule #width(i64) => 64

    rule #numBytes(ITYPE) => #width(ITYPE) /Int 8 [concrete]
```

`2 ^Int 32` and `2 ^Int 64` are used often enough to warrant providing helpers for them.

```k
    syntax Int ::= #pow  ( IValType ) [function, total, smtlib(pow )] /* 2 ^Int #width(T)          */
                 | #pow1 ( IValType ) [function, total, smtlib(pow1)] /* 2 ^Int (#width(T) -Int 1) */
 // ------------------------------------------------------------------------------------------------------
    rule #pow1(i32) => 2147483648
    rule #pow (i32) => 4294967296
    rule #pow1(i64) => 9223372036854775808
    rule #pow (i64) => 18446744073709551616
```

### Type Mutability

```k
    syntax Mut ::= ".Mut"
 // ----------------------
```

### Values

Proper values are numbers annotated with their types.

```k
    syntax IVal ::= "<" IValType ">" Int    [klabel(<_>_)]
    syntax FVal ::= "<" FValType ">" Float  [klabel(<_>_)]
    syntax  Val ::= "<"  ValType ">" Number [klabel(<_>_)]
                  | IVal | FVal
 // ---------------------------
```

We also add `undefined` as a value, which makes many partial functions in the semantics total.

```k
    syntax Val ::= "undefined"
 // --------------------------
```

#### Value Operations

The `#chop` function will ensure that an integer value is wrapped to the correct bit-width.
The `#wrap` function wraps an integer to a given byte width.
The `#get` function extracts the underlying K integer from a WASM `IVal`.

```k
    syntax IVal ::= #chop ( IVal ) [function, total]
 // -----------------------------------------------------
    rule #chop(< ITYPE > N) => < ITYPE > (N modInt #pow(ITYPE))

    syntax Int ::= #wrap ( Int , Int ) [function, total]
 // ---------------------------------------------------------
    rule [wrap-Positive] : #wrap(WIDTH,  N) => N &Int ((1 <<Int (WIDTH *Int 8)) -Int 1) requires         0 <Int WIDTH
    rule                   #wrap(WIDTH, _N) => 0                                        requires notBool 0 <Int WIDTH

    syntax Int ::= #get( IVal ) [function, total]
 // --------------------------------------------------
    rule #get(< _ > N) => N
```

In `K` all `Float` numbers are of 64-bits width by default, so we need to downcast a `f32` float to 32-bit manually.
The `#round` function casts a `f64` float to a `f32` float.
`f64` floats has 1 bit for the sign, 53 bits for the value and 11 bits for exponent.
`f32` floats has 1 bit for the sign, 23 bits for the value and 8 bits for exponent.

```k
    syntax FVal ::= #round ( FValType , Number ) [function]
 // -------------------------------------------------------
    rule #round(f64 , N:Float) => < f64 > roundFloat(N, 53, 11) [concrete]
    rule #round(f32 , N:Float) => < f32 > roundFloat(N, 24, 8)  [concrete]
    rule #round(f64 , N:Int  ) => < f64 >  Int2Float(N, 53, 11) [concrete]
    rule #round(f32 , N:Int  ) => < f32 >  Int2Float(N, 24, 8)  [concrete]
```

#### Signed Interpretation

Functions `#signed` and `#unsigned` allow for easier operation on twos-complement numbers.
These functions assume that the argument integer is in the valid range of signed and unsigned values of the respective type, so they will not correctly map arbitrary integers into the corret range.
Some operations extend integers from 1, 2, or 4 bytes, so a special function with a bit width argument helps with the conversion.

```k
    syntax Int ::= #signed      ( IValType , Int ) [function]
                 | #unsigned    ( IValType , Int ) [function]
                 | #signedWidth ( Int      , Int ) [function]
 // ---------------------------------------------------------
    rule #signed(ITYPE, N) => N                  requires 0            <=Int N andBool N <Int #pow1(ITYPE)
    rule #signed(ITYPE, N) => N -Int #pow(ITYPE) requires #pow1(ITYPE) <=Int N andBool N <Int #pow (ITYPE)

    rule #unsigned( ITYPE, N) => N +Int #pow(ITYPE) requires N  <Int 0
    rule #unsigned(_ITYPE, N) => N                  requires 0 <=Int N

    rule #signedWidth(1, N) => N            requires 0     <=Int N andBool N <Int 128
    rule #signedWidth(1, N) => N -Int 256   requires 128   <=Int N andBool N <Int 256
    rule #signedWidth(2, N) => N            requires 0     <=Int N andBool N <Int 32768
    rule #signedWidth(2, N) => N -Int 65536 requires 32768 <=Int N andBool N <Int 65536
    rule #signedWidth(4, N) => #signed(i32, N)
```

#### Boolean Interpretation

Function `#bool` converts a `Bool` into an `Int`.

```k
    syntax Int ::= #bool ( Bool ) [function, total]
 // ----------------------------------------------------
    rule #bool( B:Bool ) => 1 requires         B
    rule #bool( B:Bool ) => 0 requires notBool B
```

### Data Structures

WebAssembly is a stack-machine, so here we provide the stack to operate over.
Operator `_++_` implements an append operator for sort `ValStack`.

```k
    syntax ValStack ::= ".ValStack"
                      | Val      ":"  ValStack
                      | ValStack "++" ValStack [function, total]
 // -----------------------------------------------------------------
    rule .ValStack       ++ VALSTACK' => VALSTACK'
    rule (SI : VALSTACK) ++ VALSTACK' => SI : (VALSTACK ++ VALSTACK')
```

`#zero` will create a specified stack of zero values in a given type.
`#take` will take the prefix of a given stack.
`#drop` will drop the prefix of a given stack.
`#revs` will reverse a given stack.

**NOTE**: `#take` and `#drop` are _total_, so in case they could not take/drop enough values before running out, they just return the empty `.ValStack`.
Each call site _must_ ensure that this is desired behavior before using these functions.

```k
    syntax ValStack ::= #zero ( ValTypes )            [function, total]
                      | #take ( Int , ValStack )      [function, total]
                      | #drop ( Int , ValStack )      [function, total]
                      | #revs ( ValStack )            [function, total]
                      | #revs ( ValStack , ValStack ) [function, total, klabel(#revsAux)]
 // ------------------------------------------------------------------------------------------
    rule #zero(.ValTypes)             => .ValStack
    rule #zero(ITYPE:IValType VTYPES) => < ITYPE > 0   : #zero(VTYPES)
    rule #zero(FTYPE:FValType VTYPES) => < FTYPE > 0.0 : #zero(VTYPES)

    rule #take(N, _)         => .ValStack               requires notBool N >Int 0
    rule #take(N, .ValStack) => .ValStack               requires         N >Int 0
    rule #take(N, V : VS)    => V : #take(N -Int 1, VS) requires         N >Int 0

    rule #drop(N, VS)        => VS                  requires notBool N >Int 0
    rule #drop(N, .ValStack) => .ValStack           requires         N >Int 0
    rule #drop(N, _ : VS)    => #drop(N -Int 1, VS) requires         N >Int 0

    rule #revs(VS) => #revs(VS, .ValStack)

    rule #revs(.ValStack, VS') => VS'
    rule #revs(V : VS   , VS') => #revs(VS, V : VS')
```

### Strings

Wasm uses a different character escape rule with K, so we need to define the `unescape` function ourselves.

```k
    syntax String ::= unescape(String)              [function]
                    | unescape(String, Int, String) [function, klabel(unescapeAux)]
 // -------------------------------------------------------------------------------
    rule unescape(S         ) => unescape(S, 1, "")
    rule unescape(S, IDX, SB) => SB                 requires IDX ==Int lengthString(S) -Int 1
```

Unescaped characters just directly gets added to the output.
The escaped character starts with a "\" and followed by 2 hexdigits will be converted to a unescaped character before stored.

```k
    rule unescape(S, IDX, SB) => unescape(S, IDX +Int 1, SB +String substrString(S, IDX, IDX +Int 1))
      requires IDX <Int lengthString(S) -Int 1
       andBool substrString(S, IDX, IDX +Int 1) =/=K "\\"
    rule unescape(S, IDX, SB) => unescape(S, IDX +Int 3, SB +String chrChar(String2Base(substrString(S, IDX +Int 1, IDX +Int 3), 16)))
      requires IDX <Int lengthString(S) -Int 3
       andBool substrString(S, IDX, IDX +Int 1) ==K "\\"
       andBool (findChar("0123456789abcdefABCDEF", substrString(S, IDX +Int 1, IDX +Int 2), 0) =/=Int -1 )
```

The characters "\t", "\n", "\r", """, "'", and "\" are interpreted as regular escaped ascii symbols.

```k
    rule unescape(S, IDX, SB) => unescape(S, IDX +Int 2, SB +String chrChar(String2Base("09", 16)))
      requires IDX <Int lengthString(S) -Int 2
       andBool substrString(S, IDX, IDX +Int 1) ==K "\\"
       andBool substrString(S, IDX +Int 1, IDX +Int 2) ==K "t"
    rule unescape(S, IDX, SB) => unescape(S, IDX +Int 2, SB +String chrChar(String2Base("0A", 16)))
      requires IDX <Int lengthString(S) -Int 2
       andBool substrString(S, IDX, IDX +Int 1) ==K "\\"
       andBool substrString(S, IDX +Int 1, IDX +Int 2) ==K "n"
    rule unescape(S, IDX, SB) => unescape(S, IDX +Int 2, SB +String chrChar(String2Base("0D", 16)))
      requires IDX <Int lengthString(S) -Int 2
       andBool substrString(S, IDX, IDX +Int 1) ==K "\\"
       andBool substrString(S, IDX +Int 1, IDX +Int 2) ==K "r"
    rule unescape(S, IDX, SB) => unescape(S, IDX +Int 2, SB +String chrChar(String2Base("22", 16)))
      requires IDX <Int lengthString(S) -Int 2
       andBool substrString(S, IDX, IDX +Int 1) ==K "\\"
       andBool substrString(S, IDX +Int 1, IDX +Int 2) ==K "\""
    rule unescape(S, IDX, SB) => unescape(S, IDX +Int 2, SB +String chrChar(String2Base("27", 16)))
      requires IDX <Int lengthString(S) -Int 2
       andBool substrString(S, IDX, IDX +Int 1) ==K "\\"
       andBool substrString(S, IDX +Int 1, IDX +Int 2) ==K "'"
    rule unescape(S, IDX, SB) => unescape(S, IDX +Int 2, SB +String chrChar(String2Base("5C", 16)))
      requires IDX <Int lengthString(S) -Int 2
       andBool substrString(S, IDX, IDX +Int 1) ==K "\\"
       andBool substrString(S, IDX +Int 1, IDX +Int 2) ==K "\\"
```

Longer byte sequences can be encoded as escaped "Unicode", with `\u{<hexdigit>+}`.
The implementation is not correct for now because the UTF-8 encoding is not implemented.

```k
    syntax Int ::= #idxCloseBracket ( String, Int ) [function]
 // ----------------------------------------------------------
    rule #idxCloseBracket ( S, I ) => I                                requires substrString(S, I, I +Int 1)  ==String "}"
    rule #idxCloseBracket ( S, I ) => #idxCloseBracket ( S, I +Int 1 ) requires substrString(S, I, I +Int 1) =/=String "}"

    syntax Bytes ::= #encodeUTF8 ( Int ) [function]
 // -----------------------------------------------
    rule #encodeUTF8 (I) => Int2Bytes(I, BE, Unsigned) requires I <=Int 127
    rule #encodeUTF8 (I) => Int2Bytes(((((I &Int 1984) >>Int 6) +Int 192) <<Int 8) +Int ((I &Int 63) +Int 128), BE, Unsigned)
      requires I >=Int 128   andBool I <=Int 2047
    rule #encodeUTF8 (I) => Int2Bytes(((((I &Int 61440) >>Int 12) +Int 224) <<Int 16) +Int ((((I &Int 4032) >>Int 6) +Int 128) <<Int 8) +Int ((I &Int 63) +Int 128), BE, Unsigned)
      requires I >=Int 2048  andBool I <=Int 65535
    rule #encodeUTF8 (I) => Int2Bytes(((((I &Int 1835008) >>Int 18) +Int 240) <<Int 24) +Int ((((I &Int 258048) >>Int 12) +Int 128) <<Int 16) +Int ((((I &Int 4032) >>Int6) +Int 128) <<Int 8) +Int ((I &Int 63) +Int 128), BE, Unsigned)
      requires I >=Int 65536 andBool I <=Int 1114111

    rule unescape(S, IDX, SB) => unescape(S, #idxCloseBracket(S, IDX) +Int 1, SB +String Bytes2String(#encodeUTF8(String2Base(substrString(S, IDX +Int 3, #idxCloseBracket(S, IDX +Int 3)), 16))))
      requires substrString(S, IDX, IDX +Int 1) ==K "\\"
       andBool substrString(S, IDX +Int 1, IDX +Int 2) ==K "u"
```

`DataString`, as is defined in the wasm semantics, is a list of `WasmString`s.
`#concatDS` concatenates them together into a single string.
The strings to connect needs to be unescaped before concatenated, because the `unescape` function removes the quote sign `"` before and after each substring.

```k
    syntax String ::= #concatDS ( DataString )         [function]
                    | #concatDS ( DataString, String ) [function, klabel(#concatDSAux)]
 // -----------------------------------------------------------------------------------
    rule #concatDS ( DS ) => #concatDS ( DS, "" )
    rule #concatDS ( .DataString            , S ) => S
    rule #concatDS (  WS:WasmStringToken DS , S ) => #concatDS ( DS , S +String unescape(#parseWasmString(WS)) )
```

`#DS2Bytes` converts a `DataString` to a K builtin `Bytes`.

```k
    syntax Bytes ::= #DS2Bytes (DataString) [function]
 // --------------------------------------------------
    rule #DS2Bytes(DS) => String2Bytes(#concatDS(DS))
```

### Linear Memory

Wasm memories are byte arrays, sized in pages of 65536 bytes, initialized to be all zero bytes.

- `#setBytesRange(BM, START, BS)` assigns a contiguous chunk of `BS` to `BM` starting at position `N`.
- `#setRange(BM, START, VAL, WIDTH)` writes the integer `I` to memory as bytes (little-endian), starting at index `N`.

```k
    syntax Bytes ::= #setBytesRange ( Bytes , Int , Bytes ) [function, total]
 // ------------------------------------------------------------------------------
    rule #setBytesRange(BM, START, BS) => replaceAtBytes(padRightBytes(BM, START +Int lengthBytes(BS), 0), START, BS)
      requires          0 <=Int START

    rule #setBytesRange(_, START, _ ) => .Bytes
      requires notBool (0 <=Int START)

    syntax Bytes ::= #setRange ( Bytes , Int , Int , Int ) [function, total, smtlib(setRange)]
 // -----------------------------------------------------------------------------------------------
    rule                       #setRange(BM, ADDR, VAL, WIDTH) => BM
      requires notBool (0 <Int WIDTH andBool 0 <=Int VAL andBool 0 <=Int ADDR)

    rule [setRange-Positive] : #setRange(BM, ADDR, VAL, WIDTH) => #setBytesRange(BM, ADDR, Int2Bytes(WIDTH, VAL, LE))
      requires          0 <Int WIDTH andBool 0 <=Int VAL andBool 0 <=Int ADDR
```

- `#getBytesRange(BM, START, WIDTH)` reads off `WIDTH` elements from `BM` beginning at position `START` (padding with zeros as needed).
- `#getRange(BM, START, WIDTH)` reads off `WIDTH` elements from `BM` beginning at position `START`, and converts it into an unsigned integer. The function interprets the range of bytes as little-endian.

```k
    syntax Bytes ::= #getBytesRange ( Bytes , Int , Int ) [function, total]
 // ----------------------------------------------------------------------------
    rule #getBytesRange(_, START, WIDTH) => .Bytes
      requires notBool (0 <=Int START andBool 0 <=Int WIDTH)

    rule #getBytesRange(BM, START, WIDTH) => substrBytes(padRightBytes(BM, START +Int WIDTH, 0), START, START +Int WIDTH)
      requires         (0 <=Int START andBool 0 <=Int WIDTH)
       andBool         START <Int lengthBytes(BM)

    rule #getBytesRange(BM, START, WIDTH) => padRightBytes(.Bytes, WIDTH, 0)
      requires         (0 <=Int START andBool 0 <=Int WIDTH)
       andBool notBool (START <Int lengthBytes(BM))

    syntax Int ::= #getRange(Bytes, Int, Int) [function, total, smtlib(getRange)]
 // ----------------------------------------------------------------------------------
    rule                       #getRange( _, ADDR, WIDTH) => 0
      requires notBool (0 <Int WIDTH andBool 0 <=Int ADDR)

    rule [getRange-Positive] : #getRange(BM, ADDR, WIDTH) => Bytes2Int(#getBytesRange(BM, ADDR, WIDTH), LE, Unsigned)
      requires          0 <Int WIDTH andBool 0 <=Int ADDR
```

```k
endmodule
```
