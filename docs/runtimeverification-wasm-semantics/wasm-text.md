WebAssembly Text Format
=======================

```k
require "wasm.md"
require "data.md"

module WASM-TEXT-SYNTAX
    imports WASM-TEXT
    imports WASM-SYNTAX
    imports WASM-TOKEN-SYNTAX
endmodule
```

Wasm Tokens
-----------

`WASM-TOKEN-SYNTAX` module defines the tokens used in parsing programs.

```k
module WASM-TOKEN-SYNTAX
```

### Strings

In WebAssembly, strings are defined differently to K's built-in strings, so we have to write the definition of WebAssembly `WasmString` in a separate module, and use the module just for parsing the program.
Note that you cannot use a normal K `String` in any production definitions, because the definitions of `String` and `WasmString` overlap, and the K tokenizer does not support ambiguity.

```k
    syntax WasmStringToken ::= r"\\\"(([^\\\"\\\\])|(\\\\[0-9a-fA-F]{2})|(\\\\t)|(\\\\n)|(\\\\r)|(\\\\\\\")|(\\\\')|(\\\\\\\\)|(\\\\u\\{[0-9a-fA-F]{1,6}\\}))*\\\"" [token]
 // -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

### Identifiers

In WebAssembly, identifiers are defined by the regular expression below.

```k
    syntax IdentifierToken ::= r"\\$[0-9a-zA-Z!$%&'*+/<>?_`|~=:\\@^.-]+" [token]
 // ----------------------------------------------------------------------------
```

### Integers

In WebAssembly, integers could be represented in either the decimal form or hexadecimal form.
In both cases, digits can optionally be separated by underscores.

```k
    syntax WasmIntToken ::= r"[\\+-]?[0-9]+(_[0-9]+)*"               [token]
                          | r"[\\+-]?0x[0-9a-fA-F]+(_[0-9a-fA-F]+)*" [token]
 // ------------------------------------------------------------------------
```

### Layout

WebAssembly allows for block comments using `(;` and `;)`, and line comments using `;;`.
Additionally, white-space is skipped/ignored.
Declaring regular expressions of sort `#Layout` infroms the K lexer to drop these tokens.

```k
    syntax #Layout ::= r"\\(;([^;]|(;+([^;\\)])))*;\\)" [token]
                     | r";;[^\\n\\r]*"                  [token]
                     | r"[\\ \\n\\r\\t]"                [token]
 // -----------------------------------------------------------
```

```k
endmodule
```

Wasm Textual Format Syntax
--------------------------

### Values

```k
module WASM-TEXT-COMMON-SYNTAX
    imports WASM-COMMON-SYNTAX

    syntax WasmInt ::= Int
    syntax WasmInt ::= WasmIntToken [klabel(WasmInt), avoid, symbol, function]

    syntax Index ::= Identifier
 // ---------------------------
```

### Instructions

#### Plain Instructions

```k
    syntax PlainInstr ::= "br" Index
                        | "br_if" Index
                        | "br_table" ElemSegment
                        | "call" Index
                        | "global.get" Index
                        | "global.set" Index
                        | "local.get" Index
                        | "local.set" Index
                        | "local.tee" Index
 // ---------------------------------------

    syntax PlainInstr ::= IValType  "." StoreOpM
                        | FValType  "." StoreOpM
                        | IValType "." LoadOpM
                        | FValType "." LoadOpM
    syntax StoreOpM   ::= StoreOp | StoreOp MemArg
    syntax LoadOpM    ::= LoadOp | LoadOp MemArg
    syntax MemArg     ::= OffsetArg | AlignArg | OffsetArg AlignArg
    syntax OffsetArg  ::= "offset=" WasmInt
    syntax AlignArg   ::= "align="  WasmInt
 // ---------------------------------------
```

#### Block Instructions

```k
    syntax Instr ::= BlockInstr
 // ---------------------------

    syntax BlockInstr ::= "block" OptionalId TypeDecls Instrs "end" OptionalId
                        | "loop" OptionalId TypeDecls Instrs "end" OptionalId
                        | "if" OptionalId TypeDecls Instrs "else" OptionalId Instrs "end" OptionalId
                        | "if" OptionalId TypeDecls Instrs                          "end" OptionalId
 // ------------------------------------------------------------------------------------------------
```

##### Folded Instructions

Folded instructions are a syntactic sugar where expressions can be grouped using parentheses for higher readability.

```k
    syntax Instr ::= FoldedInstr
 // ----------------------------
```

One type of folded instruction are `PlainInstr`s wrapped in parentheses and optionally includes nested folded instructions to indicate its operands.

```k
    syntax FoldedInstr ::= "(" PlainInstr Instrs ")"
                         | "(" PlainInstr        ")" [prefer]
 // ---------------------------------------------------------

    syntax FoldedInstr ::= "(" "block" OptionalId TypeDecls Instrs ")"
                         | "(" "loop" OptionalId TypeDecls Instrs ")"
                         | "(" "if" OptionalId TypeDecls Instrs "(" "then" Instrs ")" ")"
                         | "(" "if" OptionalId TypeDecls Instrs "(" "then" Instrs ")" "(" "else" Instrs ")" ")"
 // -----------------------------------------------------------------------------------------------------------
```

### Types

```k
    syntax TypeDefn ::= "(type" OptionalId "(" "func" TypeDecls ")" ")"
 // -------------------------------------------------------------------

    syntax TextLimits ::= Int | Int Int
 // -----------------------------------
```

### Exports

Exports can be declared like regular functions, memories, etc., by giving an inline export declaration.
In that case, it simply desugars to the definition followed by an export of it.
If no identifer is present, one must be introduced so that the export can refer to it.
Note that it is possible to define multiple exports inline, i.e. export a single entity under many names.

```k
    syntax ExportDefn ::= "(" "export" WasmString "(" Externval ")" ")"
 // -------------------------------------------------------------------

    syntax InlineExport  ::= "(" "export" WasmString ")"
 // ----------------------------------------------------
```

### Imports

```k
    syntax ImportDefn ::= "(" "import" WasmString WasmString ImportDesc ")"
 // -----------------------------------------------------------------------
```

Imports can be declared like regular functions, memories, etc., by giving an inline import declaration.

```k
    syntax InlineImport ::= "(" "import" WasmString WasmString ")"
 // --------------------------------------------------------------
```

The following is the text format representation of an import specification.

```k
    syntax ImportDesc ::= "(" "func"   OptionalId TypeUse              ")" [klabel(funcImportDesc)]
                        | "(" "global" OptionalId TextFormatGlobalType ")" [klabel(globImportDesc)]
                        | "(" "table"  OptionalId TableType            ")" [klabel( tabImportDesc)]
                        | "(" "memory" OptionalId MemType              ")" [klabel( memImportDesc)]
 // -----------------------------------------------------------------------------------------------
```

### Functions

```k
    syntax FuncDefn ::= "(" "func" OptionalId  FuncSpec ")"
    syntax FuncSpec ::= TypeUse LocalDecls Instrs
                      | InlineImport TypeUse
                      | InlineExport FuncSpec
 // -----------------------------------------
```

#### Function Local Declaration

```k
    syntax LocalDecl  ::= "(" LocalDecl ")"           [bracket]
                        | "local"            ValTypes
                        | "local" Identifier ValType
    syntax LocalDecls ::= List{LocalDecl , ""}        [klabel(listLocalDecl)]
 // -------------------------------------------------------------------------
```

### Tables

```k
    syntax TableDefn ::= "(" "table" OptionalId TableSpec ")"
    syntax TableSpec ::= TableType
                       | TableElemType "(" "elem" ElemSegment ")"
                       | InlineImport TableType
                       | InlineExport TableSpec
 // -------------------------------------------

    syntax TableType ::= TextLimits TableElemType
    syntax TableElemType ::= "funcref"
 // ----------------------------------
```

### Memories

```k
    syntax MemoryDefn ::= "(" "memory" OptionalId MemorySpec ")"
 // ------------------------------------------------------------

    syntax MemorySpec ::= MemType
 // --------------------------------

    syntax MemorySpec ::= "(" "data" DataString ")"
                        | InlineImport MemType
                        | InlineExport MemorySpec
 // ---------------------------------------------

    syntax MemType    ::= TextLimits
 // --------------------------------
```

### Globals

```k
    syntax GlobalDefn ::= "(" "global" OptionalId  GlobalSpec ")"
    syntax GlobalSpec ::= TextFormatGlobalType Instr
                        | InlineImport TextFormatGlobalType
                        | InlineExport GlobalSpec
 // ---------------------------------------------

    syntax TextFormatGlobalType ::= ValType | "(" "mut" ValType ")"
 // ---------------------------------------------------------------
```

### Offset

The `elem` and `data` initializers take an offset, which is an instruction.
This is not optional.

```k
    syntax Offset ::= "(" "offset" Instrs ")"
 // -----------------------------------------
```

The offset can either be specified explicitly with the `offset` key word, or be a single instruction.

```k
    syntax Offset ::= Instrs
 // ------------------------
```

### Element Segments

```k

    syntax ElemDefn ::= "(" "elem" Index Offset ElemSegment ")"
                      | "(" "elem" Offset        ElemSegment ")"
                      | "(" "elem" Offset "func" ElemSegment ")"
 // ------------------------------------------------------------
```

### Data Segments

```k
    syntax DataDefn ::= "(" "data"     Index Offset DataString ")"
                      | "(" "data" Offset DataString ")"
 // ----------------------------------------------------
```

### Start Function

```k
    syntax StartDefn ::= "(" "start" Index ")"
 // ------------------------------------------
```

### Modules

Modules are defined as a sequence of definitions, that may come in any order.
The only requirements are that all imports must precede all other definitions, and that there may be at most one start function.

```k
    syntax Stmt       ::= ModuleDecl
    syntax ModuleDecl ::= "(" "module" OptionalId Defns ")"
 // -------------------------------------------------------
```

```k
endmodule
```

Translation from Text Format to Core Format
-------------------------------------------

```k
module WASM-TEXT
    imports WASM-TEXT-COMMON-SYNTAX
    imports WASM
```

The text format is a concrete syntax for Wasm.
It allows specifying instructions in a folded, S-expression like format, and a few other syntactic sugars.
Most instructions, those in the sort `PlainInstr`, have identical keywords in the abstract and concrete syntax, and can be used directly.

### Text Integers

All integers given in the text format are automatically turned into regular integers.
That means converting between hexadecimal and decimal when necessary, and removing underscores.

**TODO**: Symbolic reasoning for sort `WasmIntToken` not tested yet.
In the future should investigate which direction the subsort should go.
(`WasmIntToken` under `Int`/`Int` under `WasmIntToken`)

```k
    rule `WasmInt`(VAL) => WasmIntToken2Int(VAL)

    syntax String ::= WasmIntToken2String    ( WasmIntToken ) [function, total, hook(STRING.token2string)]
    syntax Int    ::= WasmIntTokenString2Int ( String       ) [function]
                    | WasmIntToken2Int       ( WasmIntToken ) [function]
 // --------------------------------------------------------------------
    rule WasmIntTokenString2Int(S)  => String2Base(replaceFirst(S, "0x", ""), 16) requires findString(S, "0x", 0) =/=Int -1
    rule WasmIntTokenString2Int(S)  => String2Base(                        S, 10) requires findString(S, "0x", 0)  ==Int -1

    rule WasmIntToken2Int(VAL) => WasmIntTokenString2Int(replaceAll(WasmIntToken2String(VAL), "_", ""))
```

### Identifiers

When we want to specify an identifier, we can do so with the following helper function.

```k
    syntax IdentifierToken ::= String2Identifier(String) [function, total, hook(STRING.string2token)]
 // ------------------------------------------------------------------------------------------------------
```

### Looking up Indices

In the abstract Wasm syntax, indices are always integers.
In the text format, we extend indices to incorporate identifiers.
We also enable context lookups with identifiers.

```k
    rule #ContextLookup(IDS:Map, ID:Identifier) => {IDS [ ID ]}:>Int
      requires ID in_keys(IDS)
```

### Desugaring

The text format is one of the concrete formats of Wasm.
Every concrete format maps to a common structure, described as an abstract syntax.
The function `text2abstract` is a partial function which maps valid programs in the text format to the abstract format.
Some classes of invalid programs, such as those where an identifier appears in a context in which it is not declared, are undefined.

The function deals with the desugarings which are context dependent.
Other desugarings are either left for runtime or expressed as macros (for now).

#### Unfolding Abbreviations

```k
    syntax Stmts ::= unfoldStmts  ( Stmts )                  [function]
    syntax Defns ::= unfoldDefns  ( Defns )                  [function]
                   | #unfoldDefns ( Defns , Int, TypesInfo ) [function]
 // -------------------------------------------------------------------
    rule unfoldStmts(( module OID:OptionalId DS) SS) => ( module OID unfoldDefns(DS) ) unfoldStmts(SS)
    rule unfoldStmts(.Stmts) => .Stmts
    rule unfoldStmts(S SS) => S unfoldStmts(SS) [owise]

    rule unfoldDefns(DS) => #unfoldDefns(DS, 0, types2indices(DS))
    rule #unfoldDefns(.Defns, _, _) => .Defns
    rule #unfoldDefns(D:Defn DS, I, TI) => D #unfoldDefns(DS, I, TI) [owise]

    syntax Defns ::= Defns "appendDefn" Defn [function]
 // ---------------------------------------------------
    rule (D DS) appendDefn D' => D (DS appendDefn D')
    rule .Defns appendDefn D' => D' .Defns
```

#### Types

The text format allows declaring a function without referencing a module-level type for that function.
If there is a module-level type matching the function type, the function is automatically assigned to that type.
The `TypeDecl` of the type is kept, since it may contain parameter identifiers.
If there is no matching module-level type, a new such type is inserted *at the end of the module*.
Since the inserted type is module-level, any subsequent functions declaring the same type will not implicitly generate a new type.

```k
    rule #unfoldDefns(( func _OID:OptionalId (TDECLS:TypeDecls => (type {M [asFuncType(TDECLS)]}:>Int) TDECLS) _LOCALS:LocalDecls _BODY:Instrs ) _DS
                    , _I
                    , #ti(... t2i: M))
      requires         asFuncType(TDECLS) in_keys(M)

    rule #unfoldDefns(( func _OID:OptionalId (TDECLS:TypeDecls => (type N) TDECLS) _LOCALS:LocalDecls _BODY:Instrs ) (DS => DS appendDefn  (type (func TDECLS)))
                   , _I
                   , #ti(... t2i: M => M [ asFuncType(TDECLS) <- N ], count: N => N +Int 1))
      requires notBool asFuncType(TDECLS) in_keys(M)

    rule #unfoldDefns(( func OID:OptionalId TUSE:TypeUse LOCALS:LocalDecls    BODY)   DS, I, TI)
      => (( func OID            TUSE         LOCALS unfoldInstrs(BODY)))
         #unfoldDefns(DS, I, TI)
      requires notBool isTypeDecls(TUSE)

    rule #unfoldDefns(( import MOD NAME (func OID:OptionalId TDECLS:TypeDecls )) DS, I, #ti(... t2i: M) #as TI)
      => (import MOD NAME (func OID (type {M [asFuncType(TDECLS)]}:>Int) TDECLS ))
         #unfoldDefns(DS, I, TI)
      requires         asFuncType(TDECLS) in_keys(M)

    rule #unfoldDefns(( import MOD NAME (func OID:OptionalId TDECLS:TypeDecls)) DS, I, #ti(... t2i: M, count: N))
      => (import MOD NAME (func OID (type N) TDECLS))
         #unfoldDefns(DS appendDefn (type (func TDECLS)), I, #ti(... t2i: M [asFuncType(TDECLS) <- N], count: N +Int 1))
      requires notBool asFuncType(TDECLS) in_keys(M)

    syntax TypesInfo ::= #ti( t2i: Map, count: Int )
    syntax TypesInfo ::=  types2indices( Defns            ) [function]
                       | #types2indices( Defns, TypesInfo ) [function]
 // ------------------------------------------------------------------
    rule types2indices(DS) => #types2indices(DS, #ti(... t2i: .Map, count: 0))

    rule #types2indices(.Defns, TI) => TI

    rule #types2indices((type _OID (func TDECLS)) DS, #ti(... t2i: M, count: N))
      => #types2indices(DS, #ti(... t2i: M [ asFuncType(TDECLS) <- (M [ asFuncType(TDECLS) ] orDefault N) ], count: N +Int 1))

    rule #types2indices(_D DS, M) => #types2indices(DS, M) [owise]
```

#### Functions

```k
    rule #unfoldDefns(( func OID:OptionalId (import MOD NAME) TUSE) DS, I, M)
      => #unfoldDefns(( import MOD NAME (func OID TUSE) ) DS, I, M)

    rule #unfoldDefns(( func EXPO:InlineExport SPEC:FuncSpec ) DS, I, M)
      => #unfoldDefns(( func #freshId(I) EXPO  SPEC) DS, I +Int 1, M)

    rule #unfoldDefns(( func ID:Identifier ( export ENAME ) SPEC:FuncSpec ) DS, I, M)
      => ( export ENAME ( func ID ) ) #unfoldDefns(( func ID SPEC ) DS, I, M)
```

#### Tables

```k
    rule #unfoldDefns(( table funcref ( elem ELEM ) ) DS, I, M)
      => #unfoldDefns(( table #freshId(I) funcref ( elem ELEM ) ) DS, I +Int 1, M)

    rule #unfoldDefns(( table ID:Identifier funcref ( elem ELEM ) ) DS, I, M)
      => ( table ID #lenElemSegment(ELEM) #lenElemSegment(ELEM) funcref ):TableDefn
         ( elem  ID (offset (i32.const 0) .Instrs) ELEM )
         #unfoldDefns(DS, I, M)

    rule #unfoldDefns(( table OID:OptionalId (import MOD NAME) TT:TableType ) DS, I, M)
      => #unfoldDefns(( import MOD NAME (table OID TT) ) DS, I, M)

    rule #unfoldDefns(( table EXPO:InlineExport SPEC:TableSpec ) DS, I, M)
      => #unfoldDefns(( table #freshId(I) EXPO SPEC ) DS, I +Int 1, M)

    rule #unfoldDefns(( table ID:Identifier ( export ENAME ) SPEC:TableSpec ) DS, I, M)
      => ( export ENAME ( table ID ) ) #unfoldDefns(( table ID SPEC ) DS, I, M)
```

#### Memories

```k
    rule #unfoldDefns(( memory ( data DATA ) ) DS, I, M)
      => #unfoldDefns(( memory #freshId(I) ( data DATA ) ) DS, I +Int 1, M)

    rule #unfoldDefns(( memory ID:Identifier ( data DATA ) ) DS, I, M)
      => ( memory ID #lengthDataPages(DATA) #lengthDataPages(DATA) ):MemoryDefn
         ( data   ID (offset (i32.const 0) .Instrs) DATA )
         #unfoldDefns(DS, I, M)

    rule #unfoldDefns(( memory OID:OptionalId (import MOD NAME) MT:MemType ) DS, I, M)
      => #unfoldDefns(( import MOD NAME (memory OID MT  ) ) DS, I, M)

    rule #unfoldDefns(( memory EXPO:InlineExport SPEC:MemorySpec ) DS, I, M)
      => #unfoldDefns(( memory #freshId(I:Int) EXPO SPEC ) DS, I +Int 1, M)

    rule #unfoldDefns(( memory ID:Identifier ( export ENAME ) SPEC:MemorySpec ) DS, I, M)
      => ( export ENAME ( memory ID ) ) #unfoldDefns( ( memory ID SPEC ) DS, I, M)

    syntax Int ::= #lengthDataPages ( DataString ) [function]
 // ---------------------------------------------------------
    rule #lengthDataPages(DS:DataString) => lengthBytes(#DS2Bytes(DS)) up/Int #pageSize()
```

#### Globals

```k
    syntax GlobalType ::= asGMut (TextFormatGlobalType) [function]
 // --------------------------------------------------------------
    rule asGMut ( (mut T:ValType ) ) => var   T
    rule asGMut (      T:ValType   ) => const T

    rule #unfoldDefns((( global OID TYP:TextFormatGlobalType IS:Instr) => #global(... type: asGMut(TYP), init: unfoldInstrs(IS .Instrs), metadata: OID)) _DS, _I, _M)

    rule #unfoldDefns(( global OID:OptionalId (import MOD NAME) TYP ) DS, I, M)
      => #unfoldDefns(( import MOD NAME (global OID TYP ) ) DS, I, M)

    rule #unfoldDefns(( global EXPO:InlineExport SPEC:GlobalSpec ) DS, I, M)
      => #unfoldDefns(( global #freshId(I) EXPO SPEC ) DS, I +Int 1, M)

    rule #unfoldDefns(( global ID:Identifier ( export ENAME ) SPEC:GlobalSpec ) DS, I, M)
      => ( export ENAME ( global ID ) ) #unfoldDefns(( global ID SPEC ) DS, I, M)
```

#### Element Segments

```k
    rule #unfoldDefns(((elem OFFSET func ES) => (elem OFFSET ES)) _DS, _I, _M)
    rule #unfoldDefns(((elem OFFSET:Offset ES ) => ( elem 0 OFFSET ES )) _DS, _I, _M)
    rule #unfoldDefns(((elem IDX OFFSET:Instrs ES ) => ( elem IDX ( offset OFFSET ) ES )) _DS, _I, _M)

    rule #unfoldDefns((elem IDX (offset IS) ES) DS, I, M) => (elem IDX (offset unfoldInstrs(IS)) ES) #unfoldDefns(DS, I, M)
```

#### Data Segments

```k
    rule #unfoldDefns(((data OFFSET:Offset DATA ) => ( data 0 OFFSET DATA )) _DS, _I, _M)
    rule #unfoldDefns(((data IDX OFFSET:Instrs DATA ) => ( data IDX ( offset OFFSET ) DATA )) _DS, _I, _M)

    rule #unfoldDefns((data IDX (offset IS) DATA) DS, I, M) => (data IDX (offset unfoldInstrs(IS)) DATA) #unfoldDefns(DS, I, M)
```

#### Instructions

```k
    syntax Instrs ::=  unfoldInstrs ( Instrs           ) [function]
                    | #unfoldInstrs ( Instrs, Int, Map ) [function]
 // ---------------------------------------------------------------
    rule  unfoldInstrs(IS) => #unfoldInstrs(IS, 0, .Map)
    rule #unfoldInstrs(.Instrs, _, _) => .Instrs
    rule #unfoldInstrs(I IS, DEPTH, M) => I #unfoldInstrs(IS, DEPTH, M) [owise]

    syntax Instrs ::= Instrs "appendInstrs" Instrs      [function]
                    | #appendInstrs  ( Instrs, Instrs ) [function]
                    | #reverseInstrs ( Instrs, Instrs ) [function]
 // --------------------------------------------------------------
    rule IS appendInstrs IS' => #appendInstrs(#reverseInstrs(IS, .Instrs), IS')

    rule #appendInstrs(I IS => IS, IS' => I IS')
    rule #appendInstrs(.Instrs   , IS') => IS'

    rule #reverseInstrs(.Instrs, ACC) => ACC
    rule #reverseInstrs(I IS => IS, ACC => I ACC)
```

##### Block Instructions

In the text format, block instructions can have identifiers attached to them, and branch instructions can refer to these identifiers.
Branching with an identifier is the same as branching to the label with that identifier.
The correct label index is calculated by looking at whih depth the index occured and what depth execution is currently at.

Conceptually, `br ID => br CURRENT_EXECUTION_DEPTH -Int IDENTIFIER_LABEL_DEPTH -Int 1`.

```k
    rule #unfoldInstrs(br       ID:Identifier  IS, DEPTH, M) => br      DEPTH -Int {M [ ID ]}:>Int -Int 1 #unfoldInstrs(IS, DEPTH, M)
    rule #unfoldInstrs(br_if    ID:Identifier  IS, DEPTH, M) => br_if   DEPTH -Int {M [ ID ]}:>Int -Int 1 #unfoldInstrs(IS, DEPTH, M)
    rule #unfoldInstrs(br_table ES:ElemSegment IS, DEPTH, M) => br_table elemSegment2Indices(ES, DEPTH, M) #unfoldInstrs(IS, DEPTH, M)

    syntax ElemSegment ::= elemSegment2Indices( ElemSegment, Int, Map ) [function]
 // ------------------------------------------------------------------------------
    rule elemSegment2Indices(.ElemSegment    , _DEPTH, _M) => .ElemSegment
    rule elemSegment2Indices(ID:Identifier ES,  DEPTH,  M) => DEPTH -Int {M [ ID ]}:>Int -Int 1 elemSegment2Indices(ES, DEPTH, M)
    rule elemSegment2Indices(E             ES,  DEPTH,  M) => E                                 elemSegment2Indices(ES, DEPTH, M) [owise]
```

There are several syntactic sugars for block instructions, some of which may have identifiers.
The same identifier can optionally be repeated at the end of the block instruction (to mark which block is ending) and on the branches in an `if`.
`if` blocks may omit the `else`-branch, as long as the type declaration is empty.

```k
    rule #unfoldInstrs( (block ID:Identifier TDS IS end _OID' => block    TDS IS end) _IS',  DEPTH,  M => M [ ID <- DEPTH ])
    rule #unfoldInstrs(block TDS:TypeDecls IS end IS', DEPTH, M) => block TDS #unfoldInstrs(IS, DEPTH +Int 1, M) end #unfoldInstrs(IS', DEPTH, M)

    rule #unfoldInstrs( (loop ID:Identifier TDS IS end _OID' => loop    TDS IS end) _IS',  DEPTH,  M => M [ ID <- DEPTH ])
    rule #unfoldInstrs(loop TDS:TypeDecls IS end IS', DEPTH, M) => loop TDS #unfoldInstrs(IS, DEPTH +Int 1, M) end #unfoldInstrs(IS', DEPTH, M)

   // TODO: Only unfold empty else-branch if the type declaration is empty.
    rule #unfoldInstrs( (if ID:Identifier  TDS      IS                                   end _OID'' => if  ID TDS IS else .Instrs end) _IS'', _DEPTH, _M)
    rule #unfoldInstrs( (if                TDS      IS                                   end _OID'' => if     TDS IS else .Instrs end) _IS'', _DEPTH, _M)
    rule #unfoldInstrs( (if ID:Identifier  TDS      IS         else _OID':OptionalId IS' end _OID'' => if     TDS IS else IS'     end) _IS'',  DEPTH,  M => M [ ID <- DEPTH ])
    rule #unfoldInstrs(if TDS IS else IS' end IS'', DEPTH, M) => if TDS #unfoldInstrs(IS, DEPTH +Int 1, M) else #unfoldInstrs(IS', DEPTH +Int 1, M) end #unfoldInstrs(IS'', DEPTH, M)
```

#### Folded Instructions

```k
    rule #unfoldInstrs(( PI:PlainInstr  IS:Instrs ):FoldedInstr IS', DEPTH, M)
      =>             (#unfoldInstrs(IS        , DEPTH, M)
         appendInstrs #unfoldInstrs(PI .Instrs, DEPTH, M))
         appendInstrs #unfoldInstrs(IS'       , DEPTH, M)
    rule #unfoldInstrs(( PI:PlainInstr            ):FoldedInstr IS', DEPTH, M)
      =>              #unfoldInstrs(PI .Instrs, DEPTH, M)
         appendInstrs #unfoldInstrs(IS'       , DEPTH, M)
```

Another type of folded instruction is control flow blocks wrapped in parentheses, in which case the `end` keyword is omitted.

```k
    rule #unfoldInstrs(((block ID:Identifier TDS IS)          => block ID TDS IS end) _IS', _DEPTH, _M)
    rule #unfoldInstrs(((block               TDS IS)          => block    TDS IS end) _IS', _DEPTH, _M)

    rule #unfoldInstrs(((loop ID:Identifier TDS IS)          => loop ID TDS IS end) _IS', _DEPTH, _M)
    rule #unfoldInstrs(((loop               TDS IS)          => loop    TDS IS end) _IS', _DEPTH, _M)

    rule #unfoldInstrs(((if OID:OptionalId TDS COND (then IS)) => (if OID TDS COND (then IS) (else .Instrs))) _IS'', _DEPTH, _M)
    rule #unfoldInstrs(((if ID:Identifier  TDS COND (then IS) (else IS')) IS'':Instrs) => (COND appendInstrs if ID TDS IS else IS' end IS''), _DEPTH, _M)
    rule #unfoldInstrs(((if                TDS COND (then IS) (else IS')) IS'':Instrs) => (COND appendInstrs if    TDS IS else IS' end IS''), _DEPTH, _M)
```

#### Structuring Modules

The text format allows definitions to appear in any order in a module.
In the abstract format, the module is a record, one for each type of definition.
The following functions convert the text format module, given as a list of definitions, into the abstract format.
In doing so, the respective ordering of all types of definitions are preserved.

```k
    syntax Stmts ::= structureModules ( Stmts ) [function]
 // ------------------------------------------------------
    rule structureModules((module OID:OptionalId DS) SS) => structureModule(DS, OID) structureModules(SS)
    rule structureModules(.Stmts) => .Stmts
    rule structureModules(S SS) => S structureModules(SS) [owise]

    syntax ModuleDecl ::=  structureModule ( Defns , OptionalId ) [function]
                        | #structureModule ( Defns , ModuleDecl ) [function]
 // ------------------------------------------------------------------------
    rule structureModule(DEFNS, OID) => #structureModule(#reverseDefns(DEFNS, .Defns), #emptyModule(OID))

    rule #structureModule(.Defns, SORTED_MODULE) => SORTED_MODULE

    rule #structureModule((T:TypeDefn   DS:Defns => DS), #module(... types:       TS => T TS))
    rule #structureModule((I:ImportDefn DS:Defns => DS), #module(... importDefns: IS => I IS))
    rule #structureModule((X:FuncDefn   DS:Defns => DS), #module(... funcs:       FS => X FS))
    rule #structureModule((X:GlobalDefn DS:Defns => DS), #module(... globals:     GS => X GS))
    rule #structureModule((T:TableDefn  DS:Defns => DS), #module(... tables:      TS => T TS))
    rule #structureModule((M:MemoryDefn DS:Defns => DS), #module(... mems:        MS => M MS))
    rule #structureModule((E:ExportDefn DS:Defns => DS), #module(... exports:     ES => E ES))
    rule #structureModule((I:DataDefn   DS:Defns => DS), #module(... data:        IS => I IS))
    rule #structureModule((I:ElemDefn   DS:Defns => DS), #module(... elem:        IS => I IS))
    rule #structureModule((S:StartDefn  DS:Defns => DS), #module(... start:   .Defns => S .Defns))

    syntax Defns ::= #reverseDefns(Defns, Defns) [function]
 // -------------------------------------------------------
    rule #reverseDefns(       .Defns  , ACC) => ACC
    rule #reverseDefns(D:Defn DS:Defns, ACC) => #reverseDefns(DS, D ACC)
```

### Replacing Identifiers and Unfolding Instructions

The desugaring is done on the module level.
First, if the program is just a list of definitions, that's an abbreviation for a single module.
If not, we distribute the text to abstract transformation out over all the statements in the file.

**TODO:**

-   Get rid of inline type declarations.
    The text format allows specifying the type directly in the function header using the `param` and `result` keywords.
    However, these will either be desugared to a new top-level `type` declaration or they must match an existing one.
    In the abstract format, a function's type is a pointer to a top-level `type` declaration.
    This could either be done by doing an initial pass to gather all type declarations, or they could be desugared locally, which is similar to what we do currently: `(func (type X) TDS:TDecls ... ) => (func (type X))` and `(func TDS:TDecls ...) => (type TDECLS) (func (type NEXT_TYPE_ID))`.
-   Remove module names.
-   Give the text format and abstract format different sorts, and have `text2abstract` handle the conversion.
    Then identifiers and other text-only constructs can be completely removed from the abstract format.


#### The Context

The `Context` contains information of how to map text-level identifiers to corresponding indices.
Record updates can currently not be done in a function rule which also does other updates, so we have helper functions to update specific fields.

```k
    syntax Context ::= ctx(localIds: Map, globalIds: Map, funcIds: Map, typeIds: Map)
                     | #freshCtx ( )                               [function, total]
                     | #updateLocalIds    ( Context , Map )        [function, total]
                     | #updateLocalIdsAux ( Context , Map , Bool ) [function, total]
                     | #updateFuncIds     ( Context , Map )        [function, total]
                     | #updateFuncIdsAux  ( Context , Map , Bool ) [function, total]
 // -------------------------------------------------------------------------------------
    rule #freshCtx ( ) => ctx(... localIds: .Map, globalIds: .Map, funcIds: .Map, typeIds: .Map)

    rule #updateLocalIds(C, M) => #updateLocalIdsAux(C, M, false)
    rule #updateLocalIdsAux(ctx(... localIds: (_ => M)), M, false => true)
    rule #updateLocalIdsAux(C, _, true) => C

    rule #updateFuncIds(C, M) => #updateFuncIdsAux(C, M, false)
    rule #updateFuncIdsAux(ctx(... funcIds: (_ => M)), M, false => true)
    rule #updateFuncIdsAux(C, _, true) => C
```

#### Traversing the Full Program

The program is traversed in full once, context being gathered along the way.
Since we do not have polymorphic functions available, we define one function per sort of syntactic construct we need to traverse, and for each type of list we encounter.

```k
    syntax Stmt       ::= "#t2aStmt"       "<" Context ">" "(" Stmt       ")" [function]
    syntax ModuleDecl ::= "#t2aModuleDecl" "<" Context ">" "(" ModuleDecl ")" [function]
    syntax ModuleDecl ::= "#t2aModule"     "<" Context ">" "(" ModuleDecl ")" [function]
    syntax Defn       ::= "#t2aDefn"       "<" Context ">" "(" Defn       ")" [function]
 // ------------------------------------------------------------------------------------
    rule text2abstract(DS:Defns) => text2abstract(( module DS ) .Stmts)
    rule text2abstract(SS)       => #t2aStmts<#freshCtx()>(structureModules(unfoldStmts(SS))) [owise]

    rule #t2aStmt<C>(M:ModuleDecl) => #t2aModuleDecl<C>(M)
    rule #t2aStmt<C>(D:Defn)  => #t2aDefn<C>(D)
    rule #t2aStmt<C>(I:Instr) => #t2aInstr<C>(I)
    rule #t2aStmt<_>(S) => S [owise]

    rule #t2aModuleDecl<_>(#module(... types: TS, funcs: FS, globals: GS, importDefns: IS) #as M) => #t2aModule<ctx(... localIds: .Map, globalIds: #idcGlobals(IS, GS), funcIds: #idcFuncs(IS, FS), typeIds: #idcTypes(TS))>(M)
    rule #t2aModule<ctx(... funcIds: FIDS) #as C>(#module(... types: TS, funcs: FS, tables: TABS, mems: MS, globals: GS, elem: EL, data: DAT, start: S, importDefns: IS, exports: ES, metadata: #meta(... id: OID)))
      => #module( ... types: #t2aDefns<C>(TS)
                    , funcs: #t2aDefns<C>(FS)
                    , tables: #t2aDefns<C>(TABS)
                    , mems: #t2aDefns<C>(MS)
                    , globals: #t2aDefns<C>(GS)
                    , elem: #t2aDefns<C>(EL)
                    , data: #t2aDefns<C>(DAT)
                    , start: #t2aDefns<C>(S)
                    , importDefns: #t2aDefns<C>(IS)
                    , exports: #t2aDefns<C>(ES)
                    , metadata: #meta(... id: OID, funcIds: FIDS, filename: .String)
                )
```

#### Types

```k
    rule #t2aDefn<_>((type OID (func TDECLS))) => #type(... type: asFuncType(TDECLS), metadata: OID)
```

#### Imports

```k
    rule #t2aDefn<ctx(... typeIds: TIDS)>(( import MOD NAME (func   OID:OptionalId (type ID:Identifier)            ))) => #import(MOD, NAME, #funcDesc(... id: OID:OptionalId, type: {TIDS[ID]}:>Int))
    rule #t2aDefn<ctx(... typeIds: TIDS)>(( import MOD NAME (func   OID:OptionalId (type ID:Identifier) _:TypeDecls))) => #import(MOD, NAME, #funcDesc(... id: OID:OptionalId, type: {TIDS[ID]}:>Int))
    rule #t2aDefn<_                     >(( import MOD NAME (func   OID:OptionalId (type IDX:Int)                  ))) => #import(MOD, NAME, #funcDesc(... id: OID:OptionalId, type: IDX))
    rule #t2aDefn<_                     >(( import MOD NAME (func   OID:OptionalId (type IDX:Int      ) _:TypeDecls))) => #import(MOD, NAME, #funcDesc(... id: OID:OptionalId, type: IDX))

    rule #t2aDefn<_                     >(( import MOD NAME (global OID:OptionalId TYP:TextFormatGlobalType)))         => #import(MOD, NAME, #globalDesc(... id: OID:OptionalId, type: asGMut(TYP)))

    rule #t2aDefn<_                     >(( import MOD NAME (table  OID:OptionalId LIM:TextLimits funcref)))           => #import(MOD, NAME, #tableDesc(...  id: OID:OptionalId, type: t2aLimits(LIM)))
    rule #t2aDefn<_                     >(( import MOD NAME (memory OID:OptionalId LIM:TextLimits        )))           => #import(MOD, NAME, #memoryDesc(... id: OID:OptionalId, type: t2aLimits(LIM)))
```

#### Globals

```k
    rule #t2aDefn<C>(#global(... type: GTYP, init: IS, metadata: OID)) => #global(... type: GTYP, init: #t2aInstrs<C>(IS), metadata: OID)
```

#### Functions

After unfolding, each type use in a function starts with an explicit reference to a module-level function.

```k
    rule #t2aDefn<ctx(... typeIds: TIDS) #as C>(( func OID:OptionalId T:TypeUse LS:LocalDecls IS:Instrs ))
      => #func(... type: typeUse2typeIdx(T, TIDS)
                 , locals: locals2vectype(LS)
                 , body: #t2aInstrs <#updateLocalIds(C, #ids2Idxs(T, LS))>(IS)
                 , metadata: #meta(... id: OID, localIds: #ids2Idxs(T, LS))
              )

    syntax Int ::= typeUse2typeIdx ( TypeUse , Map ) [function]
 // -----------------------------------------------------------
    rule typeUse2typeIdx( (type IDX ) _:TypeDecls => (type IDX), _TIDS )

    rule typeUse2typeIdx( (type ID:Identifier )  ,  TIDS ) => {TIDS [ ID ]}:>Int
    rule typeUse2typeIdx( (type IDX:Int       )  , _TIDS ) => IDX

    syntax VecType ::=  locals2vectype ( LocalDecls            ) [function]
                     | #locals2vectype ( LocalDecls , ValTypes ) [function]
 // -----------------------------------------------------------------------
    rule  locals2vectype(LDECLS) => #locals2vectype(LDECLS, .ValTypes)

    rule #locals2vectype(.LocalDecls                                             , VTYPES) => [ VTYPES ]
    rule #locals2vectype(local                VTYPES':ValTypes LDECLS:LocalDecls , VTYPES) => #locals2vectype(LDECLS , VTYPES + VTYPES')
    rule #locals2vectype(local _ID:Identifier VTYPE:ValType    LDECLS:LocalDecls , VTYPES) => #locals2vectype(LDECLS , VTYPES + VTYPE .ValTypes)
```

#### Tables

```k
    rule #t2aDefn<_>((table OID:OptionalId LIMITS:TextLimits funcref )) => #table(... limits: t2aLimits(LIMITS), metadata: OID)
```

#### Memories

```k
    rule #t2aDefn<_>((memory OID:OptionalId LIMITS:TextLimits )) => #memory(... limits: t2aLimits(LIMITS), metadata: OID)
```

```k
    syntax Limits ::= t2aLimits(TextLimits) [function, total]
 // --------------------------------------------------------------
    rule t2aLimits(MIN:Int) => #limitsMin(MIN)
    rule t2aLimits(MIN:Int MAX:Int) => #limits(MIN, MAX)
```

#### Start Function

```k
    rule #t2aDefn<ctx(... funcIds: FIDS)>(( start ID:Identifier )) => #start({FIDS[ID]}:>Int)
      requires ID in_keys(FIDS)
    rule #t2aDefn<_>(( start I:Int )) => #start(I)
```

#### Element Segments

Wasm currently supports only one table, so we do not need to resolve any identifiers.

```k
    rule #t2aDefn<C>(( elem _:Index (offset IS) ES )) => #elem(0, #t2aInstrs<C>(IS), #t2aElemSegment<C>(ES) )

    syntax Ints ::= "#t2aElemSegment" "<" Context ">" "(" ElemSegment ")" [function]
 // --------------------------------------------------------------------------------
    rule #t2aElemSegment<ctx(... funcIds: FIDS) #as C>(ID:Identifier ES) => {FIDS[ID]}:>Int #t2aElemSegment<C>(ES)
      requires ID in_keys(FIDS)
    rule #t2aElemSegment<C>(I:Int ES) => I #t2aElemSegment<C>(ES)
    rule #t2aElemSegment<_C>(.ElemSegment) => .Ints
```

#### Data Segments

Wasm currently supports only one memory, so we do not need to resolve any identifiers.

```k
    rule #t2aDefn<C>(( data _:Index (offset IS) DS )) => #data(0, #t2aInstrs<C>(IS), #DS2Bytes(DS))
```

#### Exports

```k
    rule #t2aDefn<ctx(...   funcIds: IDS)>(( export ENAME ( func   ID:Identifier ) )) => #export(ENAME, {IDS[ID]}:>Int) requires ID in_keys(IDS)
    rule #t2aDefn<ctx(... globalIds: IDS)>(( export ENAME ( global ID:Identifier ) )) => #export(ENAME, {IDS[ID]}:>Int) requires ID in_keys(IDS)
    rule #t2aDefn<_>(( export ENAME ( func   I:Int ) )) => #export(ENAME, I)
    rule #t2aDefn<_>(( export ENAME ( global I:Int ) )) => #export(ENAME, I)

    rule #t2aDefn<_>(( export ENAME ( table   _ ) )) => #export(ENAME, 0)
    rule #t2aDefn<_>(( export ENAME ( memory  _ ) )) => #export(ENAME, 0)
```

#### Other Definitions

```k
    rule #t2aDefn<_C>(D:Defn) => D [owise]
```

#### Instructions

```k
    syntax Instr ::= "#t2aInstr" "<" Context ">" "(" Instr ")" [function]
 // ---------------------------------------------------------------------
    rule #t2aInstr<C>(( PI:PlainInstr  IS:Instrs ):FoldedInstr) => ({#t2aInstr<C>(PI)}:>PlainInstr #t2aInstrs<C>(IS))
    rule #t2aInstr<C>(( PI:PlainInstr            ):FoldedInstr) =>  #t2aInstr<C>(PI)
```

#### Basic Instructions

```k
    rule #t2aInstr<_>(unreachable) => unreachable
    rule #t2aInstr<_>(nop)         => nop
    rule #t2aInstr<_>(br L:Int)    => #br(L)
    rule #t2aInstr<_>(br_if L:Int) => #br_if(L)
    rule #t2aInstr<_>(br_table ES) => #br_table(elemSegment2Ints(ES))
    rule #t2aInstr<_>(return)      => return

    rule #t2aInstr<ctx(... funcIds: FIDS)>(call ID:Identifier) => #call({FIDS[ID]}:>Int)
      requires ID in_keys(FIDS)
    rule #t2aInstr<_>                     (call I:Int)         => #call(I)

    rule #t2aInstr<_>(call_indirect TU) => call_indirect TU
```

#### Parametric Instructions

```k
    rule #t2aInstr<_>(drop)   => drop
    rule #t2aInstr<_>(select) => select
```

#### Variable Instructions

```k
    rule #t2aInstr<ctx(... localIds: LIDS)>(local.get ID:Identifier) => #local.get({LIDS[ID]}:>Int)
      requires ID in_keys(LIDS)
    rule #t2aInstr<ctx(... localIds: LIDS)>(local.set ID:Identifier) => #local.set({LIDS[ID]}:>Int)
      requires ID in_keys(LIDS)
    rule #t2aInstr<ctx(... localIds: LIDS)>(local.tee ID:Identifier) => #local.tee({LIDS[ID]}:>Int)
      requires ID in_keys(LIDS)

    rule #t2aInstr<_>(local.get I:Int) => #local.get(I)
    rule #t2aInstr<_>(local.set I:Int) => #local.set(I)
    rule #t2aInstr<_>(local.tee I:Int) => #local.tee(I)

    rule #t2aInstr<ctx(... globalIds: GIDS)>(global.get ID:Identifier) => #global.get({GIDS[ID]}:>Int)
      requires ID in_keys(GIDS)
    rule #t2aInstr<ctx(... globalIds: GIDS)>(global.set ID:Identifier) => #global.set({GIDS[ID]}:>Int)
      requires ID in_keys(GIDS)

    rule #t2aInstr<_>(global.get I:Int) => #global.get(I)
    rule #t2aInstr<_>(global.set I:Int) => #global.set(I)
```

#### Memory Instructions

`MemArg`s can optionally be passed to `load` and `store` operations.
The `offset` parameter is added to the the address given on the stack, resulting in the "effective address" to store to or load from.
The `align` parameter is for optimization only and is not allowed to influence the semantics, so we ignore it.

```k
    rule #t2aInstr<_>(ITYPE:IValType.OP:StoreOp)        => #store(ITYPE, OP, 0)
    rule #t2aInstr<_>(ITYPE:IValType.OP:StoreOp MemArg) => #store(ITYPE, OP, #getOffset(MemArg))
    rule #t2aInstr<_>(FTYPE:FValType.OP:StoreOp)        => #store(FTYPE, OP, 0)
    rule #t2aInstr<_>(FTYPE:FValType.OP:StoreOp MemArg) => #store(FTYPE, OP, #getOffset(MemArg))
    rule #t2aInstr<_>(ITYPE:IValType.OP:LoadOp)         => #load(ITYPE, OP, 0)
    rule #t2aInstr<_>(ITYPE:IValType.OP:LoadOp MemArg)  => #load(ITYPE, OP, #getOffset(MemArg))
    rule #t2aInstr<_>(FTYPE:FValType.OP:LoadOp)         => #load(FTYPE, OP, 0)
    rule #t2aInstr<_>(FTYPE:FValType.OP:LoadOp MemArg)  => #load(FTYPE, OP, #getOffset(MemArg))
    rule #t2aInstr<_>(memory.size)                => memory.size
    rule #t2aInstr<_>(memory.grow)                => memory.grow

    syntax Int ::= #getOffset ( MemArg ) [function, total]
 // -----------------------------------------------------------
    rule #getOffset(           _:AlignArg) => 0
    rule #getOffset(offset= OS           ) => OS
    rule #getOffset(offset= OS _:AlignArg) => OS
```

#### Numeric Instructions

```k
    rule #t2aInstr<_>(ITYPE:IValType.const I) => ITYPE.const I
    rule #t2aInstr<_>(FTYPE:FValType.const N) => FTYPE.const N
    rule #t2aInstr<_>(ITYPE.OP:IUnOp)         => ITYPE.OP
    rule #t2aInstr<_>(FTYPE.OP:FUnOp)         => FTYPE.OP
    rule #t2aInstr<_>(ITYPE.OP:IBinOp)        => ITYPE.OP
    rule #t2aInstr<_>(FTYPE.OP:FBinOp)        => FTYPE.OP
    rule #t2aInstr<_>(ITYPE.OP:TestOp)        => ITYPE.OP
    rule #t2aInstr<_>(ITYPE.OP:IRelOp)        => ITYPE.OP
    rule #t2aInstr<_>(FTYPE.OP:FRelOp)        => FTYPE.OP
    rule #t2aInstr<_>(ATYPE.OP:CvtOp)         => ATYPE.OP
```

#### Block Instructions

There are several formats of block instructions, and the text-to-abstract transformation must be distributed over them.
At this point, all branching identifiers should have been resolved, so we can remove the id.

```k
    rule #t2aInstr<C>( block _OID:OptionalId TDS:TypeDecls IS end _OID') => #block(gatherTypes(result, TDS), #t2aInstrs<C>(IS), .Int)
    rule #t2aInstr<C>( loop  _OID:OptionalId TDS IS end _OID') => #loop(gatherTypes(result, TDS), #t2aInstrs<C>(IS), .Int)
    rule #t2aInstr<C>( if    _OID:OptionalId TDS IS else _OID':OptionalId IS' end _OID'') => #if(gatherTypes(result, TDS), #t2aInstrs<C>(IS), #t2aInstrs<C>(IS'), .Int)
```

#### KWasm Administrative Instructions

The following instructions are not part of the official Wasm text format.
They are currently supported in KWasm text files, but may be deprecated.

```k
    rule #t2aInstr<_C>(trap) => trap

    rule #t2aInstr<C>(#block(VT:VecType, IS:Instrs, BLOCKINFO)) => #block(VT, #t2aInstrs<C>(IS), BLOCKINFO)

    rule #t2aInstr<_>(init_local I V) => init_local I V
    rule #t2aInstr<_>(init_locals VS) => init_locals VS
```

#### List Functions

The following are helper functions.
They distribute the text-to-abstract functions above over lists.

```k
    syntax Stmts      ::= "#t2aStmts"      "<" Context ">" "(" Stmts      ")" [function]
    syntax Defns      ::= "#t2aDefns"      "<" Context ">" "(" Defns      ")" [function]
    syntax Instrs     ::= "#t2aInstrs"     "<" Context ">" "(" Instrs     ")" [function]
 // ------------------------------------------------------------------------------------
    rule #t2aStmts<C>(S:Stmt SS:Stmts) => #t2aStmt<C>(S) #t2aStmts<C>(SS)
    rule #t2aStmts<_>(.Stmts) => .Stmts

    rule #t2aDefns<C>(D:Defn DS:Defns) => #t2aDefn<C>(D) #t2aDefns<C>(DS)
    rule #t2aDefns<_>(.Defns) => .Defns

    rule #t2aInstrs<C>(I:Instr IS:Instrs) => #t2aInstr<C>(I) #t2aInstrs<C>(IS)
    rule #t2aInstrs<_>(.Instrs) => .Instrs
```

#### Functions for Gathering Context

The following are helper functions for gathering and updating context.

```k
    syntax Map ::= #idcTypes    ( Defns           ) [function]
                 | #idcTypesAux ( Defns, Int, Map ) [function]
 // ----------------------------------------------------------
    rule #idcTypes(DEFNS) => #idcTypesAux(DEFNS, 0, .Map)

    rule #idcTypesAux((type ID:Identifier (func _)) TS => TS, IDX => IDX +Int 1,  ACC => ACC [ ID <- IDX ]) requires notBool ID in_keys(ACC)
    rule #idcTypesAux((type               (func _)) TS => TS, IDX => IDX +Int 1, _ACC)
    rule #idcTypesAux(.Defns, _, ACC) => ACC

    syntax Map ::= #idcFuncs    ( Defns, Defns           ) [function]
                 | #idcFuncsAux ( Defns, Defns, Int, Map ) [function]
 // -----------------------------------------------------------------
    rule #idcFuncs(IMPORTS, DEFNS) => #idcFuncsAux(IMPORTS, DEFNS, 0, .Map)

    rule #idcFuncsAux((import _ _ (func ID:Identifier _)) IS => IS, _FS,  IDX => IDX +Int 1,  ACC => ACC [ ID <-IDX ]) requires notBool ID in_keys(ACC)
    rule #idcFuncsAux((import _ _ (func               _)) IS => IS, _FS,  IDX => IDX +Int 1, _ACC)
    rule #idcFuncsAux(_I                                  IS => IS, _FS, _IDX              , _ACC) [owise]

    rule #idcFuncsAux(.Defns, (func ID:Identifier _) FS => FS, IDX => IDX +Int 1,  ACC => ACC [ ID <- IDX ]) requires notBool ID in_keys(ACC)
    rule #idcFuncsAux(.Defns, (func      _:FuncSpec) FS => FS, IDX => IDX +Int 1, _ACC)
    rule #idcFuncsAux(.Defns, .Defns, _, ACC) => ACC

    syntax Map ::= #idcGlobals    ( Defns, Defns           ) [function]
                 | #idcGlobalsAux ( Defns, Defns, Int, Map ) [function]
 // -------------------------------------------------------------------
    rule #idcGlobals(IMPORTS, DEFNS) => #idcGlobalsAux(IMPORTS, DEFNS, 0, .Map)

    rule #idcGlobalsAux((import _ _ (global ID:Identifier _)) IS => IS, _GS,  IDX => IDX +Int 1,  ACC => ACC [ ID <-IDX ]) requires notBool ID in_keys(ACC)
    rule #idcGlobalsAux((import _ _ (global               _)) IS => IS, _GS,  IDX => IDX +Int 1, _ACC)
    rule #idcGlobalsAux(_I                                    IS => IS, _GS, _IDX              , _ACC) [owise]

    rule #idcGlobalsAux(.Defns, #global(... metadata: ID:Identifier) GS => GS, IDX => IDX +Int 1,  ACC => ACC [ ID <- IDX ]) requires notBool ID in_keys(ACC)
    rule #idcGlobalsAux(.Defns, #global(...) GS => GS, IDX => IDX +Int 1, _ACC) [owise]
    rule #idcGlobalsAux(.Defns, .Defns, _, ACC) => ACC

    syntax Map ::= #ids2Idxs(TypeUse, LocalDecls)      [function, total]
                 | #ids2Idxs(Int, TypeUse, LocalDecls) [function, total]
 // -------------------------------------------------------------------------
    rule #ids2Idxs(TU, LDS) => #ids2Idxs(0, TU, LDS)

    rule #ids2Idxs(_, .TypeDecls, .LocalDecls) => .Map
    rule #ids2Idxs(N, (type _)    , LDS) => #ids2Idxs(N, .TypeDecls, LDS)
    rule #ids2Idxs(N, (type _) TDS, LDS) => #ids2Idxs(N, TDS       , LDS)

    rule #ids2Idxs(N, (param ID:Identifier _) TDS, LDS)
      => (ID |-> N) #ids2Idxs(N +Int 1, TDS, LDS)
    rule #ids2Idxs(N,  (param _)   TDS, LDS) => #ids2Idxs(N +Int 1, TDS, LDS)
    rule #ids2Idxs(N, _TD:TypeDecl TDS, LDS) => #ids2Idxs(N       , TDS, LDS) [owise]

    rule #ids2Idxs(N, .TypeDecls, local ID:Identifier _ LDS:LocalDecls)
      => (ID |-> N) #ids2Idxs(N +Int 1, .TypeDecls, LDS)
    rule #ids2Idxs(N, .TypeDecls, _LD:LocalDecl LDS) => #ids2Idxs(N +Int 1, .TypeDecls, LDS) [owise]
```

```k
endmodule
```
