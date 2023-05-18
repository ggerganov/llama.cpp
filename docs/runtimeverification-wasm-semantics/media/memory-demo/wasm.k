module WASM
  imports DOMAINS

  // Solutions to the TODOs are in a comment at the bottom of the file.

  configuration
    <k color="green"> $PGM:Instrs </k>
    <stack color="red"> .Stack </stack>
    // TODO: Extend the configuration to have a current frame and and a store.


 // Useful definitions.

    syntax Instrs ::= List{Instr, ""}
 // ---------------------------------
    rule <k> .Instrs => . </k>
    rule <k> I IS:Instrs => I ~> IS ... </k>

    syntax Stack ::= ".Stack" | StackItem ":" Stack
    syntax StackItem ::= "<" IType ">" Int
 // --------------------------------------

    syntax IType ::= "i32" | "i64"
 // ------------------------------

 // Functions.

    syntax Int ::= #chop(IType, Int) [function]
 // -------------------------------------------
    rule #chop(i32, I) => I modInt (2 ^Int 32)
    rule #chop(i64, I) => I modInt (2 ^Int 64)


 // Basic instructions.
    syntax Instr ::= "(" IType "." "const" Int ")"
 // ----------------------------------------------
    rule <k> ( ITYPE . const I ) => . ... </k>
         <stack> S => < ITYPE > #chop(ITYPE, I) : S </stack>

 // Memory instructions.

    // Memory instantiation.
    syntax Instr ::= "(" "memory" ")"
 // ---------------------------------
    // TODO: Add a rule that:
    // allocates memory with size 0 to the store, and stores saves the address
    // to that memory in the current frame.

    syntax Instr ::= "(" "memory.size" ")"
 // --------------------------------------
    // TODO: Add a rule that:
    // Looks up the memory of the current frame in the store, and pushes its
    // size to the stack.

    syntax Instr ::= "(" "memory.grow" ")" | "(" "memory.grow" Instr ")"
 // --------------------------------------------------------------------
    // TODO: Add a rule that:
    // Unfolds the folded syntax rule.
    // TODO: Add a rule that:
    // Grows the current memory by a specified value (look at the top of the
    // stack). This rule should only apply if the resulting memory size is less
    // than 2^16 pages.
    // TODO: Add a rule that:
    // Only applies if we have a toggle set in the configuration, called
    // <spuriousMemoryFail>. If so, it just pushes (i32.const -1).

endmodule

















/*

  configuration
    <k color="green"> $PGM:Instrs </k>
    <stack color="red"> .Stack </stack>
    <frame>
       <module>
         <memAddrs color="Orchid"> .Map </memAddrs>
       </module>
    </frame>
    <store>
      <nextMemAddr> 0 </nextMemAddr>
      <mems color="orange">
        <memInst multiplicity="*" type="Map">
          <memAddr> 0 </memAddr>
          <memSize> 0 </memSize>
        </memInst>
      </mems>
    </store>
    <spuriousMemoryFail> false </spuriousMemoryFail>

    rule <k> (( memory ) ~> ELSE) => ELSE </k>
         <memAddrs> .Map => 0 |-> NEXT </memAddrs>
         <nextMemAddr> NEXT => NEXT +Int 1 </nextMemAddr>
         <mems>
           (.Bag =>
             <memInst>
               <memAddr> NEXT </memAddr>
               ...
             </memInst>)
           ...
         </mems>

    syntax Instr ::= "(" "memory.size" ")"
 // --------------------------------------
    rule <k> ( memory.size ) => ( i32.const SZ ) ... </k>
         <memAddrs> 0 |-> A </memAddrs>
         <memInst>
           <memAddr> A  </memAddr>
           <memSize> SZ </memSize>
           ...
         </memInst>
    rule <k> ( memory.size ) => ( i32.const -1 ) ... </k>

    syntax Instr ::= "(" "memory.grow" ")" | "(" "memory.grow" Instr ")"
 // --------------------------------------------------------------------
    rule <k> (memory.grow I:Instr) => I ~> (memory.grow) ... </k>
    rule <k> (memory.grow) => (i32.const SZ) ... </k>
         <stack> < i32 > V : S => S </stack>
         <memAddrs> 0 |-> A </memAddrs>
         <memInst>
           <memAddr> A </memAddr>
           <memSize> SZ => SZ +Int V </memSize>
         </memInst>
         requires SZ +Int V <Int 2 ^Int 16
    rule <k> (memory.grow) => (i32.const -1) ... </k>
         <stack> < i32 > _ : S => S </stack>
         <spuriousMemoryFail> true </spuriousMemoryFail>

*/