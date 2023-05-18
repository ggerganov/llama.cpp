module MEMORY-SPEC
  imports WASM

  rule <k> ( memory.size ) => (i32.const SZ) ... </k>
      <memAddrs> 0 |-> A </memAddrs>
      <memInst>
        <memAddr> A </memAddr>
        <memSize> SZ </memSize>
      </memInst>

  rule <k> (memory.grow (i32.const X)) => (i32.const SZ) ...</k>
      <memAddrs> 0 |-> A </memAddrs>
      <memInst>
        <memAddr> A </memAddr>
        <memSize> SZ => (SZ +Int X) </memSize>
      </memInst>
      // Necessary conditions. Remove them to see what happens!
     <spuriousMemoryFail> false </spuriousMemoryFail>
     requires X >=Int 0
      andBool X <=Int (2 ^Int 16)
      andBool SZ >=Int 0

  rule <k> (memory.grow (i32.const X)) => (i32.const -1) ...</k>
      <memAddrs> 0 |-> A </memAddrs>
      <memInst>
        <memAddr> A </memAddr>
        <memSize> SZ </memSize>
      </memInst>
      // Necessary conditions. Remove them to see what happens!
     <spuriousMemoryFail> true </spuriousMemoryFail>
     requires X >=Int (2 ^Int 16)
      andBool X  <Int (2 ^Int 32)
      andBool SZ >=Int 0

endmodule
