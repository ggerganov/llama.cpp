requires "kwasm-lemmas.md"

module MEMORY-SPEC
    imports KWASM-LEMMAS

    claim <instrs> i32.const ADDR ~> i32.const ADDR ~> #load(i64, load32_u, 0) ~> #store(i64, store16, 0) => . ... </instrs>
          <curModIdx> CUR </curModIdx>
          <moduleInst>
            <modIdx> CUR </modIdx>
            <memAddrs> 0 |-> MEMADDR </memAddrs>
            ...
          </moduleInst>
          <memInst>
            <mAddr> MEMADDR </mAddr>
            <msize> SIZE    </msize>
            <mdata> _BM      </mdata>
            ...
          </memInst>
        requires
         ADDR +Int #numBytes(i64) <=Int SIZE *Int #pageSize()
         andBool #inUnsignedRange(i32, ADDR)

    claim <instrs>  i32.const ADDR ~> i32.const ADDR ~> #load(ITYPE:IValType, load, 0) ~> #store(ITYPE, store, 0) => . ... </instrs>
          <curModIdx> CUR </curModIdx>
          <moduleInst>
            <modIdx> CUR </modIdx>
            <memAddrs> 0 |-> MEMADDR </memAddrs>
            ...
          </moduleInst>
          <memInst>
            <mAddr> MEMADDR </mAddr>
            <msize> SIZE    </msize>
            <mdata> _BM      </mdata>
            ...
          </memInst>
        requires
         #get(#chop(<i32> ADDR)) +Int #numBytes(ITYPE) <=Int SIZE *Int #pageSize()
endmodule
