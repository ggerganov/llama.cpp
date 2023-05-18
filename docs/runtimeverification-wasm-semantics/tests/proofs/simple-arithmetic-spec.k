requires "kwasm-lemmas.md"

module SIMPLE-ARITHMETIC-SPEC
    imports KWASM-LEMMAS

    claim <instrs> ITYPE:IValType . const X:Int => . ... </instrs>
          <valstack> S:ValStack => < ITYPE > X : S </valstack>
       requires #inUnsignedRange(ITYPE, X)

    claim <instrs> ITYPE:IValType . const X:Int => . ... </instrs>
          <valstack> S:ValStack => < ITYPE > (X +Int #pow(ITYPE)) : S </valstack>
       requires (#minSigned(ITYPE) <=Int X) andBool (X <Int 0)

    claim <instrs> ITYPE:IValType . const X:Int ~> ITYPE . const Y:Int ~> ITYPE . add => . ... </instrs>
          <valstack> S:ValStack => < ITYPE > (X +Int Y) : S </valstack>
       requires 0 <=Int X andBool 0 <=Int Y
        andBool (X +Int Y) <Int #pow(ITYPE)
endmodule
