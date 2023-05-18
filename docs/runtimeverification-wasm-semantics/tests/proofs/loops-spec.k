requires "kwasm-lemmas.md"

module LOOPS-SPEC
    imports KWASM-LEMMAS

    // Lemma
    claim <instrs> #br(0)
                ~> label // Loop label.
                     [ .ValTypes ]
                     { #loop([ .ValTypes ],
                         #local.get(0)
                         #local.get(1)
                         ITYPE.add
                         #local.set(1)
                         #local.get(0)
                         ITYPE.const 1
                         ITYPE.sub
                         #local.tee(0)
                         ITYPE.eqz
                         #br_if(1)
                         #br(0),
                         _
                       )
                    }
                    .ValStack
                ~> label [ .ValTypes ] {.Instrs } STACK // Block label.
                => .
                ...
          </instrs>
          <valstack> _ => STACK </valstack>
          <locals>
            0 |-> < ITYPE > (I => 0)
            1 |-> < ITYPE > (X => X +Int ((I *Int (I +Int 1)) /Int 2))
          </locals>
       requires #inUnsignedRange(ITYPE, I)
        andBool I >Int 0
        andBool #inUnsignedRange(ITYPE, X +Int I)
        andBool #inUnsignedRange(ITYPE, X +Int ((I *Int (I +Int 1)) /Int 2))

    // Main claim.
    claim <instrs>
               #block([ .ValTypes ],
                  #loop([ .ValTypes],
                      #local.get(0)
                      #local.get(1)
                      ITYPE.add
                      #local.set(1)
                      #local.get(0)
                      ITYPE.const 1
                      ITYPE.sub
                      #local.tee(0)
                      ITYPE.eqz
                      #br_if(1)
                      #br(0),
                      _
                 ),
                 _
              )
           => .
           ...
          </instrs>
          <locals>
            0 |-> < ITYPE > (N => 0)
            1 |-> < ITYPE > (0 => (N *Int (N +Int 1)) /Int 2)
          </locals>
       requires #inUnsignedRange(ITYPE, N)
        andBool N >Int 0
        andBool #inUnsignedRange(ITYPE, ((N *Int (N +Int 1)) /Int 2))

endmodule
