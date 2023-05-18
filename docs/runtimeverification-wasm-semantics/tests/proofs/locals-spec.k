requires "kwasm-lemmas.md"

module LOCALS-SPEC
    imports KWASM-LEMMAS

    claim <instrs> #local.get(X) ~> #local.set(X) => . ... </instrs>
          <locals>
            X |-> < _ITYPE > (VAL => VAL)
          </locals>
endmodule
