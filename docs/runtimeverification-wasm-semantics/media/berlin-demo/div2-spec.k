requires "kwasm-lemmas.k"

module DIV2-SPEC
    imports WASM
    imports KWASM-LEMMAS

// Try to prove an incorrect property of unsigned division:
// If  X' = X / Y when Y =/= 0
// and X' = 0 otherwise,
// and X > 0,
// then X' < X

    rule <k> (if (local.get 1)
                (then (local.set 0
                  (i32.div_u (local.get 0) (local.get 1))))
              (else

                (local.set 0 (i32.const 0))))
             => .
           </k>
           <locals>
            0 |-> < i32 > (X => X')
            1 |-> < i32 > Y
           </locals>
           requires #inUnsignedRange(i32, X)
            andBool #inUnsignedRange(i32, Y)

            andBool X =/=Int 0
           ensures X' <Int X

// Errors case that will cause the proof to fail:
// * Y = 1

endmodule