requires "kwasm-lemmas.k"

module DIV3-SPEC
    imports WASM
    imports KWASM-LEMMAS

// Prove a correct property of unsigned division:
// If  X' = X / Y when Y =/= 0
// and X' = 0 otherwise,
// and X > 0,
// and Y > 1,
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
            andBool Y =/=Int 1
            andBool X =/=Int 0
           ensures X' <Int X
endmodule