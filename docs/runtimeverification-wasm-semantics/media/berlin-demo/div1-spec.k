requires "kwasm-lemmas.k"

module DIV1-SPEC
    imports WASM
    imports KWASM-LEMMAS

// Try to prove an incorrect property of unsigned division:
// If X' = X / Y, then X' < X.

    rule <k> (local.set 0
               (i32.div_u (local.get 0) (local.get 1)))
            // Rewriting to "." means we expect the program to complete
            // without trapping.
             => .
           </k>
           <locals>
            0 |-> < i32 > (X => X')
            1 |-> < i32 > Y
           </locals>
           // We must specify that X and Y fulfill the invariant for locals to
           // be in the unsigned range. Otherwise, X and Y can be *any* integers.
           requires #inUnsignedRange(i32, X)
            andBool #inUnsignedRange(i32, Y)
           ensures X' <Int X

// Errors cases that will cause the proof to fail:
// * X = 0
// * Y = 0
// * Y = 1

endmodule
