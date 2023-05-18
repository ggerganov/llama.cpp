requires "kwasm-lemmas.md"

module FUNCTIONS-LEMMAS
    imports KWASM-LEMMAS

    syntax KItem ::= run ( Step ) | done ( Step )
 // ---------------------------------------------
    rule <instrs> run ( S ) => done ( S ) ... </instrs>

    syntax Step ::= Bytes | Int
 // ---------------------------
endmodule

module FUNCTIONS-SPEC
    imports FUNCTIONS-LEMMAS

    claim <instrs> run ( #wrap(2, #getRange(BM, ADDR, 4)) ) => done ( #getRange(BM, ADDR, 2) ) ... </instrs> requires 0 <=Int ADDR

    claim <instrs> run ( #setRange(BM, ADDR, #wrap(2, #getRange(BM, ADDR, 4)), 2) ) => done ( BM ) ... </instrs> requires 0 <=Int ADDR

    claim <instrs> run ( #getRange(_BM, _ADDR, 0)                          ) => done ( 0                             ) ... </instrs>
    claim <instrs> run ( #getRange(_BM, ADDR, _WIDTH)                      ) => done ( 0                             ) ... </instrs> requires ADDR <Int 0
    claim <instrs> run ( #getRange(BM, ADDR, 1) modInt 256                 ) => done ( #getRange(BM, ADDR       , 1) ) ... </instrs>
    claim <instrs> run ( #getRange(BM, ADDR, 3) modInt 256                 ) => done ( #getRange(BM, ADDR       , 1) ) ... </instrs>
    claim <instrs> run ( #getRange(BM, ADDR, 3) modInt (2 ^Int (8 *Int 3)) ) => done ( #getRange(BM, ADDR       , 3) ) ... </instrs>
    claim <instrs> run ( #getRange(BM, ADDR, 7) modInt (2 ^Int (8 *Int 3)) ) => done ( #getRange(BM, ADDR       , 3) ) ... </instrs>
    claim <instrs> run ( #getRange(BM, ADDR, 2) >>Int 8                    ) => done ( #getRange(BM, ADDR +Int 1, 1) ) ... </instrs> requires 0 <=Int ADDR

    claim <instrs> run  ( ( ( #getRange ( BM , ADDR +Int 1 , 1 ) modInt 256 ) +Int ( #getRange ( BM , ADDR , 1 ) modInt 256 <<Int 8 ) ) <<Int 48 )
                => done ( (   #getRange ( BM , ADDR +Int 1 , 1 )              +Int ( #getRange ( BM , ADDR , 1 )            <<Int 8 ) ) <<Int 48 )
               ...
          </instrs>

    claim <instrs> run  ( ( ( #getRange ( BM , ADDR +Int 1 , 7 ) modInt 256 ) +Int ( #getRange ( BM , ADDR , 8 ) modInt 256 <<Int 8 ) ) <<Int 48 )
                => done ( ( ( #getRange ( BM , ADDR +Int 1 , 1 )            ) +Int ( #getRange ( BM , ADDR , 1 )            <<Int 8 ) ) <<Int 48 )
               ...
          </instrs>

    claim <instrs> run  ( #setRange(BM, ADDR, #getRange(BM, ADDR +Int 1, 1) +Int (#getRange(BM, ADDR, 1) <<Int 8), 2) )
                => done ( #setRange(#setRange(BM, ADDR, #getRange(BM, ADDR +Int 1, 1), 1), ADDR +Int 1, #getRange(BM, ADDR, 1), 1) )
               ...
          </instrs>
       requires ADDR >=Int 0

    claim <instrs> run  ( #setRange ( BM , ADDR modInt 4294967296 , #getRange ( BM , ADDR modInt 4294967296 , #numBytes ( ITYPE ) ) , #numBytes ( ITYPE ) ) )
                => done ( BM )
               ...
          </instrs>

    claim <instrs> run  ( (#getRange(BM, ADDR +Int 1, 1) +Int (#getRange (BM, ADDR, 1) <<Int 8)) modInt 65536 )
                => done (  #getRange(BM, ADDR +Int 1, 1) +Int (#getRange (BM, ADDR, 1) <<Int 8) )
               ...
          </instrs>

    claim <instrs> run ( #getRange ( #setRange (  BM , ADDR +Int 7 , _VAL , 1 ) , ADDR +Int 3 , 1 ) ) => done ( #getRange ( BM , ADDR +Int 3 , 1 ) ) ... </instrs>

    claim <instrs> run ( #getRange ( #setRange ( _BM , ADDR +Int 7 ,  VAL , 1 ) , ADDR +Int 7 , 1 ) ) => done ( VAL ) ... </instrs> requires 0 <=Int ADDR andBool 0 <=Int VAL andBool VAL <Int 256
    claim <instrs> run  ( #getRange ( #setRange ( #setRange ( #setRange ( #setRange ( #setRange ( #setRange ( #setRange ( #setRange ( BM , ADDR , #getRange ( BM , ADDR +Int 7 , 1 ) , 1 ) , ADDR +Int 1 , #getRange ( BM , ADDR +Int 6 , 1 ) , 1 ) , ADDR +Int 2 , #getRange ( BM , ADDR +Int 5 , 1 ) , 1 ) , ADDR +Int 3 , #getRange ( BM , ADDR +Int 4 , 1 ) , 1 ) , ADDR +Int 4 , #getRange ( BM , ADDR +Int 3 , 1 ) , 1 ) , ADDR +Int 5 , #getRange ( BM , ADDR +Int 2 , 1 ) , 1 ) , ADDR +Int 6 , #getRange ( BM , ADDR +Int 1 , 1 ) , 1 ) , ADDR +Int 7 , #getRange ( BM , ADDR , 1 ) , 1 ) , ADDR +Int 3 , 1 ) )
                => done ( #getRange ( BM , ADDR +Int 4 , 1 ) )
               ...
          </instrs>
       requires 0 <=Int ADDR

endmodule
