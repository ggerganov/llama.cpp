(i32.const 5)
(i32.const 7)
(i32.add)
#assertTopStack < i32 > 12 "add"

(i32.const 5)
(i32.const 7)
(i32.sub)
#assertTopStack < i32 > -2 "sub"

(i32.const 15)
(i32.const 3)
(i32.mul)
#assertTopStack < i32 > 45 "mul"

(i32.const 15)
(i32.const 3)
(i32.div_u)
#assertTopStack < i32 > 5 "div_u1"

(i32.const 15)
(i32.const 2)
(i32.div_u)
#assertTopStack < i32 > 7 "div_u2"

(i32.const 15)
(i32.const 0)
(i32.div_u)
#assertTrap "div_u3"

(i32.const 15)
(i32.const 3)
(i32.rem_u)
#assertTopStack < i32 > 0 "rem_u1"

(i32.const 15)
(i32.const 2)
(i32.rem_u)
#assertTopStack < i32 > 1 "rem_u2"

(i32.const 15)
(i32.const 0)
(i32.rem_u)
#assertTrap "rem_u3"

(i32.const 10)
(i32.const 3)
(i32.div_s)
#assertTopStack < i32 > 3 "i32.div_s 1"

(i32.const 10)
(i32.const 4)
(i32.div_s)
#assertTopStack < i32 > 2 "i32.div_s 2"

(i32.const 10)
(i32.const 0)
(i32.div_s)
#assertTrap "i32.div_s 3"

(i32.const #pow1(i32))
(i32.const #pow(i32) -Int 1)
(i32.div_s)
#assertTrap "i32.div_s 4"

(i32.const 10)
(i32.const 5)
(i32.div_s)
#assertTopStack < i32 > 2 "div_s"

(i32.const 91)
(i32.const 13)
(i32.rem_s)
#assertTopStack <i32 > 0 "rem_s"

(i32.const -91)
(i32.const -13)
(i32.rem_s)
#assertTopStack <i32 > 0 "rem_s"

(i32.const -1)
(i32.const -3)
(i32.rem_s)
#assertTopStack <i32 > -1 "rem_s"

(i32.const 10)
(i32.const 0)
(i32.rem_s)
#assertTrap "rem_s"

(i32.const #pow1(i32))
(i32.const #pow(i32) -Int 1)
(i32.rem_s)
#assertTopStack <i32 > 0 "rem_s edge case"

;; The following tests were generated using the reference OCaml WASM interpreter.

(i32.const 10)
(i32.const 3)
(i32.rem_s)
#assertTopStack < i32 > 1 "i32.rem_s 1"

(i32.const 10)
(i32.const 4)
(i32.rem_s)
#assertTopStack < i32 > 2 "i32.rem_s 2"

(i32.const 10)
(i32.const 5)
(i32.rem_s)
#assertTopStack < i32 > 0 "i32.rem_s 3"

(i32.const -10)
(i32.const 3)
(i32.div_s)
#assertTopStack < i32 > -3 "i32.div_s 3"

(i32.const -10)
(i32.const 4)
(i32.div_s)
#assertTopStack < i32 > -2 "i32.div_s 4"

(i32.const -10)
(i32.const 5)
(i32.div_s)
#assertTopStack < i32 > -2 "i32.div_s 5"

(i32.const -10)
(i32.const 3)
(i32.rem_s)
#assertTopStack < i32 > -1 "i32.rem_s 4"

(i32.const -10)
(i32.const 4)
(i32.rem_s)
#assertTopStack < i32 > -2 "i32.rem_s 5"

(i32.const -10)
(i32.const 5)
(i32.rem_s)
#assertTopStack < i32 > 0 "i32.rem_s 6"

(i32.const -10)
(i32.const -3)
(i32.div_s)
#assertTopStack < i32 > 3 "i32.div_s 6"

(i32.const -10)
(i32.const -4)
(i32.div_s)
#assertTopStack < i32 > 2 "i32.div_s 7"

(i32.const -10)
(i32.const -5)
(i32.div_s)
#assertTopStack < i32 > 2 "i32.div_s 8"

(i32.const -10)
(i32.const -3)
(i32.rem_s)
#assertTopStack < i32 > -1 "i32.rem_s 7"

(i32.const -10)
(i32.const -4)
(i32.rem_s)
#assertTopStack < i32 > -2 "i32.rem_s 8"

(i32.const -10)
(i32.const -5)
(i32.rem_s)
#assertTopStack < i32 > 0 "i32.rem_s 9"

(i32.add (i32.const 3) (i32.const 4))
#assertTopStack < i32 > 7 "simple add folded"

(i32.sub (i32.const 3) (i32.const 4))
#assertTopStack < i32 > -1 "simple sub, order dependent folded"

(i32.sub (i32.mul (i32.const 5) (i32.const 7)) (i32.const 4))
#assertTopStack < i32 > 31 "mul nested in sub folded"

#clearConfig
