(i32.const 0)
(i32.eqz)
#assertTopStack < i32 > 1 "eqz1"

(i32.const 3)
(i32.eqz)
#assertTopStack < i32 > 0 "eqz2"

(i32.eqz (i32.const 3))
#assertTopStack < i32 > 0 "eqz folded"

(i32.const 3)
(i32.const 3)
(i32.eq)
#assertTopStack < i32 > 1 "eq1"

(i32.const 3)
(i32.const 4)
(i32.eq)
#assertTopStack < i32 > 0 "eq2"

(i32.const 3)
(i32.const 3)
(i32.ne)
#assertTopStack < i32 > 0 "ne1"

(i32.const 3)
(i32.const 4)
(i32.ne)
#assertTopStack < i32 > 1 "ne2"

(i32.const 2)
(i32.const 32)
(i32.lt_u)
#assertTopStack < i32 > 1 "lt_u"

(i32.lt_u (i32.const 32) (i32.const 2))
#assertTopStack < i32 > 0 "lt_u"

(i32.const 2)
(i32.const 32)
(i32.gt_u)
#assertTopStack < i32 > 0 "gt_u"

(i32.const #pow1(i32) +Int 7)
(i32.const #pow1(i32) +Int 15)
(i32.lt_s)
#assertTopStack < i32 > 1 "lt_s 1"

(i32.const -32)
(i32.const 32)
(i32.lt_s)
#assertTopStack < i32 > 1 "lt_s 2"

(i32.const #pow1(i32) +Int 7)
(i32.const #pow1(i32) +Int 15)
(i32.gt_s)
#assertTopStack < i32 > 0 "gt_s 1"

(i32.const -32)
(i32.const 32)
(i32.gt_s)
#assertTopStack < i32 > 0 "gt_s 2"

(i32.const 2)
(i32.const 32)
(i32.le_u)
#assertTopStack < i32 > 1 "le_u 1"

(i32.const 32)
(i32.const 32)
(i32.le_u)
#assertTopStack < i32 > 1 "le_u 2"

(i32.const 2)
(i32.const 32)
(i32.ge_u)
#assertTopStack < i32 > 0 "ge_u 1"

(i32.const 32)
(i32.const 32)
(i32.ge_u)
#assertTopStack < i32 > 1 "ge_u 2"

(i32.const #pow1(i32) +Int 7)
(i32.const #pow1(i32) +Int 15)
(i32.le_s)
#assertTopStack < i32 > 1 "le_s 1"

(i32.const 32)
(i32.const 32)
(i32.le_s)
#assertTopStack < i32 > 1 "le_s 2"

(i32.const -32)
(i32.const 32)
(i32.le_s)
#assertTopStack < i32 > 1 "le_s 3"

(i32.const #pow1(i32) +Int 7)
(i32.const #pow1(i32) +Int 15)
(i32.ge_s)
#assertTopStack < i32 > 0 "ge_s 1"

(i32.const 32)
(i32.const 32)
(i32.ge_s)
#assertTopStack < i32 > 1 "ge_s 2"

(i32.const -32)
(i32.const 32)
(i32.ge_s)
#assertTopStack < i32 > 0 "ge_s 3"

#clearConfig
