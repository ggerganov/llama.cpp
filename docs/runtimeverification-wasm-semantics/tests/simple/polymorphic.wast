;; drop

(i32.const 15)
(drop)
#assertStack .ValStack "drop i32"

(i64.const 15)
(drop)
#assertStack .ValStack "drop i64"

(f32.const 15.0)
(drop)
#assertStack .ValStack "drop f32"

(f64.const 15.0)
(drop)
#assertStack .ValStack "drop f64"

(i32.const 5)
(drop (i32.const 1))
#assertTopStack < i32 > 5 "folded drop"

;; select

(i32.const -1)
(i32.const 1)
(i32.const 1)
(select)
#assertTopStack < i32 > -1 "select i32 true"

(i32.const -1)
(i32.const 1)
(i32.const 0)
(select)
#assertTopStack < i32 > 1 "select i32 false"

(i64.const -1)
(i64.const 1)
(i32.const 1)
(select)
#assertTopStack < i64 > -1 "select i64 true"

(i64.const -1)
(i64.const 1)
(i32.const 0)
(select)
#assertTopStack < i64 > 1 "select i64 false"

(select (i32.const 1) (i32.const 0) (i32.const 1))
#assertTopStack < i32 > 1 "folded select i32"

(select (i64.const 1) (i64.const 0) (i32.const 0))
#assertTopStack < i64 > 0 "folded select i64"

(select (unreachable) (i64.const -1) (i32.const 0))
#assertTrap                "select strict in first branch"

(select (i64.const 1) (unreachable) (i32.const 0))
#assertTrap               "select strict in second branch"
#assertTopStack < i64 > 1 "select strict in second branch"

(select (i64.const 1) (i64.const -1) (unreachable))
#assertTrap                "select strict in condition"
#assertTopStack < i64 > -1 "select strict in condition"
(drop)
#assertTopStack < i64 >  1 "select strict in condition"

#clearConfig
