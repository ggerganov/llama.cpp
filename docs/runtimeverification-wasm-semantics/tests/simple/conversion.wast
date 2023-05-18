;; Wrap.

(i64.const 4294967296)    ;; 2^32
(i32.wrap_i64)
#assertTopStack < i32 > 0 "wrap 2^32"

(i64.const 4294967295)    ;; 2^32 - 1
(i32.wrap_i64)
#assertTopStack < i32 > 4294967295 "wrap 2^32 - 1"

(i32.wrap_i64 (i64.const 4294967298))
#assertTopStack < i32 > 2 "folded wrap 2^32 + 2"

;; Extend.

(i32.const 4294967295)    ;; 2^32 - 1
(i64.extend_i32_u)
#assertTopStack < i64 > 4294967295 "extend unsig"

(i32.const -1)    ;; 2^32 - 1
(i64.extend_i32_s)
#assertTopStack < i64 > -1 "extend sig"

(i64.extend_i32_s (i32.const 15))
#assertTopStack < i64 > 15 "folded extend sig"

(module
  (func (export "i64.extend_i32_s") (param $x i32) (result i64) (i64.extend_i32_s (local.get $x)))
  (func (export "i64.extend_i32_u") (param $x i32) (result i64) (i64.extend_i32_u (local.get $x)))
  (func (export "i32.wrap_i64") (param $x i64) (result i32) (i32.wrap_i64 (local.get $x)))
)

(assert_return (invoke "i64.extend_i32_s" (i32.const 0)) (i64.const 0))
(assert_return (invoke "i64.extend_i32_s" (i32.const 10000)) (i64.const 10000))
(assert_return (invoke "i64.extend_i32_s" (i32.const -10000)) (i64.const -10000))
(assert_return (invoke "i64.extend_i32_s" (i32.const -1)) (i64.const -1))
(assert_return (invoke "i64.extend_i32_s" (i32.const 0x7fffffff)) (i64.const 0x000000007fffffff))
(assert_return (invoke "i64.extend_i32_s" (i32.const 0x80000000)) (i64.const 0xffffffff80000000))

(assert_return (invoke "i64.extend_i32_u" (i32.const 0)) (i64.const 0))
(assert_return (invoke "i64.extend_i32_u" (i32.const 10000)) (i64.const 10000))
(assert_return (invoke "i64.extend_i32_u" (i32.const -10000)) (i64.const 0x00000000ffffd8f0))
(assert_return (invoke "i64.extend_i32_u" (i32.const -1)) (i64.const 0xffffffff))
(assert_return (invoke "i64.extend_i32_u" (i32.const 0x7fffffff)) (i64.const 0x000000007fffffff))
(assert_return (invoke "i64.extend_i32_u" (i32.const 0x80000000)) (i64.const 0x0000000080000000))

#clearConfig
