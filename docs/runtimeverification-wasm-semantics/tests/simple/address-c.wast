;; Load i32 data with different offset/align arguments

(module
  (memory 1)
  (data (i32.const 0) "abcdefghijklmnopqrstuvwxyz")

  (func (export "8u_good1") (param $i i32) (result i32)
    (i32.load8_u offset=0 (local.get $i))                   ;; 97 'a'
  )
  (func (export "8u_good2") (param $i i32) (result i32)
    (i32.load8_u align=1 (local.get $i))                    ;; 97 'a'
  )
  (func (export "8u_good3") (param $i i32) (result i32)
    (i32.load8_u offset=1 align=1 (local.get $i))           ;; 98 'b'
  )
  (func (export "8u_good4") (param $i i32) (result i32)
    (i32.load8_u offset=2 align=1 (local.get $i))           ;; 99 'c'
  )
  (func (export "8u_good5") (param $i i32) (result i32)
    (i32.load8_u offset=25 align=1 (local.get $i))          ;; 122 'z'
  )

  (func (export "8s_good1") (param $i i32) (result i32)
    (i32.load8_s offset=0 (local.get $i))                   ;; 97 'a'
  )
  (func (export "8s_good2") (param $i i32) (result i32)
    (i32.load8_s align=1 (local.get $i))                    ;; 97 'a'
  )
  (func (export "8s_good3") (param $i i32) (result i32)
    (i32.load8_s offset=1 align=1 (local.get $i))           ;; 98 'b'
  )
  (func (export "8s_good4") (param $i i32) (result i32)
    (i32.load8_s offset=2 align=1 (local.get $i))           ;; 99 'c'
  )
  (func (export "8s_good5") (param $i i32) (result i32)
    (i32.load8_s offset=25 align=1 (local.get $i))          ;; 122 'z'
  )

  (func (export "16u_good1") (param $i i32) (result i32)
    (i32.load16_u offset=0 (local.get $i))                  ;; 25185 'ab'
  )
  (func (export "16u_good2") (param $i i32) (result i32)
    (i32.load16_u align=1 (local.get $i))                   ;; 25185 'ab'
  )
  (func (export "16u_good3") (param $i i32) (result i32)
    (i32.load16_u offset=1 align=1 (local.get $i))          ;; 25442 'bc'
  )
  (func (export "16u_good4") (param $i i32) (result i32)
    (i32.load16_u offset=2 align=2 (local.get $i))          ;; 25699 'cd'
  )
  (func (export "16u_good5") (param $i i32) (result i32)
    (i32.load16_u offset=25 align=2 (local.get $i))         ;; 122 'z\0'
  )

  (func (export "16s_good1") (param $i i32) (result i32)
    (i32.load16_s offset=0 (local.get $i))                  ;; 25185 'ab'
  )
  (func (export "16s_good2") (param $i i32) (result i32)
    (i32.load16_s align=1 (local.get $i))                   ;; 25185 'ab'
  )
  (func (export "16s_good3") (param $i i32) (result i32)
    (i32.load16_s offset=1 align=1 (local.get $i))          ;; 25442 'bc'
  )
  (func (export "16s_good4") (param $i i32) (result i32)
    (i32.load16_s offset=2 align=2 (local.get $i))          ;; 25699 'cd'
  )
  (func (export "16s_good5") (param $i i32) (result i32)
    (i32.load16_s offset=25 align=2 (local.get $i))         ;; 122 'z\0'
  )

  (func (export "32_good1") (param $i i32) (result i32)
    (i32.load offset=0 (local.get $i))                      ;; 1684234849 'abcd'
  )
  (func (export "32_good2") (param $i i32) (result i32)
    (i32.load align=1 (local.get $i))                       ;; 1684234849 'abcd'
  )
  (func (export "32_good3") (param $i i32) (result i32)
    (i32.load offset=1 align=1 (local.get $i))              ;; 1701077858 'bcde'
  )
  (func (export "32_good4") (param $i i32) (result i32)
    (i32.load offset=2 align=2 (local.get $i))              ;; 1717920867 'cdef'
  )
  (func (export "32_good5") (param $i i32) (result i32)
    (i32.load offset=25 align=4 (local.get $i))             ;; 122 'z\0\0\0'
  )

  (func (export "8u_bad") (param $i i32)
    (drop (i32.load8_u offset=4294967295 (local.get $i)))
  )
  (func (export "8s_bad") (param $i i32)
    (drop (i32.load8_s offset=4294967295 (local.get $i)))
  )
  (func (export "16u_bad") (param $i i32)
    (drop (i32.load16_u offset=4294967295 (local.get $i)))
  )
  (func (export "16s_bad") (param $i i32)
    (drop (i32.load16_s offset=4294967295 (local.get $i)))
  )
  (func (export "32_bad") (param $i i32)
    (drop (i32.load offset=4294967295 (local.get $i)))
  )
)

(assert_return (invoke "8u_good1" (i32.const 0)) (i32.const 97))
(assert_return (invoke "8u_good2" (i32.const 0)) (i32.const 97))
(assert_return (invoke "8u_good3" (i32.const 0)) (i32.const 98))
(assert_return (invoke "8u_good4" (i32.const 0)) (i32.const 99))
(assert_return (invoke "8u_good5" (i32.const 0)) (i32.const 122))

(assert_return (invoke "8s_good1" (i32.const 0)) (i32.const 97))
(assert_return (invoke "8s_good2" (i32.const 0)) (i32.const 97))
(assert_return (invoke "8s_good3" (i32.const 0)) (i32.const 98))
(assert_return (invoke "8s_good4" (i32.const 0)) (i32.const 99))
(assert_return (invoke "8s_good5" (i32.const 0)) (i32.const 122))

(assert_return (invoke "16u_good1" (i32.const 0)) (i32.const 25185))
(assert_return (invoke "16u_good2" (i32.const 0)) (i32.const 25185))
(assert_return (invoke "16u_good3" (i32.const 0)) (i32.const 25442))
(assert_return (invoke "16u_good4" (i32.const 0)) (i32.const 25699))
(assert_return (invoke "16u_good5" (i32.const 0)) (i32.const 122))

(assert_return (invoke "16s_good1" (i32.const 0)) (i32.const 25185))
(assert_return (invoke "16s_good2" (i32.const 0)) (i32.const 25185))
(assert_return (invoke "16s_good3" (i32.const 0)) (i32.const 25442))
(assert_return (invoke "16s_good4" (i32.const 0)) (i32.const 25699))
(assert_return (invoke "16s_good5" (i32.const 0)) (i32.const 122))

(assert_return (invoke "32_good1" (i32.const 0)) (i32.const 1684234849))
(assert_return (invoke "32_good2" (i32.const 0)) (i32.const 1684234849))
(assert_return (invoke "32_good3" (i32.const 0)) (i32.const 1701077858))
(assert_return (invoke "32_good4" (i32.const 0)) (i32.const 1717920867))
(assert_return (invoke "32_good5" (i32.const 0)) (i32.const 122))

(assert_return (invoke "8u_good1" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "8u_good2" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "8u_good3" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "8u_good4" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "8u_good5" (i32.const 65507)) (i32.const 0))

(assert_return (invoke "8s_good1" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "8s_good2" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "8s_good3" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "8s_good4" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "8s_good5" (i32.const 65507)) (i32.const 0))

(assert_return (invoke "16u_good1" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "16u_good2" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "16u_good3" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "16u_good4" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "16u_good5" (i32.const 65507)) (i32.const 0))

(assert_return (invoke "16s_good1" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "16s_good2" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "16s_good3" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "16s_good4" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "16s_good5" (i32.const 65507)) (i32.const 0))

(assert_return (invoke "32_good1" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "32_good2" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "32_good3" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "32_good4" (i32.const 65507)) (i32.const 0))
(assert_return (invoke "32_good5" (i32.const 65507)) (i32.const 0))

(assert_return (invoke "8u_good1" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "8u_good2" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "8u_good3" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "8u_good4" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "8u_good5" (i32.const 65508)) (i32.const 0))

(assert_return (invoke "8s_good1" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "8s_good2" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "8s_good3" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "8s_good4" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "8s_good5" (i32.const 65508)) (i32.const 0))

(assert_return (invoke "16u_good1" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "16u_good2" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "16u_good3" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "16u_good4" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "16u_good5" (i32.const 65508)) (i32.const 0))

(assert_return (invoke "16s_good1" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "16s_good2" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "16s_good3" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "16s_good4" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "16s_good5" (i32.const 65508)) (i32.const 0))

(assert_return (invoke "32_good1" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "32_good2" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "32_good3" (i32.const 65508)) (i32.const 0))
(assert_return (invoke "32_good4" (i32.const 65508)) (i32.const 0))
(assert_trap (invoke "32_good5" (i32.const 65508)) "out of bounds memory access")

(assert_trap (invoke "8u_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "8s_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "16u_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "16s_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "32_bad" (i32.const 0)) "out of bounds memory access")

(assert_trap (invoke "8u_bad" (i32.const 1)) "out of bounds memory access")
(assert_trap (invoke "8s_bad" (i32.const 1)) "out of bounds memory access")
(assert_trap (invoke "16u_bad" (i32.const 1)) "out of bounds memory access")
(assert_trap (invoke "16s_bad" (i32.const 1)) "out of bounds memory access")
(assert_trap (invoke "32_bad" (i32.const 1)) "out of bounds memory access")

(assert_malformed
  (module quote
    "(memory 1)"
    "(func (drop (i32.load offset=4294967296 (i32.const 0))))"
  )
  "i32 constant"
)

;; Load i64 data with different offset/align arguments

(module
  (memory 1)
  (data (i32.const 0) "abcdefghijklmnopqrstuvwxyz")

  (func (export "8u_good1") (param $i i32) (result i64)
    (i64.load8_u offset=0 (local.get $i))                   ;; 97 'a'
  )
  (func (export "8u_good2") (param $i i32) (result i64)
    (i64.load8_u align=1 (local.get $i))                    ;; 97 'a'
  )
  (func (export "8u_good3") (param $i i32) (result i64)
    (i64.load8_u offset=1 align=1 (local.get $i))           ;; 98 'b'
  )
  (func (export "8u_good4") (param $i i32) (result i64)
    (i64.load8_u offset=2 align=1 (local.get $i))           ;; 99 'c'
  )
  (func (export "8u_good5") (param $i i32) (result i64)
    (i64.load8_u offset=25 align=1 (local.get $i))          ;; 122 'z'
  )

  (func (export "8s_good1") (param $i i32) (result i64)
    (i64.load8_s offset=0 (local.get $i))                   ;; 97 'a'
  )
  (func (export "8s_good2") (param $i i32) (result i64)
    (i64.load8_s align=1 (local.get $i))                    ;; 97 'a'
  )
  (func (export "8s_good3") (param $i i32) (result i64)
    (i64.load8_s offset=1 align=1 (local.get $i))           ;; 98 'b'
  )
  (func (export "8s_good4") (param $i i32) (result i64)
    (i64.load8_s offset=2 align=1 (local.get $i))           ;; 99 'c'
  )
  (func (export "8s_good5") (param $i i32) (result i64)
    (i64.load8_s offset=25 align=1 (local.get $i))          ;; 122 'z'
  )

  (func (export "16u_good1") (param $i i32) (result i64)
    (i64.load16_u offset=0 (local.get $i))                 ;; 25185 'ab'
  )
  (func (export "16u_good2") (param $i i32) (result i64)
    (i64.load16_u align=1 (local.get $i))                  ;; 25185 'ab'
  )
  (func (export "16u_good3") (param $i i32) (result i64)
    (i64.load16_u offset=1 align=1 (local.get $i))         ;; 25442 'bc'
  )
  (func (export "16u_good4") (param $i i32) (result i64)
    (i64.load16_u offset=2 align=2 (local.get $i))         ;; 25699 'cd'
  )
  (func (export "16u_good5") (param $i i32) (result i64)
    (i64.load16_u offset=25 align=2 (local.get $i))        ;; 122 'z\0'
  )

  (func (export "16s_good1") (param $i i32) (result i64)
    (i64.load16_s offset=0 (local.get $i))                 ;; 25185 'ab'
  )
  (func (export "16s_good2") (param $i i32) (result i64)
    (i64.load16_s align=1 (local.get $i))                  ;; 25185 'ab'
  )
  (func (export "16s_good3") (param $i i32) (result i64)
    (i64.load16_s offset=1 align=1 (local.get $i))         ;; 25442 'bc'
  )
  (func (export "16s_good4") (param $i i32) (result i64)
    (i64.load16_s offset=2 align=2 (local.get $i))         ;; 25699 'cd'
  )
  (func (export "16s_good5") (param $i i32) (result i64)
    (i64.load16_s offset=25 align=2 (local.get $i))        ;; 122 'z\0'
  )

  (func (export "32u_good1") (param $i i32) (result i64)
    (i64.load32_u offset=0 (local.get $i))                 ;; 1684234849 'abcd'
  )
  (func (export "32u_good2") (param $i i32) (result i64)
    (i64.load32_u align=1 (local.get $i))                  ;; 1684234849 'abcd'
  )
  (func (export "32u_good3") (param $i i32) (result i64)
    (i64.load32_u offset=1 align=1 (local.get $i))         ;; 1701077858 'bcde'
  )
  (func (export "32u_good4") (param $i i32) (result i64)
    (i64.load32_u offset=2 align=2 (local.get $i))         ;; 1717920867 'cdef'
  )
  (func (export "32u_good5") (param $i i32) (result i64)
    (i64.load32_u offset=25 align=4 (local.get $i))        ;; 122 'z\0\0\0'
  )

  (func (export "32s_good1") (param $i i32) (result i64)
    (i64.load32_s offset=0 (local.get $i))                 ;; 1684234849 'abcd'
  )
  (func (export "32s_good2") (param $i i32) (result i64)
    (i64.load32_s align=1 (local.get $i))                  ;; 1684234849 'abcd'
  )
  (func (export "32s_good3") (param $i i32) (result i64)
    (i64.load32_s offset=1 align=1 (local.get $i))         ;; 1701077858 'bcde'
  )
  (func (export "32s_good4") (param $i i32) (result i64)
    (i64.load32_s offset=2 align=2 (local.get $i))         ;; 1717920867 'cdef'
  )
  (func (export "32s_good5") (param $i i32) (result i64)
    (i64.load32_s offset=25 align=4 (local.get $i))        ;; 122 'z\0\0\0'
  )

  (func (export "64_good1") (param $i i32) (result i64)
    (i64.load offset=0 (local.get $i))                     ;; 0x6867666564636261 'abcdefgh'
  )
  (func (export "64_good2") (param $i i32) (result i64)
    (i64.load align=1 (local.get $i))                      ;; 0x6867666564636261 'abcdefgh'
  )
  (func (export "64_good3") (param $i i32) (result i64)
    (i64.load offset=1 align=1 (local.get $i))             ;; 0x6968676665646362 'bcdefghi'
  )
  (func (export "64_good4") (param $i i32) (result i64)
    (i64.load offset=2 align=2 (local.get $i))             ;; 0x6a69686766656463 'cdefghij'
  )
  (func (export "64_good5") (param $i i32) (result i64)
    (i64.load offset=25 align=8 (local.get $i))            ;; 122 'z\0\0\0\0\0\0\0'
  )

  (func (export "8u_bad") (param $i i32)
    (drop (i64.load8_u offset=4294967295 (local.get $i)))
  )
  (func (export "8s_bad") (param $i i32)
    (drop (i64.load8_s offset=4294967295 (local.get $i)))
  )
  (func (export "16u_bad") (param $i i32)
    (drop (i64.load16_u offset=4294967295 (local.get $i)))
  )
  (func (export "16s_bad") (param $i i32)
    (drop (i64.load16_s offset=4294967295 (local.get $i)))
  )
  (func (export "32u_bad") (param $i i32)
    (drop (i64.load32_u offset=4294967295 (local.get $i)))
  )
  (func (export "32s_bad") (param $i i32)
    (drop (i64.load32_s offset=4294967295 (local.get $i)))
  )
  (func (export "64_bad") (param $i i32)
    (drop (i64.load offset=4294967295 (local.get $i)))
  )
)

(assert_return (invoke "8u_good1" (i32.const 0)) (i64.const 97))
(assert_return (invoke "8u_good2" (i32.const 0)) (i64.const 97))
(assert_return (invoke "8u_good3" (i32.const 0)) (i64.const 98))
(assert_return (invoke "8u_good4" (i32.const 0)) (i64.const 99))
(assert_return (invoke "8u_good5" (i32.const 0)) (i64.const 122))

(assert_return (invoke "8s_good1" (i32.const 0)) (i64.const 97))
(assert_return (invoke "8s_good2" (i32.const 0)) (i64.const 97))
(assert_return (invoke "8s_good3" (i32.const 0)) (i64.const 98))
(assert_return (invoke "8s_good4" (i32.const 0)) (i64.const 99))
(assert_return (invoke "8s_good5" (i32.const 0)) (i64.const 122))

(assert_return (invoke "16u_good1" (i32.const 0)) (i64.const 25185))
(assert_return (invoke "16u_good2" (i32.const 0)) (i64.const 25185))
(assert_return (invoke "16u_good3" (i32.const 0)) (i64.const 25442))
(assert_return (invoke "16u_good4" (i32.const 0)) (i64.const 25699))
(assert_return (invoke "16u_good5" (i32.const 0)) (i64.const 122))

(assert_return (invoke "16s_good1" (i32.const 0)) (i64.const 25185))
(assert_return (invoke "16s_good2" (i32.const 0)) (i64.const 25185))
(assert_return (invoke "16s_good3" (i32.const 0)) (i64.const 25442))
(assert_return (invoke "16s_good4" (i32.const 0)) (i64.const 25699))
(assert_return (invoke "16s_good5" (i32.const 0)) (i64.const 122))

(assert_return (invoke "32u_good1" (i32.const 0)) (i64.const 1684234849))
(assert_return (invoke "32u_good2" (i32.const 0)) (i64.const 1684234849))
(assert_return (invoke "32u_good3" (i32.const 0)) (i64.const 1701077858))
(assert_return (invoke "32u_good4" (i32.const 0)) (i64.const 1717920867))
(assert_return (invoke "32u_good5" (i32.const 0)) (i64.const 122))

(assert_return (invoke "32s_good1" (i32.const 0)) (i64.const 1684234849))
(assert_return (invoke "32s_good2" (i32.const 0)) (i64.const 1684234849))
(assert_return (invoke "32s_good3" (i32.const 0)) (i64.const 1701077858))
(assert_return (invoke "32s_good4" (i32.const 0)) (i64.const 1717920867))
(assert_return (invoke "32s_good5" (i32.const 0)) (i64.const 122))

(assert_return (invoke "64_good1" (i32.const 0)) (i64.const 0x6867666564636261))
(assert_return (invoke "64_good2" (i32.const 0)) (i64.const 0x6867666564636261))
(assert_return (invoke "64_good3" (i32.const 0)) (i64.const 0x6968676665646362))
(assert_return (invoke "64_good4" (i32.const 0)) (i64.const 0x6a69686766656463))
(assert_return (invoke "64_good5" (i32.const 0)) (i64.const 122))

(assert_return (invoke "8u_good1" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "8u_good2" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "8u_good3" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "8u_good4" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "8u_good5" (i32.const 65503)) (i64.const 0))

(assert_return (invoke "8s_good1" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "8s_good2" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "8s_good3" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "8s_good4" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "8s_good5" (i32.const 65503)) (i64.const 0))

(assert_return (invoke "16u_good1" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "16u_good2" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "16u_good3" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "16u_good4" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "16u_good5" (i32.const 65503)) (i64.const 0))

(assert_return (invoke "16s_good1" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "16s_good2" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "16s_good3" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "16s_good4" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "16s_good5" (i32.const 65503)) (i64.const 0))

(assert_return (invoke "32u_good1" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "32u_good2" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "32u_good3" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "32u_good4" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "32u_good5" (i32.const 65503)) (i64.const 0))

(assert_return (invoke "32s_good1" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "32s_good2" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "32s_good3" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "32s_good4" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "32s_good5" (i32.const 65503)) (i64.const 0))

(assert_return (invoke "64_good1" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "64_good2" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "64_good3" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "64_good4" (i32.const 65503)) (i64.const 0))
(assert_return (invoke "64_good5" (i32.const 65503)) (i64.const 0))

(assert_return (invoke "8u_good1" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "8u_good2" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "8u_good3" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "8u_good4" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "8u_good5" (i32.const 65504)) (i64.const 0))

(assert_return (invoke "8s_good1" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "8s_good2" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "8s_good3" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "8s_good4" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "8s_good5" (i32.const 65504)) (i64.const 0))

(assert_return (invoke "16u_good1" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "16u_good2" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "16u_good3" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "16u_good4" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "16u_good5" (i32.const 65504)) (i64.const 0))

(assert_return (invoke "16s_good1" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "16s_good2" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "16s_good3" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "16s_good4" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "16s_good5" (i32.const 65504)) (i64.const 0))

(assert_return (invoke "32u_good1" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "32u_good2" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "32u_good3" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "32u_good4" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "32u_good5" (i32.const 65504)) (i64.const 0))

(assert_return (invoke "32s_good1" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "32s_good2" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "32s_good3" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "32s_good4" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "32s_good5" (i32.const 65504)) (i64.const 0))

(assert_return (invoke "64_good1" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "64_good2" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "64_good3" (i32.const 65504)) (i64.const 0))
(assert_return (invoke "64_good4" (i32.const 65504)) (i64.const 0))
(assert_trap (invoke "64_good5" (i32.const 65504)) "out of bounds memory access")

(assert_trap (invoke "8u_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "8s_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "16u_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "16s_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "32u_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "32s_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "64_bad" (i32.const 0)) "out of bounds memory access")

(assert_trap (invoke "8u_bad" (i32.const 1)) "out of bounds memory access")
(assert_trap (invoke "8s_bad" (i32.const 1)) "out of bounds memory access")
(assert_trap (invoke "16u_bad" (i32.const 1)) "out of bounds memory access")
(assert_trap (invoke "16s_bad" (i32.const 1)) "out of bounds memory access")
(assert_trap (invoke "32u_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "32s_bad" (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "64_bad" (i32.const 1)) "out of bounds memory access")

;; Load f32 data with different offset/align arguments

;; (module
;;   (memory 1)
;;   (data (i32.const 0) "\00\00\00\00\00\00\a0\7f\01\00\d0\7f")

;;   (func (export "32_good1") (param $i i32) (result f32)
;;     (f32.load offset=0 (local.get $i))                   ;; 0.0 '\00\00\00\00'
;;   )
;;   (func (export "32_good2") (param $i i32) (result f32)
;;     (f32.load align=1 (local.get $i))                    ;; 0.0 '\00\00\00\00'
;;   )
;;   (func (export "32_good3") (param $i i32) (result f32)
;;     (f32.load offset=1 align=1 (local.get $i))           ;; 0.0 '\00\00\00\00'
;;   )
;;   (func (export "32_good4") (param $i i32) (result f32)
;;     (f32.load offset=2 align=2 (local.get $i))           ;; 0.0 '\00\00\00\00'
;;   )
;;   (func (export "32_good5") (param $i i32) (result f32)
;;     (f32.load offset=8 align=4 (local.get $i))           ;; nan:0x500001 '\01\00\d0\7f'
;;   )
;;   (func (export "32_bad") (param $i i32)
;;     (drop (f32.load offset=4294967295 (local.get $i)))
;;   )
;; )

;; (assert_return (invoke "32_good1" (i32.const 0)) (f32.const 0.0))
;; (assert_return (invoke "32_good2" (i32.const 0)) (f32.const 0.0))
;; (assert_return (invoke "32_good3" (i32.const 0)) (f32.const 0.0))
;; (assert_return (invoke "32_good4" (i32.const 0)) (f32.const 0.0))
;; (assert_return (invoke "32_good5" (i32.const 0)) (f32.const nan:0x500001))

;; (assert_return (invoke "32_good1" (i32.const 65524)) (f32.const 0.0))
;; (assert_return (invoke "32_good2" (i32.const 65524)) (f32.const 0.0))
;; (assert_return (invoke "32_good3" (i32.const 65524)) (f32.const 0.0))
;; (assert_return (invoke "32_good4" (i32.const 65524)) (f32.const 0.0))
;; (assert_return (invoke "32_good5" (i32.const 65524)) (f32.const 0.0))

;; (assert_return (invoke "32_good1" (i32.const 65525)) (f32.const 0.0))
;; (assert_return (invoke "32_good2" (i32.const 65525)) (f32.const 0.0))
;; (assert_return (invoke "32_good3" (i32.const 65525)) (f32.const 0.0))
;; (assert_return (invoke "32_good4" (i32.const 65525)) (f32.const 0.0))
;; (assert_trap (invoke "32_good5" (i32.const 65525)) "out of bounds memory access")

;; (assert_trap (invoke "32_bad" (i32.const 0)) "out of bounds memory access")
;; (assert_trap (invoke "32_bad" (i32.const 1)) "out of bounds memory access")

;; ;; Load f64 data with different offset/align arguments

;; (module
;;   (memory 1)
;;   (data (i32.const 0) "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\f4\7f\01\00\00\00\00\00\fc\7f")

;;   (func (export "64_good1") (param $i i32) (result f64)
;;     (f64.load offset=0 (local.get $i))                     ;; 0.0 '\00\00\00\00\00\00\00\00'
;;   )
;;   (func (export "64_good2") (param $i i32) (result f64)
;;     (f64.load align=1 (local.get $i))                      ;; 0.0 '\00\00\00\00\00\00\00\00'
;;   )
;;   (func (export "64_good3") (param $i i32) (result f64)
;;     (f64.load offset=1 align=1 (local.get $i))             ;; 0.0 '\00\00\00\00\00\00\00\00'
;;   )
;;   (func (export "64_good4") (param $i i32) (result f64)
;;     (f64.load offset=2 align=2 (local.get $i))             ;; 0.0 '\00\00\00\00\00\00\00\00'
;;   )
;;   (func (export "64_good5") (param $i i32) (result f64)
;;     (f64.load offset=18 align=8 (local.get $i))            ;; nan:0xc000000000001 '\01\00\00\00\00\00\fc\7f'
;;   )
;;   (func (export "64_bad") (param $i i32)
;;     (drop (f64.load offset=4294967295 (local.get $i)))
;;   )
;; )

;; (assert_return (invoke "64_good1" (i32.const 0)) (f64.const 0.0))
;; (assert_return (invoke "64_good2" (i32.const 0)) (f64.const 0.0))
;; (assert_return (invoke "64_good3" (i32.const 0)) (f64.const 0.0))
;; (assert_return (invoke "64_good4" (i32.const 0)) (f64.const 0.0))
;; (assert_return (invoke "64_good5" (i32.const 0)) (f64.const nan:0xc000000000001))

;; (assert_return (invoke "64_good1" (i32.const 65510)) (f64.const 0.0))
;; (assert_return (invoke "64_good2" (i32.const 65510)) (f64.const 0.0))
;; (assert_return (invoke "64_good3" (i32.const 65510)) (f64.const 0.0))
;; (assert_return (invoke "64_good4" (i32.const 65510)) (f64.const 0.0))
;; (assert_return (invoke "64_good5" (i32.const 65510)) (f64.const 0.0))

;; (assert_return (invoke "64_good1" (i32.const 65511)) (f64.const 0.0))
;; (assert_return (invoke "64_good2" (i32.const 65511)) (f64.const 0.0))
;; (assert_return (invoke "64_good3" (i32.const 65511)) (f64.const 0.0))
;; (assert_return (invoke "64_good4" (i32.const 65511)) (f64.const 0.0))
;; (assert_trap (invoke "64_good5" (i32.const 65511)) "out of bounds memory access")

;; (assert_trap (invoke "64_bad" (i32.const 0)) "out of bounds memory access")
;; (assert_trap (invoke "64_bad" (i32.const 1)) "out of bounds memory access")

#clearConfig
