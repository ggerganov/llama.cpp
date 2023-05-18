;; Integers
;; --------

(i32.const 3)
#assertTopStack < i32 > 3 "i32 1"

(i32.const 5)
#assertTopStack < i32 > 5 "i32 parens"

(i64.const 71)
#assertTopStack < i64 > 71 "i64"

(i32.const #unsigned(i32, -5))
#assertTopStack < i32 > #pow(i32) -Int 5 "i32 manual unsigned"

(i32.const #pow(i32) -Int 5)
#assertTopStack < i32 > -5 "i32 manual unsigned"

(i32.const -5)
#assertTopStack < i32 > #unsigned(i32, -5) "i32 signed constant"

(i32.const #unsigned(i32, -5))
#assertTopStack < i32 > -5 "i32 signed assert"

(i32.const #pow(i32) +Int 1)
#assertTopStack < i32 > 1 "i32 overflow"

(i32.const -1)
#assertTopStackExactly < i32 > #pow(i32) -Int 1 "i32 overflow"

(i64.const -1)
#assertTopStackExactly < i64 > #pow(i64) -Int 1 "i62 overflow"

;; Floating point
;; --------------

(f32.const 3.245)
#assertTopStack < f32 > 3.245 "f32"

(f64.const 3.234523)
#assertTopStack < f64 > 3.234523 "f32 parens"

(f64.const 1.21460644e+52)
#assertTopStack < f64 > 1.21460644e+52 "f64 scientific 1"

(f64.const 1.6085927714e-321)
#assertTopStack < f64 > 1.6085927714e-321 "f64 scientific 2"

(f64.const 1.63176601e-302)
#assertTopStack < f64 > 1.63176601e-302 "f64 scientific 3"

;; Below examples do not work with current float parser
;; (f64.const 0x1.da21c460a6f44p+52)
;; (f64.const 0x1.60859d2e7714ap-321)
;; (f64.const 0x1.e63f1b7b660e1p-302)

;; Helper conversions
;; ------------------

(i32.const #unsigned(i32, #signed(i32, 0)))
#assertTopStack < i32 > 0 "#unsigned . #signed 1"

(i32.const #unsigned(i32, #signed(i32, #pow1(i32))))
#assertTopStack < i32 > #pow1(i32) "#unsigned . #signed 2"

(i32.const #unsigned(i32, #signed(i32, #pow(i32) -Int 1)))
#assertTopStack < i32 > #pow(i32) -Int 1 "#unsigned . #signed 3"

(i64.const #unsigned(i64, #signed(i64, 0)))
#assertTopStack < i64 > 0 "#unsigned . #signed 4"

(i64.const #unsigned(i64, #signed(i64, #pow1(i64))))
#assertTopStack < i64 > #pow1(i64) "#unsigned . #signed 5"

(i64.const #unsigned(i64, #signed(i64, #pow(i64) -Int 1)))
#assertTopStack < i64 > #pow(i64) -Int 1 "#unsigned . #signed 6"

#clearConfig
