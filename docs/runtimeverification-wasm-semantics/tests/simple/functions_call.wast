;; Simple add function

(module
  (type $a-cool-type (func (param i32) (param $b i32) ( result i32 )))
)

#assertType 0 [ i32 i32 ] -> [ i32 ]
#assertNextTypeIdx 1

(module
  (type $a-cool-type (func (param i32) (param $b i32) ( result i32 )))
  (func $x (type $a-cool-type)
      (local.get 0)
      (local.get 1)
      (i32.add)
      (return)
  )
  (export "000" (func 0))

;; String-named add function

  (func $add (type $a-cool-type) (param $a i32) (param i32) ( result i32 )
      (local.get $a)
      (local.get 1)
      (i32.add)
      (return)
  )

  ;; Remove return statement, don't use explicit type name

  (func $0 (param $a i32) (param $b i32) result i32
      (local.get $a)
      (local.get $b)
      (i32.add)
  )


  (table 1 funcref)
  (elem 0 (i32.const 0) 2)

  ;; More complicated function with locals

  (func $1 param i64 i64 i64 result i64 local i64
      (i64.sub (local.get 2) (i64.add (local.get 0) (local.get 1)))
      (local.set 3)
      (local.get 3)
      (return)
  )

  ( export "export-1" (func 3) )
)

(assert_return (invoke "000" (i32.const 7) (i32.const 8)) (i32.const 15))
#assertFunction 0 [ i32 i32 ] -> [ i32 ] [ ] "call function 0 exists"

#assertFunction 1 [ i32 i32 ] -> [ i32 ] [ ] "function string-named add"
#assertNextTypeIdx 2

(assert_return (invoke "export-1" (i64.const 100) (i64.const 43) (i64.const 22)) (i64.const -121))
#assertFunction 3 [ i64 i64 i64 ] -> [ i64 ] [ i64 ] "call function 1 exists"


(i32.const 7)
(i32.const 8)
(i32.const 0)
(call_indirect (type $a-cool-type))

#assertTopStack < i32 > 15 "call function 0 no return"
(drop)
#assertFunction 2 [ i32 i32 ] -> [ i32 ] [ ] "call function 0 exists no return"

;; Function with complicated declaration of types
(module
  (func $2 result i32 param i32 i64 param i64 local i32
      (local.get 0)
      (return)
  )
  (func (export "out-of-order-type-declaration") (result i32)
    (i32.const 7)
    (i64.const 8)
    (i64.const 5)
    (call $2)
   )
)
(assert_return (invoke "out-of-order-type-declaration") (i32.const 7))
#assertFunction 0 [ i32 i64 i64 ] -> [ i32 ] [ i32 ] "out of order type declarations"
#assertNextTypeIdx 2

;; Function with empty declarations of types

(module
  (func $0 param i64 i64 result result i64 param local
      (local.get 0)
      (return)
  )

  (func $1 (param i64 i64) (result i64)
      (local.get 0)
      (return)
  )
  (func (export "cool") (result i64)
      i64.const 10
      i64.const 11
      call $1
  )
)

(assert_return (invoke "cool") (i64.const 10))
#assertFunction 1 [ i64 i64 ] -> [ i64 ] [ ] "empty type declarations"
#assertNextTypeIdx 2

;; Function with just a name

(module
  (func $3)
  (export "return-null" (func $3) )
)
(assert_return (invoke "return-null"))

#assertFunction 0 [ ] -> [ ] [ ] "no domain/range or locals"

(module
    (func $add (export "add")
        (param i32 i32)
        (result i32)
        (local.get 0)
        (local.get 1)
        (i32.add)
        (return)
    )

    (func $sub (export "sub")
        (param i32 i32)
        (result i32)
        (local.get 0)
        (local.get 1)
        (i32.sub)
        (return)
    )

    (func $mul (export "mul")
        (param i32 i32)
        (result i32)
        (local.get 0)
        (local.get 1)
        (i32.mul)
        (return)
    )

    (func $xor (export "xor") (param i32 i32) (result i32)
        (local.get 0)
        (local.get 1)
        (i32.xor)
    )
)

(assert_return (invoke "add" (i32.const 3) (i32.const 5)) (i32.const 8))
(assert_return (invoke "mul" (i32.const 3) (i32.const 5)) (i32.const 15))
(assert_return (invoke "sub" (i32.const 12) (i32.const 5)) (i32.const 7))
(assert_return (invoke "xor" (i32.const 3) (i32.const 5)) (i32.const 6))

#assertFunction 0 [ i32 i32 ] -> [ i32 ] [ ] "add function typed correctly"
#assertFunction 1 [ i32 i32 ] -> [ i32 ] [ ] "sub function typed correctly"
#assertFunction 2 [ i32 i32 ] -> [ i32 ] [ ] "mul function typed correctly"
#assertFunction 3 [ i32 i32 ] -> [ i32 ] [ ] "xor function typed correctly"
#assertNextTypeIdx 1

(module
    (func $f1 (param $a i32) (param i32) (result i32) (local $c i32)
        (local.get $a)
        (local.get 1)
        (i32.add)
        (local.set $c)
        (local.get $a)
        (local.get $c)
        (i32.mul)
        (return)
    )

    (func $f2 (param i32 i32 i32 ) (result i32) (local i32 i32)
        (local.get 0)
        (local.get 2)
        (call $f1)
        (local.get 1)
        (call $f1)
        (local.get 0)
        (i32.mul)
        (return)
    )

    (func (export "nested-method-call") (result i32)
        (i32.const 3)
        (i32.const 5)
        (call $f1)
        (i32.const 5)
        (i32.const 8)
        (call $f2)
    )

)

(assert_return (invoke "nested-method-call") (i32.const 14247936))
#assertFunction 0 [ i32 i32 ] -> [ i32 ] [ i32 ] "inner calling method"
#assertFunction 1 [ i32 i32 i32 ] -> [ i32 ] [ i32 i32 ] "outer calling method"

(module
    (func $func (param i32 i32) (result i32) (local.get 0))
    (func (export "aaa") (result i32)
    (block (result i32)
        (call $func
        (block (result i32) (i32.const 1)) (i32.const 2)
        )
    )
    )
)

(assert_return (invoke "aaa") (i32.const 1))

(module
    (func $2 (export "cool-align-1") (export "cool-align-2") result i32 param i32 i64 param i64 local i32
        (local.get 0)
        (return)
    )
)

(assert_return (invoke "cool-align-1" (i32.const 7) (i64.const 8) (i64.const 3)) (i32.const 7))
(assert_return (invoke "cool-align-2" (i32.const 1) (i64.const 5) (i64.const 7)) (i32.const 1))

#assertFunction 0 [ i32 i64 i64 ] -> [ i32 ] [ i32 ] "out of order type declarations"

(module
  (func (export "foo") (result i32)
    (block $a (result i32)
      (block $b (result i32)
        (call $bar)
        (i32.const 0)
        (br $b)
      )
      (drop)
      (i32.const 1)
    )
  )

 (func $bar
   (block $b (block $a (br $a)))
 )
)

(assert_return (invoke "foo") (i32.const 1))

;; Check type is correctly desugared.

(module
  (func $1 param i64 i64 i64 result i64 local i64
      (i64.sub (local.get 2) (i64.add (local.get 0) (local.get 1)))
      (local.set 3)
      (local.get 3)
      (return)
  )

  ( export "export-1" (func $1) )

  (func $2 param i64 i64 i64 result i64 local i64
      (i64.sub (local.get 2) (i64.add (local.get 0) (local.get 1)))
      (local.set 3)
      (local.get 3)
      (return)
  )
)

(assert_return (invoke "export-1" (i64.const 100) (i64.const 43) (i64.const 22)) (i64.const -121))
#assertFunction 0 [ i64 i64 i64 ] -> [ i64 ] [ i64 ] "call function 1 exists"
#assertType 0 [ i64 i64 i64 ] -> [ i64 ]
;; Check type was only added once.
#assertNextTypeIdx 1

#clearConfig
