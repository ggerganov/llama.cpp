(module
  (func (export "as-if-cond") (param i32) (result i32)
    (block (result i32)
      (if (result i32)
        (br_if 0 (i32.const 1) (local.get 0))
        (then (i32.const 2))
        (else (i32.const 3))
      )
    )
  )

  (func (export "to-top-level0") (br 0))
  (func (export "to-top-level1") (block (br 0)))
)

(assert_return (invoke "as-if-cond" (i32.const 1)) (i32.const 1))
(assert_return (invoke "to-top-level0"))
(assert_return (invoke "to-top-level1"))

#clearConfig