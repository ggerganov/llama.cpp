(module
  (func (export "foo") (result i32) (i32.const 0))
  (func (export "bar") (result i32) (i32.const 1))
)

(register "a")

;; Test that imports get ordered correctly.
;; Function "bar" should get index 0, since it is imported first.
(module
  (func (import "a" "bar") (result i32))
  (import "a" "foo" (func (result i32)))
  (export "bar" (func 0) )
)

(assert_return (invoke "bar") (i32.const 1))

;; Test that data initializations get ordered correctly.
;; The results of the inlined `data` should overwrite the results of the non-inlined data.
(module
  (data (offset (i32.const 0)) "b")
  (memory (data "a"))
  (func (export "baz") (result i32)
    (i32.load (i32.const 0))
  )
)

(assert_return (invoke "baz") (i32.const 97))

;; Same as above but for `elem`
(module
  (elem (offset (i32.const 0)) 0)
  (table funcref (elem 1))
  (func (result i32) (i32.const 0))
  (func (result i32) (i32.const 1))
  (func (export "biz") (result i32)
    (call_indirect (result i32) (i32.const 0))
  )
)

(assert_return (invoke "biz") (i32.const 1))

;; Regression test: A module with hex integers inside a function after a `table` with inline `elem`.
(module
  (table funcref (elem))
  (func (export "break-inner") (result i32)
    (local i32)
    (local.set 0 (i32.const 0))
    (local.set 0 (i32.add (local.get 0) (block (result i32) (block (result i32) (br 1 (i32.const 0x1))))))
    (local.set 0 (i32.add (local.get 0) (block (result i32) (block (br 0)) (i32.const 0x2))))
    (local.set 0
      (i32.add (local.get 0) (block (result i32) (i32.ctz (br 0 (i32.const 0x4)))))
    )
    (local.set 0
      (i32.add (local.get 0) (block (result i32) (i32.ctz (block (result i32) (br 1 (i32.const 0x8))))))
    )
    (local.get 0)
  )
)

#clearConfig
