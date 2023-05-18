(module
  (func (param) (result i32 i32)
    (local)
    (return
      (const.i32 1)
      (const.i32 2)
    )
  )

  (func (param) (result i32)
    (local i32 i32)
    (destruct 0 1 (call 0))
    (return (add.i32 (getlocal 0) (getlocal 1)))
  )

  (export 0 1)

  (memory 0)
)

(invoke 1)
