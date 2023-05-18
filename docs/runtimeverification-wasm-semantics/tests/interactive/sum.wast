(module
  (func (param i32) (result i32)
    (local i32)
    (setlocal 1 (const.i32 0))
    (label
      (loop
        (if
          (eq.i32 (getlocal 0) (const.i32 0))
          (break 0)
          (block
            (setlocal 1 (add.i32 (getlocal 0) (getlocal 1)))
            (setlocal 0 (sub.i32 (getlocal 0) (const.i32 1)))
          )
        )
      )
    )
    (return (getlocal 1))
  )

  (export 0)

  (memory 0)
)

(invoke 0 (const.i32 10))
