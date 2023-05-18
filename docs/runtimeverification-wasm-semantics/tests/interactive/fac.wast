// (c) 2015 Andreas Rossberg

(module
  // Recursive factorial
  (func (param i32) (result i32)
    (local) // opt
    (return
      (if
        (eq.i32 (getlocal 0) (const.i32 0))
        (const.i32 1)
        (mul.i32 (getlocal 0) (call 0 (sub.i32 (getlocal 0) (const.i32 1))))
      )
    )
  )

  // Recursive factorial with implicit return value
  (func (param i32) (result i32)
    (local) // opt
      (if
        (eq.i32 (getlocal 0) (const.i32 0))
        (const.i32 1)
        (mul.i32 (getlocal 0) (call 0 (sub.i32 (getlocal 0) (const.i32 1))))
      )
  )

  // Iterative factorial
  (func (param i32) (result i32)
    (local i32)
    (setlocal 1 (const.i32 1))
    (label
      (loop
        (if
          (eq.i32 (getlocal 0) (const.i32 0))
          (break 0)
          (block
            (setlocal 1 (mul.i32 (getlocal 0) (getlocal 1)))
            (setlocal 0 (sub.i32 (getlocal 0) (const.i32 1)))
          )
        )
      )
    )
    (return (getlocal 1))
  )

  (export 0 1 2)

  (memory 0)
)

(invoke 0 (const.i32 3))
(invoke 1 (const.i32 3))
(invoke 2 (const.i32 3))
