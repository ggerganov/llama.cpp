// (c) 2015 Andreas Rossberg

(module
  // Aligned read/write
  (func (param) (result i32)
    (local i32 i32 i32)
    (setlocal 0 (const.i32 10))
    (label
      (loop
        (if
          (eq.i32 (getlocal 0) (const.i32 0))
          (break)
        )
        (setlocal 2 (mul.i32 (getlocal 0) (const.i32 4)))
        (setnears.i32 (getlocal 2) (getlocal 0))
        (setlocal 1 (getnears.i32 (getlocal 2)))
        (if
          (neq.i32 (getlocal 0) (getlocal 1))
          (return (const.i32 0))
        )
        (setlocal 0 (sub.i32 (getlocal 0) (const.i32 1)))
      )
    )
    (return (const.i32 1))
  )

  // Unaligned read/write
  (func (param) (result i32)
    (local i32 i32 i32)
    (setlocal 0 (const.i32 10))
    (label
      (loop
        (if
          (eq.i32 (getlocal 0) (const.i32 0))
          (break)
        )
        (setlocal 2 (getlocal 0))
        (setnearunaligneds.i32 (getlocal 0) (getlocal 2))
        (setlocal 1 (getnearunaligneds.i32 (getlocal 0)))
        (if
          (neq.i32 (getlocal 2) (getlocal 1))
          (return (const.i32 0))
        )
        (setlocal 0 (sub.i32 (getlocal 0) (const.i32 1)))
      )
    )
    (return (const.i32 1))
  )

  // Memory cast
  (func (param) (result f64)
    (local)
    (setnears.i64 (const.i32 8) (const.i64 -12345))
    (if
      (eq.f64 (getnear.f64 (const.i32 8)) (cast.i64.f64 (const.i64 -12345)))
      (return (const.f64 0))
    )
    (setfarunaligneds.i16 (const.i64 3) (const.i32 -23423))
    (return (getnear.f64 (const.i32 0)))
  )

  (export 0 1 2)

  (memory 64)
)

(invoke 0)
(invoke 1)
//(invoke 2)
