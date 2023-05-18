;; Test locals

init_locals < i32 > 0 : < i32 > 0 : < i32 > 0 : .ValStack

(i32.const 43)
(local.set 0)
#assertLocal 0 < i32 > 43 "set_local"

(i32.const 55)
(local.set 1)
(local.get 1)
#assertTopStack < i32 > 55 "set_local stack"
#assertLocal 1 < i32 > 55 "set_local"

(i32.const 67)
(local.tee 2)
#assertTopStack < i32 > 67 "tee_local stack"
#assertLocal 2 < i32 > 67 "tee_local local"

;; Test globals

(module
    (global (mut i32) (i32.const 0))
    (global $someglobal (mut i32) (i32.const 0))

    (func
        (i32.const 43)
        (global.set 0)
    )

    (func (export "set")
      (i32.const 55)
      (global.set $someglobal)
    )

    (start 0)
)
#assertGlobal 0 < i32 > 43 "set_global"

(invoke "set")
#assertGlobal $someglobal < i32 > 55 "set_global"

;; Test global folded forms

#clearConfig

(module
    (global (mut i32) (i32.const 0))
    (global (mut i32) (i32.const 0))

    (func
        (global.set 1 (i32.const 99))
        (global.set 0 (i32.const 77))
    )

    (start 0)
)

#assertGlobal 1 < i32 > 99 "set_global folded"
#assertGlobal 0 < i32 > 77 "set_global folded 2"

#clearConfig
