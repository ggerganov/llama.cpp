(module
  (memory 1)
  (func $inc
    (i32.store8
      (i32.const 0)
      (i32.add
        (i32.load8_u (i32.const 0))
        (i32.const 1)))
  )
  (func $main
    (i32.store (i32.const 0) (i32.const 65))
    (call $inc)
    (call $inc)
    (call $inc)
  )
  (start $main)
)

#assertMemoryData (0, 68) "start inc"
#assertFunction 0  [ ] -> [ ] [ ] "$inc"
#assertFunction 1 [ ] -> [ ] [ ] "$main"
#assertMemory 0 1 .Int ""

#clearConfig

(module
  (func $foo (unreachable))
  (start $foo)
)
#assertTrap "Trap propagates through start invocation"
#assertFunction 0  [ ] -> [ ] [ ] ""

(assert_trap
  (module (func $main (unreachable)) (start $main))
  "unreachable"
)

#clearConfig
