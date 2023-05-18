(module $myMod)

#assertNamedModule $myMod "named empty module"

(module $anotherName)

(register "a module name")

#assertRegistrationNamed "a module name" $anotherName "registration1"
#assertNamedModule $anotherName "named registered module"

(module $myMod2)

(module)

(module $myMod3)

(register "a module name 2" $myMod2)
(register "another module name" $myMod3)
(register "third module name")

#assertRegistrationNamed "another module name" $myMod3 "registration3"
#assertRegistrationNamed "a module name 2" $myMod2 "registration4"
#assertRegistrationUnnamed "third module name" "registration5"

(assert_malformed
  (module quote "(func block end $l)")
  "mismatching label"
)

(assert_malformed
  (module quote "(func block $a end $l)")
  "mismatching label"
)

#clearConfig

;; Test ordering of definitions in modules.

(module
  (start $main) ;; Should initialize memory position 1.
  (elem (i32.const 1) $store)
  (data (i32.const 100) "ba")
  (data (i32.const 100) "c") ;; Should overwrite previous, leaving "5 1" as memory bytes
  (func)
  (func $main (call_indirect (i32.const 1))) ;; Should call $store.
  (func $store (i32.store (i32.const 1) (i32.const 42)))
  (func $get (export "get") (result i32)
    (i32.add (i32.load (i32.const 1)) (i32.load (i32.const 100))) ;; For checking both data initialization.
  )
  (memory 10 10)
  (elem (i32.const 0) 0)
  (table 2 funcref)
)

(assert_return (invoke "get") i32.const 24973 )

#clearConfig
