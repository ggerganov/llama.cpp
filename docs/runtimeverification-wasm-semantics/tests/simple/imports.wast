(module $a
   (global (export "g") (export "glob") (mut i32) (i32.const 42))
   (memory (export "m") (export "mem") (data "A"))
   (type $t (func ))
   (func (export "f") (export "func"))
   (func (export "gunc") (param i64) (param i32) (result i32) (local.get 1))
   )

(register "m")

(module
 (import "m" "gunc" (func (type $t)))
 (memory (import "m" "mem") 1)
 (export "x" (global $x))
 (type $t (func (param i64) (param i32) (result i32)))
 (func (import "m" "gunc") (type $t))
 (func (import "m" "f"))
 (global $x (import "m" "g") (mut i32))
 (import "m" "g" (global (mut i32)))
 (func (export "foo") (result i32) (global.get 0))
 (func (export "mod") (global.set 0 (i32.const 10)))
 )

(assert_return (invoke "foo") (i32.const 42))
(invoke "mod")
(invoke $a "f")
(assert_return (invoke "foo") (i32.const 10))
(assert_return (get $a "g") (i32.const 10))
(assert_return (get "x") (i32.const 10))

#clearConfig