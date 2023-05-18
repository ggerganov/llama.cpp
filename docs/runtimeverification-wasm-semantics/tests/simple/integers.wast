(module
  (func (export "add0") (param $x i32) (result i32) (i32.add (local.get $x) (i32.const 0)))
)

(assert_return (invoke "add0" (i32.const 123)) (i32.const 123))
(assert_return (invoke "add0" (i32.const +123)) (i32.const 1_2_3))
(assert_return (invoke "add0" (i32.const +123)) (i32.const 1_2_3))
(assert_return (invoke "add0" (i32.const -1_23)) (i32.const -12_3))
(assert_return (invoke "add0" (i32.const -0x11)) (i32.const -17))
(assert_return (invoke "add0" (i32.const -0x1_1)) (i32.const -1_7))
(assert_return (invoke "add0" (i32.const 0xF_FF_F)) (i32.const 65535))
(assert_return (invoke "add0" (i32.const 0xF_FF_F)) (i32.const 65_535))
(assert_return (invoke "add0" (i32.const -0xF_F111_1)) (i32.const -16716049))
(assert_return (invoke "add0" (i32.const -0xAABBCCDD)) (i32.const -0xA_A_B_B_C_C_D_D))

#clearConfig
