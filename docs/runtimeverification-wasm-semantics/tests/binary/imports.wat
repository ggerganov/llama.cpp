(module
  (type (func (param i32) (result i64)))
  (import "env" "foo" (func))
  (import "env" "baz" (global (mut f64)))
  (import "env" "faz" (memory 1 3))
  (import "env" "bar" (table 1 3 funcref)))
