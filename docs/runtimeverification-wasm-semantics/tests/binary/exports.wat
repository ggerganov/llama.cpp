(module
  (func   (export "foo"))
  (func   (export "fom"))
  (memory (export "baz") 1 1)
  (global (export "faz") (mut f64) (f64.const 0))
  (table  (export "bar") 1 3 funcref))