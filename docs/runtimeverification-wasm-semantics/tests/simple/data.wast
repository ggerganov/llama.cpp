;; Instantiating with data

(module
(memory $an-ident (data "WASM" "2\2E0"))
)

(memory.size)

#assertTopStack < i32 > 1 "size of stack"
#assertMemoryData (0, 87) "text to ascii W"
#assertMemoryData (1, 65) "text to ascii A"
#assertMemoryData (2, 83) "text to ascii S"
#assertMemoryData (3, 77) "text to ascii M"
#assertMemoryData (4, 50) "text to ascii 2"
#assertMemoryData (5, 46) "text to ascii ."
#assertMemoryData (6, 48) "text to ascii 0"
#assertMemory $an-ident 1 1 "memorys string length"

#clearConfig

(module
(memory 1 1)
(data (offset (i32.const 100)) "W" "AS" "M")
)
#assertMemoryData (100, 87) "text to ascii W"
#assertMemoryData (101, 65) "text to ascii A"
#assertMemoryData (102, 83) "text to ascii S"
#assertMemoryData (103, 77) "text to ascii M"
#assertMemory 0 1 1 "memorys string length"

#clearConfig

(module
(memory (data))
)
#clearConfig

(module
(memory (data "W"))
)
#assertMemoryData (0, 87) "text to ascii W"
#assertMemory 0 1 1 "memorys string length"

#clearConfig

(module
(memory (data "\"\t\n\n\t\'\"\r\u{090A}"))
)
#assertMemoryData (0, 34) "text to ascii special"
#assertMemoryData (1, 9) "text to ascii special"
#assertMemoryData (2, 10) "text to ascii special"
#assertMemoryData (3, 10) "text to ascii special"
#assertMemoryData (4, 9) "text to ascii special"
#assertMemoryData (5, 39) "text to ascii special"
#assertMemoryData (6, 34) "text to ascii special"

(module
  (memory $m 1 1)
  (data (offset (i32.const 0)) "\00")
  (data (offset (nop) (i32.const 1)) "\01")
  (data (offset (i32.const 2) (nop)) "\02")
  (data $m (offset (i32.const 3)) "\03")
  (data $m (offset (nop) (i32.const 4)) "\04")
  (data $m (offset (i32.const 5) (nop)) "\05")

  (data  (offset (i32.const 6 (nop))) "\06")
  (data $m (offset (i32.const 7 (nop))) "\07")

  (global $g i32 (i32.const 8))
  (global $h i32 (i32.const 9))

  (data (offset (global.get $g)) "\08")
  (data $m (offset (global.get $h)) "\09")

  (func $main (local i32)
    (local.set 0 (i32.const 9))
    loop
      (i32.load8_u (local.get 0))
      (local.get 0)
      (if (i32.ne) (then (unreachable)))
      (i32.sub (local.get 0) (i32.const 1))
      (local.tee 0)
      (i32.eqz)
      (br_if 1)
      (br 0)
    end
    )

    (start $main)
)
#clearConfig
