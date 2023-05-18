( memory 34)
#assertMemory 0 34 .Int "memory initial 2"

#clearConfig

( memory $a-memory 34)
#assertMemory $a-memory 34 .Int "memory initial 2"

#clearConfig

( memory 4 10 )
#assertMemory 0 4 10 "memory initial 3"

#clearConfig

( memory $more-memory 4 10 )
#assertMemory $more-memory 4 10 "memory initial 3"

#clearConfig

( memory $mem 0 10 )
(memory.size)
#assertTopStack <i32> 0 "memory.size 1"
#assertMemory $mem 0 10 "memory ungrown"

#clearConfig

( memory $mem 0 10 )
(memory.grow (i32.const 10))
(memory.size)
#assertStack <i32> 10 : < i32 > 0 : .ValStack "memory grow"
(memory.grow (i32.const 1))
#assertTopStack <i32> -1 "memory grow"
#assertMemory $mem 10 10 "memory grown"

#clearConfig

( memory #maxMemorySize())
(memory.grow (i32.const 1))
#assertTopStack <i32> -1 "memory grow max too large"
#assertMemory 0 #maxMemorySize() .Int "memory grow max too large"

#clearConfig

( memory 0 )
(memory.grow (i32.const #maxMemorySize()))
(memory.size)
#assertStack <i32> #maxMemorySize() : < i32 > 0 : .ValStack "memory grow unbounded"
(memory.grow (i32.const 1))
(memory.size)
#assertStack <i32> #maxMemorySize() : < i32 > -1 : .ValStack "memory grow unbounded"
#assertMemory 0 #maxMemorySize() .Int "memory grown unbounded"

;; Store and load

#clearConfig

(memory 1)
(i32.const 1)
(i64.const 1)
(i64.store offset=2)
#assertMemoryData (3, 1) "store is little endian"
(i32.const 1)
(i64.const 257)
(i64.store8 offset=2)
#assertMemoryData (3, 1) "store8"
(i32.const 1)
(i64.const 65537)
(i64.store16 offset=2)
#assertMemoryData (3, 1) "store16"
(i32.const 1)
(i64.const #pow(i32) +Int 1)
(i64.store16 offset=2)
#assertMemoryData (3, 1) "store32"
#assertMemory 0 1 .Int ""

#clearConfig

(memory $foo 0)
(i32.const 0)
(i32.const 0)
(i32.store8)
#assertTrap "store to 0 size memory"
#assertMemory $foo 0 .Int ""

#clearConfig

(memory 1)
(i32.const 65535)
(i32.const 1)
(i32.store8)
#assertMemoryData (65535, 1) "store to memory edge"
(i32.const 65535)
(i32.const 1)
(i32.store16)
#assertTrap "store outside of size memory"
#assertMemory 0 1 .Int ""

#clearConfig

(memory 1)
(i32.const 15)
(i64.const #pow(i32) -Int 1)
(i64.store)
(i32.const 15)
(i32.load8_u)
#assertTopStack <i32> 255 "load8 unsigned"
(i32.const 15)
(i32.load8_s )
#assertTopStack <i32> -1 "load8 signed"
(i32.const 16)
(i32.load16_u )
#assertTopStack <i32> 65535 "load16 unsigned"
(i32.const 16)
(i32.load16_s )
#assertTopStack <i32> -1 "load16 signed"
(i32.const 15)
(i64.load32_u )
#assertTopStack <i64> #pow(i32) -Int 1 "load32 unsigned1"
(i32.const 15)
(i64.load32_s )
#assertTopStack <i64> -1 "load32 signed1"
(i32.const 17)
(i64.load32_u )
#assertTopStack <i64> 65535 "load32 unsigned2"
(i32.const 17)
(i64.load32_u )
#assertTopStack <i64> 65535 "load32 signed2"
#assertMemoryData (15, 255) ""
#assertMemoryData (16, 255) ""
#assertMemoryData (17, 255) ""
#assertMemoryData (18, 255) ""
#assertMemory 0 1 .Int ""

;; Updating

#clearConfig

(memory 1)
(i32.const 1)
(i64.const #pow(i64) -Int 1)
(i64.store)
(i32.const 5) (i32.const 0)
(i32.store   )
(i32.const 3) (i32.const 0)
(i32.store16 )
(i32.const 1) (i32.const 0)
(i32.store8  )
(i32.const 2) (i32.const 0)
(i32.store8  )
#assertMemory 0 1 .Int "Zero updates erases memory"

#clearConfig

(memory 1)
(i32.const 1) (i64.const #pow(i64) -Int 1)
(i64.store )
(i32.const 2) (i32.const 0)
(i32.store8 )
(i32.const 4) (i32.const 0)
(i32.store )
#assertMemoryData (1, 255) ""
#assertMemoryData (3, 255) ""
#assertMemoryData (8, 255) ""
#assertMemory 0 1 .Int "Zero updates don't over-erase"

#clearConfig

(module
  (memory 0)
)

(module
  (memory (data "A"))
)

#assertMemoryData (0, 65) ""

(module
  (memory 1)
  (func $start (i32.store (i32.const 0) (i32.const 42)))
  (start $start)
)

#assertMemoryData 1 (0, 65) "Start didn't modify other memory"
#assertMemoryData (0, 42) "Start function modified its own memory"

(module
    (memory 0)

    (func (export "load_at_zero") (result i32) (i32.load (i32.const 0)))
    (func (export "store_at_zero") (i32.store (i32.const 0) (i32.const 2)))
)

(assert_trap (invoke "store_at_zero") "out of bounds memory access")
(assert_trap (invoke "load_at_zero") "out of bounds memory access")

#clearConfig
