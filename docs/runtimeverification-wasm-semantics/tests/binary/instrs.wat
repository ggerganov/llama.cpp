(module
  (memory 0)
  (table 100 funcref)
  (global (mut i32) (i32.const 0))
  (func (param i32 i64) (local f64))
  (func (local i32)
        ;; `unreachable` and `drop` are inserted as needed to
        ;; ensure correct typing.
        unreachable

        ;; Numeric Instrs
        ;; --------------
        ;; Commented out instructions are not currently supported by py-wasm.
        i32.const 0                   drop
        f32.const 0                   drop
     ;; i32.extend8_s                 drop
     ;; i32.extend16_s                drop
     ;; i64.extend32_s                drop
        i32.wrap_i64                  drop
        i64.extend_i32_s              drop
        i64.extend_i32_u              drop
        i32.trunc_f64_s               drop
        i64.trunc_f32_u               drop
     ;; i32.trunc_sat_f64_s           drop
     ;; i32.trunc_sat_f64_u           drop
        f32.demote_f64                drop
        f64.promote_f32               drop
        f64.convert_i32_s             drop
        f64.convert_i32_u             drop
     ;; i32.reinterpret_f32           drop
     ;; f64.reinterpret_i64           drop
        ;; --
        unreachable

        ;; IUnOp
        ;; -----
        i32.clz
        i32.ctz
        i32.popcnt
        ;; --
        unreachable

        ;; IBinOp
        ;; ------
        i64.add
        i64.sub
        i64.mul
        i64.div_s
        i64.div_u
        i64.rem_s
        i64.rem_u
        i64.and
        i64.or
        i64.xor
        i64.shl
        i64.shr_s
        i64.shr_u
        i64.rotl
        i64.rotr
        ;; --
        unreachable

        ;; FUnOp
        ;; -----
        f32.abs
        f32.neg
        f32.sqrt
        f32.ceil
        f32.floor
        f32.trunc
        f32.nearest
        ;; --
        unreachable

        ;; FBinOp
        ;; ------
        f64.add
        f64.sub
        f64.mul
        f64.div
        f64.min
        f64.max
        f64.copysign
        ;; --
        unreachable

        ;; ITestop
        ;; -------
        i32.eqz
        ;; --
        unreachable

        ;; IRelOp
        ;; ------
        i32.eq
        i32.ne
        i32.lt_s
        i32.lt_u
        i32.gt_s
        i32.gt_u
        i32.le_s
        i32.le_u
        i32.ge_s
        i32.ge_u
        ;; --
        unreachable

        ;; FRelOp
        ;; ------
        f64.eq     drop
        f64.ne     drop
        f64.lt     drop
        f64.gt     drop
        f64.le     drop
        f64.ge     drop
        ;; --
        unreachable

        ;; Stack Instrs
        ;; ------------
        drop
        select
        ;; --
        unreachable

        ;; Variable Instrs
        ;; ---------------
        local.get 0
        local.set 0
        local.tee 0
        global.get 0
        global.set 0
        ;; --
        unreachable

        ;; Memory Instrs
        ;; -------------
        i32.load offset=10           drop
        f32.load offset=10           drop
        i32.store offset=10          drop
        f32.store offset=10          drop
        i32.load8_s offset=10        drop
        i32.load8_u offset=10        drop
        i32.load16_s offset=1        drop
        i32.load16_u offset=1        drop
        i64.load32_s offset=1        drop
        i64.load32_u offset=1        drop
        i32.store8 offset=10         drop
        i32.store16 offset=10        drop
        i64.store32 offset=10        drop
        memory.size
        memory.grow
        ;; --
        unreachable

        ;; Control Instrs
        ;; --------------
        nop
        unreachable
        block (result i32) unreachable end
        block (result i32) unreachable nop end
        block end
        if    (result f32) unreachable else unreachable end
        loop               unreachable end
        br 0
        br_if 0
        br_table 0 0 0
        return
        call 0
        call_indirect (type 0)
        ;; --
        unreachable
    )

)
