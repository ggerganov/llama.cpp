;; Unreachable
( unreachable )
#assertTrap "unreachable"

;; Blocks

block (result i32 i32 i32)
    (i32.const 1)
    (i32.const 2)
    (i32.const 3)
end
#assertStack < i32 > 3 : < i32 > 2 : < i32 > 1 : .ValStack "block 1"

block (result i32 i32)
    (i32.const 1)
    (i32.const 2)
    (i32.const 3)
    (drop)
end
#assertStack < i32 > 2 : < i32 > 1 : .ValStack "block 2"

block (result i32 i32)
    (i32.const 1)
    (i32.const 2)
    (i32.const 3)
end
#assertStack < i32 > 3 : < i32 > 2 : .ValStack "block 3 (invalid)"

;; (block (result i32)
;;     (i32.const 1)
;; )
;; #assertTopStack < i32 > 1 "block with named result 1"

;; (block result i64 i32
;;     (i32.const 2)
;;     (i32.const 1)
;;     (i64.const 5)
;; )
;; #assertStack < i64 > 5 : < i32 > 1 : .ValStack "block with named result 2"

;; Breaks

(i32.const 1)
(i32.const 2)
block
    (i32.const 3)
    (br 0)
    (i32.const 4)
    (br 0)
end
#assertStack < i32 > 2 : < i32 > 1 : .ValStack "br 1"

(i32.const 1)
(i32.const 2)
block
    (i32.const 3)
    block (result i32 i32)
        (i32.const 4)
        (i32.const 5)
        (br 1)
    end
    (i32.const 6)
    (br 0)
end
#assertStack < i32 > 2 : < i32 > 1 : .ValStack "br 2"

(i32.const 1)
(i32.const 2)
block (result i32 i32)
    (i32.const 3)
    block (result i32 i32)
        (i32.const 4)
        (i32.const 5)
        (br 1)
    end
    (i32.const 6)
    (br 0)
end
#assertStack < i32 > 5 : < i32 > 4 : < i32 > 2 : < i32 > 1 : .ValStack "br 3"

(i32.const 1)
(i32.const 2)
block (result i32 i32)
    (i32.const 3)
    block
        (i32.const 4)
        (i32.const 5)
        (br 1)
    end
    (i32.const 6)
    (br 0)
end
#assertStack < i32 > 5 : < i32 > 4 : < i32 > 2 : < i32 > 1 : .ValStack "br 4"

(i32.const 1)
(i32.const 2)
block (result i32)
    (i32.const 3)
    (i32.const 0)
    (br_if 0)
    (i32.const 4)
    (br 0)
end
#assertStack < i32 > 4 : < i32 > 2 : < i32 > 1 : .ValStack "br_if 1 false"

(i32.const 1)
(i32.const 2)
block (result i32)
    (i32.const 3)
    (i32.const 1)
    (br_if 0)
    (i32.const 4)
    (br 0)
end
#assertStack < i32 > 3 : < i32 > 2 : < i32 > 1 : .ValStack "br_if 1 true"

(i32.const 1)
(i32.const 2)
block
    (i32.const 3)
    (i32.const 1)
    (br_if 0)
    (i32.const 4)
    (br 0)
end
#assertStack < i32 > 2 : < i32 > 1 : .ValStack "br_if 2 true"

;; Conditional

(i32.const 1)
if (result i32) i32.const 1 else i32.const -1 end
#assertTopStack < i32 > 1 "if true"

(i32.const 0)
if (result i32) i32.const 1 else i32.const -1 end
#assertTopStack < i32 > -1 "if false"

;; (i32.const -1)
;; (if (i32.const 0) (then))
;; #assertTopStack < i32 > -1 "if folded false empty"

;; (i32.const -1)
;; (if (result i32) (i32.const 1) (then (i32.const 1)) (else (i32.const 2)))
;; #assertStack < i32 > 1 : < i32 > -1 : .ValStack "if folded true"

;; (i32.const -1)
;; (if (result i32) (i32.const 0) (then (i32.const 1)) (else (i32.const 2)))
;; #assertStack < i32 > 2 : < i32 > -1 : .ValStack "if folded false"

;; (if (result i32) (i32.const 1) (then (unreachable)) (else (i32.const 1)))
;; #assertTrap "if lazy first branch true"

;; (if (result i32) (i32.const 0) (then (unreachable)) (else (i32.const 1)))
;; #assertTopStack < i32 > 1 "if lazy first branch false"

;; (if (result i32) (i32.const 1) (then (i32.const -1)) (else (unreachable)))
;; #assertTopStack < i32 > -1 "if lazy second branch true"

;; (if (result i32) (i32.const 0) (then (i32.const -1)) (else (unreachable)))
;; #assertTrap "if lazy second branch false"

;; (if (result i32) (unreachable) (then (i32.const -1)) (else (unreachable)))
;; #assertTrap "if strict condition"

;; Looping

init_locals < i32 > 10 : < i32 > 0 : .ValStack
loop
    (local.get 0)
    (local.get 1)
    (i32.add)
    (local.set 1)
    (local.get 0)
    (i32.const 1)
    (i32.sub)
    (local.tee 0)
    (br_if 0)
end
#assertLocal 0 < i32 > 0  "sum 1 -> 10 loop"
#assertLocal 1 < i32 > 55 "sum 1 -> 10 loop"

init_locals < i32 > 10 : < i32 > 0 : .ValStack
block
    loop
        (local.get 0)
        (local.get 1)
        (i32.add)
        (local.set 1)
        (local.get 0)
        (i32.const 1)
        (i32.sub)
        (local.tee 0)
        (i32.eqz)
        (br_if 1)
        (br 0)
    end
end
#assertLocal 0 < i32 > 0  "sum 1 -> 10 loop concrete syntax"
#assertLocal 1 < i32 > 55 "sum 1 -> 10 loop concrete syntax"

;; Stack Underflow
;; TODO: We need to give semantics to stack underflow (though it could not happen with a validated program).
;; We need `trap` semantics first.
;; (i32.const 0)
;; block [ i32 i32 ]
;;     (i32.const 7)
;; end

#clearConfig
