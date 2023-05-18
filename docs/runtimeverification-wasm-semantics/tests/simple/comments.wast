(i32.const 7)
;; this should be ignored
(i32.const 8) ;; this should be ignored as well
(i32.add)

(;
all this text
should be ignored
;)

#assertTopStack < i32 > 15 "dummy test 1"

(i32.const -3)
(i32.const 6)     (; comment at end of line ;)
(i32.add)
#assertTopStack < i32 > 3 "dummy test 2"

(i32.const -3)
(i32.(;comment in the middle;)const 6)
(i32.add)
#assertTopStack < i32 > 3 "dummy test 2"

#clearConfig
