;; tests of function identifier names

(module
  (func $oeauth
      (param i32 i32)
      (result i32)
      (local.get 0)
      (local.get 1)
      (i32.add)
      (return)
  )

  (func $023eno!thu324
      (param i32 i32)
      (result i32)
      (local.get 0)
      (local.get 1)
      (i32.add)
      (return)
  )

  (func $02$3e%no!t&hu324
      (param i32 i32)
      (result i32)
      (local.get 0)
      (local.get 1)
      (i32.add)
      (return)
  )

  (func $02$3e%no!t&hu3'24*32++2ao-eunth
      (param i32 i32)
      (result i32)
      (local.get 0)
      (local.get 1)
      (i32.add)
      (return)
  )

  (func $02$3e%no!t&hu3'24*32++2ao-eu//n<t>h?
      (param i32 i32)
      (result i32)
      (local.get 0)
      (local.get 1)
      (i32.add)
      (return)
  )

  (func $aenuth_ae`st|23~423
      (param i32 i32)
      (result i32)
      (local.get 0)
      (local.get 1)
      (i32.add)
      (return)
  )

  (func $bioi::..@@?^
      (param i32 i32)
      (result i32)
      (local.get 0)
      (local.get 1)
      (i32.add)
      (return)
  )
)

#assertFunction 0 [ i32 i32 ] -> [ i32 ] [ ] "simple function name"
#assertFunction 1 [ i32 i32 ] -> [ i32 ] [ ] "identifier function name 1"
#assertFunction 2 [ i32 i32 ] -> [ i32 ] [ ] "identifier function name 2"
#assertFunction 3 [ i32 i32 ] -> [ i32 ] [ ] "identifier function name 3"
#assertFunction 4 [ i32 i32 ] -> [ i32 ] [ ] "identifier function name 3"
#assertFunction 5 [ i32 i32 ] -> [ i32 ] [ ] "identifier function name 3"
#assertFunction 6 [ i32 i32 ] -> [ i32 ] [ ] "identifier function name 3"

#clearConfig
