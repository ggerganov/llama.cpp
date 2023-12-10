(* Copyright (C) 2017 Sio Kreuzer. All Rights Reserved. *)

let init () =
    Printf.printf "Initializing Game module...\n";
    flush stdout;;

let shutdown () =
    Printf.printf "Shutting down Game module...\n";
    flush stdout;;

let step () =
    (* Printf.printf "bla\n"; *)
    flush stdout;;

(* main/init *)
let () =
    Callback.register "init_fn" init;
    Callback.register "shutdown_fn" shutdown;
    Callback.register "step_fn" step;;
