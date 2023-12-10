(* Copyright (C) 2017 Sio Kreuzer. All Rights Reserved. *)

let init () =
    Printf.printf "Initializing Game module...\n";
    flush stdout;;

let shutdown () =
    Printf.printf "Shutting down Game module...\n";
    flush stdout;;

let step (s:string) : string =
    Printf.printf "Hello Ocaml from LLama\n";
    flush stdout;
    s ^ "Hello Ocaml\n";;




(* main/init *)
let () =
    Callback.register "init_fn" init;
    Callback.register "shutdown_fn" shutdown;
    Callback.register "step_fn" step;;
