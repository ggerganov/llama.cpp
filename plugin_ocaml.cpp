#include "plugin_ocaml.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <assert.h>
//#include <GL_headers.hpp>
//#include <SDL_ttf.h>
//#include <SDL_image.h>
//#include <OCaml.hpp>
#include <assert.h>
#define Assert assert

#include<caml/address_class.h>
#include<caml/alloc.h>
#include<caml/backtrace_prim.h>
#include<caml/backtrace.h>
#include<caml/bigarray.h>
#include<caml/callback.h> // this one's the big important one for embedding OCaml
//#include<caml/compact.h>
#include<caml/compare.h>
//#include<caml/compatibility.h>
#include<caml/config.h>
#include<caml/custom.h>
#include<caml/debugger.h>
#include<caml/dynlink.h>
#include<caml/exec.h>
#include<caml/fail.h>
#include<caml/finalise.h>
#include<caml/fix_code.h>
//#include<caml/freelist.h>
#include<caml/gc_ctrl.h>
#include<caml/gc.h>
#include<caml/globroots.h>
#include<caml/hash.h>
#include<caml/hooks.h>
#include<caml/instrtrace.h>
#include<caml/instruct.h>
//#include<caml/int64_emul.h>
//#include<caml/int64_format.h>
//#include<caml/int64_native.h>
#include<caml/interp.h>
#include<caml/intext.h>
#include<caml/io.h>
//#include<caml/jumptbl.h> // gives compile errors
#include<caml/major_gc.h>
#include<caml/md5.h>
#include<caml/memory.h>
#include<caml/minor_gc.h>
#include<caml/misc.h>
#include<caml/mlvalues.h>
#include<caml/osdeps.h>
#include<caml/prims.h>
#include<caml/printexc.h>
#include<caml/reverse.h>
#include<caml/roots.h>
//#include<caml/signals_machdep.h>
#include<caml/signals.h>
#include<caml/socketaddr.h>
//#include<caml/spacetime.h>
#include<caml/stack.h>
//#include<caml/stacks.h>
#include<caml/startup_aux.h>
#include<caml/startup.h>
#include<caml/sys.h>
#include<caml/threads.h>
//#include<caml/ui.h>
#include<caml/unixsupport.h>
#include<caml/version.h>
#include<caml/weak.h>

void OCaml_shutdown()
{
  //caml_shutdown(); // This function exists, but is not exported by the OCaml runtime libraries.
}

std::string process_output_plugin_ocaml(const std::string start,
				  const std::string state,
				  const std::string input) {

    auto step_fn = caml_named_value( "step_fn" );
    assert( step_fn );
    value ocamlString = caml_copy_string(input.c_str());

  //
    
    value result= caml_callback( *step_fn, ocamlString );
    std::string resultString = String_val(result);
    return resultString;
    
}

void process_output_plugin_ocaml_init()
{
    printf( "Linked against OCaml version %s\n", OCAML_VERSION_STRING );
    const char *argv[] = {"llamacpp", NULL };
    caml_startup( argv );
}

void process_output_plugin_ocaml_destroy()
{
  OCaml_shutdown();
}
