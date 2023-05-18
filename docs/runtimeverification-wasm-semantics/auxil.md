Auxiliary Wasm Commands
=======================

Generally useful commands that are not part of the actual Wasm semantics.

```k
require "wasm.md"

module WASM-AUXIL
    imports WASM

    syntax Stmt ::= Auxil
 // ---------------------

    syntax Auxil ::= "#clearConfig"
 // -------------------------------
    rule <instrs> #clearConfig => . ...     </instrs>
         <curModIdx>         _ => .Int      </curModIdx>
         <valstack>          _ => .ValStack </valstack>
         <locals>            _ => .Map      </locals>
         <moduleInstances>   _ => .Bag      </moduleInstances>
         <moduleIds>         _ => .Map      </moduleIds>
         <nextModuleIdx>     _ => 0         </nextModuleIdx>
         <moduleRegistry>    _ => .Map      </moduleRegistry>
         <mainStore>
           <nextFuncAddr>    _ => 0         </nextFuncAddr>
           <funcs>           _ => .Bag      </funcs>
           <nextTabAddr>     _ => 0         </nextTabAddr>
           <tabs>            _ => .Bag      </tabs>
           <nextMemAddr>     _ => 0         </nextMemAddr>
           <mems>            _ => .Bag      </mems>
           <nextGlobAddr>    _ => 0         </nextGlobAddr>
           <globals>         _ => .Bag      </globals>
         </mainStore>

endmodule
```
