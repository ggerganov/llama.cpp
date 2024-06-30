{
  perSystem =
    { config, lib, ... }:
    {
      apps =
        let
          inherit (config.packages) default;
          binaries = [
            "llama-cli"
            "llama-embedding"
            "llama-server"
            "llama-quantize"
            "llama-train-text-from-scratch"
          ];
          mkApp = name: {
            type = "app";
            program = "${default}/bin/${name}";
          };
        in
        lib.genAttrs binaries mkApp;
    };
}
