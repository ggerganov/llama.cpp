{
  perSystem =
    { config, lib, ... }:
    {
      apps =
        let
          inherit (config.packages) default;
          binaries = [
            "llama"
            "llama-embedding"
            "llama-server"
            "quantize"
            "train-text-from-scratch"
          ];
          mkApp = name: {
            type = "app";
            program = "${default}/bin/${name}";
          };
        in
        lib.genAttrs binaries mkApp;
    };
}
