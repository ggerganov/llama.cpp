{
  perSystem =
    { config, lib, ... }:
    {
      apps =
        let
          inherit (config.packages) default;
          binaries = [
            "jarvis-cli"
            "jarvis-embedding"
            "jarvis-server"
            "jarvis-quantize"
          ];
          mkApp = name: {
            type = "app";
            program = "${default}/bin/${name}";
          };
        in
        lib.genAttrs binaries mkApp;
    };
}
