{ package, binaries }:

let
  default = builtins.elemAt binaries 0;
  mkApp = name: {
    ${name} = {
      type = "app";
      program = "${package}/bin/${name}";
    };
  };
  result = builtins.foldl' (acc: name: (mkApp name) // acc) { } binaries;
in

result // { default = result.${default}; }
