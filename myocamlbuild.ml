open Ocamlbuild_plugin
let () =
  dispatch (function
    | Before_options ->
      Options.use_ocamlfind := true
    | After_rules ->
      dep ["c"; "compile"] [ "src/wrapper/c_api.h" ];
      rule "cstubs"
        ~prods:["src/wrapper/%_stubs.c"; "src/wrapper/%_generated.ml"]
        ~deps: ["src/wrapper/%_gen.byte"]
        (fun env build ->
          Cmd (A(env "src/wrapper/%_gen.byte")));
      ocaml_lib "tensorflow";
      dep ["link"; "ocaml"; "use_tensorflowcstubs"] ["libtensorflowcstubs.a"];
      flag ["link"; "ocaml"; "use_tensorflow"]
        (S[A"-ccopt"; A"-L../lib"; A"-cclib"; A"-ltensorflow"]);
    | _ -> ())
