open Ocamlbuild_plugin
let () =
  dispatch (function
    | Before_options ->
      Options.use_ocamlfind := true
    | After_rules ->
      Pathname.define_context "src/fnn" [ "src/graph"; "src/fnn" ];
      dep ["c"; "compile"] [ "src/wrapper/c_api.h" ];
      rule "cstubs"
        ~prods:["src/wrapper/%_stubs.c"; "src/wrapper/%_generated.ml"]
        ~deps: ["src/wrapper/%_gen.byte"]
        (fun env build ->
          Cmd (A(env "src/wrapper/%_gen.byte")));
      ocaml_lib "tensorflow";
      ocaml_lib "tensorflow_core";
      dep ["link"; "ocaml"; "use_tensorflowcstubs"]
        ["libtensorflowcstubs.a"];
      flag ["link"; "ocaml"; "use_tensorflowcstubs"]
        (S[A"-cclib"; A"-ltensorflowcstubs"; A"-cclib"; A"-ltensorflow"]);
      flag ["link"; "ocaml"; "use_tensorflow"]
        (S[A"-ccopt"; A"-L."; A"-ccopt"; A"-L../lib"; A"-cclib"; A"-ltensorflow"]);
      flag ["link"; "ocaml"; "use_tensorflow_core"]
        (S[A"-ccopt"; A"-L."; A"-ccopt"; A"-L../lib"; A"-cclib"; A"-ltensorflow"]);
    | _ -> ())
