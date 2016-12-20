open Ocamlbuild_plugin
let () =
  dispatch (function
    | Before_options ->
      flag ["ocaml"; "link"] (S [ A "-cclib"; A "-ltensorflow"; A "-cclib"; A "-L../lib" ]);
      Options.use_ocamlfind := true
    | After_rules ->
      dep ["c"; "compile"] [ "src/wrapper/c_api.h" ];
      pdep ["link"] "linkdep" (fun param -> [param]);
      rule "cstubs"
        ~prods:["src/wrapper/%_stubs.c"; "src/wrapper/%_generated.ml"]
        ~deps: ["src/wrapper/%_gen.byte"]
        (fun env build ->
          Cmd (A(env "src/wrapper/%_gen.byte")));
      ocaml_lib "src/tensorflow"
    | _ -> ())
