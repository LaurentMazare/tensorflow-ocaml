open Ocamlbuild_plugin
let () =
  dispatch (function
    | Before_options ->
      let link_at_compile_time =
        try
          ignore (Sys.getenv "LINK_TF" : string);
          true
        with | _ -> false
      in
      if link_at_compile_time
      then
        flag ["ocaml"; "link"] (S [ A "-cclib"; A "-ltensorflow"; A "-cclib"; A "-L../lib2" ]);
      Options.use_ocamlfind := true
    | After_rules ->
      ocaml_lib "src/tensorflow"
    | _ -> ())
