open Ocamlbuild_plugin
let () =
  dispatch (function
    | Before_options ->
      flag ["ocaml"; "link"] (S [ A "-cclib"; A "-ltensorflow"; A "-cclib"; A "-lpython2.7"; A "-cclib"; A "-L../lib" ]);
      Options.use_ocamlfind := true
    | _ -> ())
