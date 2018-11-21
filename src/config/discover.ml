module C = Configurator.V1

let () =
  let tensorflow_flags =
    match Sys.getenv_opt "LIBTENSORFLOW" with
    | None -> []
    | Some lib_dir ->
      [ Printf.sprintf "-Wl,-R%s" lib_dir
      ; Printf.sprintf "-L%s" lib_dir
      ]
  in
  C.main ~name:"tensorflow-config" (fun _c ->
      C.Flags.write_sexp "c_library_flags.sexp" tensorflow_flags)
