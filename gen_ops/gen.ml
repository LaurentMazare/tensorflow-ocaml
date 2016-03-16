exception Not_supported of string

let value_exn = function
  | None -> raise (Not_supported "value_exn")
  | Some value -> value

let ops_file = "gen_ops/ops.pb"
let output_file = "generated_ops"

let gen_mli ops =
  let out_channel = open_out (Printf.sprintf "%s.mli" output_file) in
  let handle_one_op (op : Op_def_piqi.Op_def.t) =
    let buffer = Buffer.create 128 in
    let p s =
      Printf.ksprintf (fun line ->
        Buffer.add_string buffer line;
        Buffer.add_char buffer '\n') s
    in
    let name = value_exn op.name in
    try
      let output_type =
        match op.output_arg with
        | [] -> "[ `unit ]"
        | _ :: _ :: _ -> raise (Not_supported "multiple outputs")
        | [ output_arg ] ->
          match output_arg.type_attr with
          | Some type_attr ->
              "'" ^ String.uncapitalize type_attr
          | None ->
              match output_arg.type_ with
              | Some `dt_float -> "[ `float ]"
              | Some `dt_double -> "[ `double ]"
              | Some _ -> raise (Not_supported "unknown output type")
              | None -> raise (Not_supported "no output type")
      in
      p "val %s" name;
      p "  :  ?name:string";
      p "  -> %s Node.t" output_type;
      p "";
      Buffer.output_buffer out_channel buffer
    with
    | Not_supported str ->
      Printf.printf "Error reading op %s: %s.\n%!" name str
  in
  List.iter handle_one_op ops;
  close_out out_channel

let gen_ml _ops = ()

let run () =
  let ops =
    open_in ops_file
    |> Piqirun.init_from_channel
    |> Op_def_piqi.parse_op_list
    |> fun op_list -> op_list.op
  in
  Printf.printf "Found %d ops.\n%!" (List.length ops);
  gen_mli ops;
  gen_ml ops

let () = run ()
