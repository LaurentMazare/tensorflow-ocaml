open Core.Std
exception Not_supported of string

let ops_file = "gen_ops/ops.pb"
let output_file = "generated_ops"

let value_exn = function
  | None -> raise (Not_supported "value_exn")
  | Some value -> value

module Op = struct
  type t =
    { name : string
    ; input_types : string list 
    ; output_type : string
    }

  let read_type types (arg : Op_def_piqi.op_def_arg_def) =
    match arg.type_attr with
    | Some type_attr ->
      let alpha =
        let type_attr = String.uncapitalize type_attr in
        if type_attr = "type"
        then "'type'"
        else "'" ^ type_attr
      in
      begin
        match List.Assoc.find types type_attr with
        | None -> alpha
        | Some types ->
          String.concat types ~sep:" | "
          |> fun types -> sprintf "([< %s ] as %s)" types alpha
      end
    | None ->
      match arg.type_ with
      | Some `dt_float -> "[ `float ]"
      | Some `dt_double -> "[ `double ]"
      | Some _ -> raise (Not_supported "unknown output type")
      | None -> raise (Not_supported "no output type")

  let extract_types (attrs : Op_def_piqi.op_def_attr_def list) =
    List.filter_map attrs (fun (attr : Op_def_piqi.op_def_attr_def) ->
      match attr.name, attr.type_ with
      | Some name, Some "type" ->
        let allowed_values =
          match attr.allowed_values with
          | None -> []
          | Some allowed_values ->
            match allowed_values.list with
            | None -> []
            | Some allowed_values ->
              List.filter_map allowed_values.type_ (fun typ ->
                match typ with
                | `dt_float -> Some "`float"
                | `dt_double -> Some "`double"
                | _ -> None)
        in
        if allowed_values = []
        then None
        else Some (name, allowed_values)
      | _ -> None)

  let create (op : Op_def_piqi.Op_def.t) =
    let name = value_exn op.name in
    try
      let types = extract_types op.attr in
      let input_types = List.map op.input_arg ~f:(read_type types) in
      let output_type =
        match op.output_arg with
        | [] -> "[ `unit ]"
        | _ :: _ :: _ -> raise (Not_supported "multiple outputs")
        | [ output_arg ] -> read_type types output_arg
      in
      Ok { name; input_types; output_type }
    with
    | Not_supported str ->
      Error (sprintf "%s: %s" name str)

  let caml_name t =
    match t.name with
    | "Mod" -> "mod_"
    | otherwise -> String.uncapitalize otherwise
end

let gen_mli ops =
  let out_channel = Out_channel.create (sprintf "%s.mli" output_file) in
  let p s =
    ksprintf (fun line ->
      Out_channel.output_string out_channel line;
      Out_channel.output_char out_channel '\n') s
  in
  let handle_one_op (op : Op.t) =
    p "val %s" (Op.caml_name op);
    p "  :  ?name:string";
    List.iter op.input_types ~f:(fun typ -> p "  -> %s Node.t" typ);
    p "  -> %s Node.t" op.output_type;
    p "";
  in
  List.iter ops ~f:handle_one_op;
  Out_channel.close out_channel

let gen_ml ops =
  let out_channel = Out_channel.create (sprintf "%s.ml" output_file) in
  let p s =
    ksprintf (fun line ->
      Out_channel.output_string out_channel line;
      Out_channel.output_char out_channel '\n') s
  in
  let handle_one_op (op : Op.t) =
    p "let %s" (Op.caml_name op);
    p "    ?(name = \"%s\")" op.name;
    List.iteri op.input_types ~f:(fun i typ -> p "    (x%d : %s Node.t)" i typ);
    p "  =";
    p "  Node";
    p "    { name = Name.make_fresh ~name";
    (* TODO: adapt these... *)
    p "    ; output_type = x.output_type";
    p "    ; inputs = [ P x ]";
    p "    ; attributes = [ \"T\", Type (P x.output_type) ]";
    p "    }";
    p "";
  in
  p "open Node";
  p "";
  List.iter ops ~f:handle_one_op;
  Out_channel.close out_channel

let run () =
  let ops =
    open_in ops_file
    |> Piqirun.init_from_channel
    |> Op_def_piqi.parse_op_list
    |> fun op_list -> op_list.op
  in
  printf "Found %d ops.\n%!" (List.length ops);
  let ops =
    List.filter_map ops ~f:(fun op ->
      match Op.create op with
      | Ok op -> Some op
      | Error err -> printf "Error %s\n" err; None)
  in
  gen_mli ops;
  gen_ml ops

let () = run ()
