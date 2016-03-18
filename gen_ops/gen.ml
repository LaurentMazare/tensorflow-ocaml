open Core.Std
exception Not_supported of string

let ops_file = "gen_ops/ops.pb"
let output_file = "src/ops"

let types_to_string type_ =
  "`" ^ String.uncapitalize (Node.Type.to_string type_)

module Type = struct
  type t =
    | Polymorphic of string * [ `allow_only of Node.Type.p list | `allow_all ]
    | Fixed of Node.Type.p

  let to_string = function
    | Polymorphic (alpha, `allow_all) -> alpha
    | Polymorphic (alpha, `allow_only types) ->
      List.map types ~f:types_to_string
      |> String.concat ~sep:" | "
      |> fun types -> sprintf "([< %s ] as %s)" types alpha
    | Fixed type_ -> sprintf "[ %s ]" (types_to_string type_)
end

module Input = struct
  type t =
    { name : string option
    ; type_ : Type.t
    }

  let name t ~idx =
    match t.name with
    | Some "begin" -> "begin__"
    | Some "in" -> "in__"
    | Some name -> name
    | None -> sprintf "x%d" idx
end

module Op = struct
  type t =
    { name : string
    ; inputs : Input.t list 
    ; output_type : Type.t
    ; output_type_name : string option
    }

  let read_type types (arg : Op_def_piqi.op_def_arg_def) =
    match arg.type_attr with
    | Some type_attr ->
      let alpha =
        let type_attr = String.uncapitalize type_attr in
        if type_attr = "type"
        then "'type__"
        else "'" ^ type_attr
      in
      let type_ =
        match List.Assoc.find types type_attr with
        | None -> Type.Polymorphic (alpha, `allow_all)
        | Some types ->
          Polymorphic (alpha, `allow_only types)
      in
      Some type_attr, type_
    | None ->
      let raise_not_supported msg =
        let name = Option.value arg.name ~default:"unknown" in
        raise (Not_supported (Printf.sprintf "%s: %s" msg name))
      in
      match arg.type_ with
      | None -> raise_not_supported "no input/output type"
      | Some dt_type ->
        match Node.Type.of_dt_type dt_type with
        | Some p -> None, Fixed p
        | None -> raise_not_supported "unknown input/output type"

  let extract_types (attrs : Op_def_piqi.op_def_attr_def list) =
    List.filter_map attrs ~f:(fun (attr : Op_def_piqi.op_def_attr_def) ->
      match attr.name, attr.type_ with
      | Some name, Some "type" ->
        let allowed_values =
          match attr.allowed_values with
          | None -> []
          | Some allowed_values ->
            match allowed_values.list with
            | None -> []
            | Some allowed_values ->
              List.filter_map allowed_values.type_ ~f:Node.Type.of_dt_type
        in
        if allowed_values = []
        then None
        else Some (name, allowed_values)
      | _ -> None)

  let create (op : Op_def_piqi.Op_def.t) =
    let name = Option.value_exn op.name in
    try
      let types = extract_types op.attr in
      let inputs =
        List.map op.input_arg ~f:(fun input_arg ->
          { Input.name = input_arg.name
          ; type_ = read_type types input_arg |> snd
          })
      in
      let output_type_name, output_type =
        match op.output_arg with
        | [] -> None, Type.Fixed (P Unit)
        | _ :: _ :: _ -> raise (Not_supported "multiple outputs")
        | [ output_arg ] -> read_type types output_arg
      in
      Ok { name; inputs; output_type; output_type_name }
    with
    | Not_supported str ->
      Error (sprintf "%s: %s" name str)

  let caml_name t =
    match t.name with
    | "Mod" -> "mod_"
    | otherwise -> String.uncapitalize otherwise
end

let same_input_and_output_type (op : Op.t) ~alpha =
  List.find_map op.inputs ~f:(fun input ->
    match input.type_, input.name with
    | Polymorphic (alpha', _), Some name when alpha = alpha' -> Some name
    | _ -> None)

let output_type_string op =
  match op.Op.output_type with
  | Fixed p -> "Type." ^ Node.Type.to_string p
  | Polymorphic (alpha, _) ->
    match same_input_and_output_type op ~alpha with
    | Some input_name -> sprintf "%s.output_type" input_name
    | None -> "type_"

let needs_variable_for_output_type op =
  match op.Op.output_type with
  | Fixed _ -> false
  | Polymorphic (alpha, _) ->
    same_input_and_output_type op ~alpha |> Option.is_none

let gen_mli ops =
  let out_channel = Out_channel.create (sprintf "%s.mli" output_file) in
  let p s =
    ksprintf (fun line ->
      Out_channel.output_string out_channel line;
      Out_channel.output_char out_channel '\n') s
  in
  let handle_one_op (op : Op.t) =
    let needs_variable_for_output_type = needs_variable_for_output_type op in
    p "val %s" (Op.caml_name op);
    p "  :  ?name:string";
    if needs_variable_for_output_type
    then p "  -> type_ : %s Node.Type.t" (Type.to_string op.output_type);
    List.iter op.inputs ~f:(fun { Input.name = _; type_ } ->
      p "  -> %s Node.t" (Type.to_string type_));
    if List.is_empty op.inputs
    then p "  -> unit";
    p "  -> %s Node.t" (Type.to_string op.output_type);
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
    let needs_variable_for_output_type = needs_variable_for_output_type op in
    p "let %s" (Op.caml_name op);
    p "    ?(name = \"%s\")" op.name;
    if needs_variable_for_output_type
    then p "    ~type_";
    List.iteri op.inputs ~f:(fun idx input ->
      let name = Input.name input ~idx in
      p "    (%s : %s t)" name (Type.to_string input.type_));
    if List.is_empty op.inputs
    then p "    ()";
    let output_type_string = output_type_string op in
    p "  =";
    p "  { name = Name.make_fresh ~name";
    p "  ; op_name = \"%s\"" op.name;
    p "  ; output_type = %s" output_type_string;
    let inputs =
      List.mapi op.inputs ~f:(fun idx input ->
        sprintf "P %s" (Input.name input ~idx))
      |> String.concat ~sep:"; "
    in
    p "  ; inputs = [ %s ]" inputs;
    (* TODO: handle more attributes... *)
    p "  ; attributes = [";
    Option.iter op.output_type_name ~f:(fun output_type_name ->
      p "      \"%s\", Type (P %s);" output_type_name output_type_string);
    p "    ]";
    p "  }";
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
