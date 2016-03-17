open Core.Std
exception Not_supported of string

let ops_file = "gen_ops/ops.pb"
let output_file = "graph/ops"

let value_exn = function
  | None -> raise (Not_supported "value_exn")
  | Some value -> value

type type_ =
  [ `float
  | `double
  ]

let types_to_string = function
  | `float -> "`float"
  | `double -> "`double"

module Type = struct
  type t =
    | Polymorphic of string * [ `allow_only of type_ list | `allow_all ]
    | Fixed of type_
    | Unit

  let to_string = function
    | Polymorphic (alpha, `allow_all) -> alpha
    | Polymorphic (alpha, `allow_only types) ->
      List.map types ~f:types_to_string
      |> String.concat ~sep:" | "
      |> fun types -> sprintf "([< %s ] as %s)" types alpha
    | Fixed type_ -> sprintf "[ %s ]" (types_to_string type_)
    | Unit -> "[ `unit ]"
end

module Input = struct
  type t =
    { name : string option
    ; type_ : Type.t
    }

  let name t ~idx =
    match t.name with
    | Some name -> name
    | None -> sprintf "x%d" idx
end

module Op = struct
  type t =
    { name : string
    ; inputs : Input.t list 
    ; output_type : Type.t
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
      begin
        match List.Assoc.find types type_attr with
        | None -> Type.Polymorphic (alpha, `allow_all)
        | Some types ->
          Polymorphic (alpha, `allow_only types)
      end
    | None ->
      match arg.type_ with
      | Some `dt_float -> Fixed `float
      | Some `dt_double -> Fixed `double
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
                | `dt_float -> Some `float
                | `dt_double -> Some `double
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
      let inputs =
        List.map op.input_arg ~f:(fun input_arg ->
          { Input.name = input_arg.name
          ; type_ = read_type types input_arg
          })
      in
      let output_type =
        match op.output_arg with
        | [] -> Type.Unit
        | _ :: _ :: _ -> raise (Not_supported "multiple outputs")
        | [ output_arg ] -> read_type types output_arg
      in
      Ok { name; inputs; output_type }
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
    List.iter op.inputs ~f:(fun { Input.name = _; type_ } ->
      p "  -> %s Node.t" (Type.to_string type_));
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
    p "let %s" (Op.caml_name op);
    p "    ?(name = \"%s\")" op.name;
    List.iteri op.inputs ~f:(fun idx input ->
      let name = Input.name input ~idx in
      p "    (%s : %s Node.t)" name (Type.to_string input.type_));
    p "  =";
    p "  Node";
    p "    { name = Name.make_fresh ~name";
    let output_type =
      match op.output_type with
      | Fixed `float -> "Type.(P Float)"
      | Fixed `double -> "Type.(P Double)"
      | Unit -> "Type.(P Unit)"
      | Polymorphic (alpha, _) ->
        List.find_map op.inputs ~f:(fun input ->
          match input.type_, input.name with
          | Polymorphic (alpha', _), Some name when alpha = alpha' -> Some name
          | _ -> None)
        |> function
        | Some input_name -> sprintf "%s.output_type" input_name
        | None -> "TODO: add a parameter"
    in
    p "    ; output_type = %s" output_type;
    let inputs =
      List.mapi op.inputs ~f:(fun idx input ->
        sprintf "P %s" (Input.name input ~idx))
      |> String.concat ~sep:"; "
    in
    p "    ; inputs = [ %s ]" inputs;
    (* TODO: adapt this... *)
    p "    ; attributes = []";
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
