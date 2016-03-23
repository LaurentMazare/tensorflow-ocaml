open Core.Std
exception Not_supported of string

let ops_file = "src/gen_ops/ops.pb"
let output_file = "src/graph/ops"
let do_not_generate_these_ops =
  String.Set.of_list
    [ "Const"
    ]

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
    ; type_name : string option
    }

  let caml_name t ~idx =
    match t.name with
    | Some "begin" -> "begin__"
    | Some "in" -> "in__"
    | Some name -> name
    | None -> sprintf "x%d" idx
end

module Attribute = struct
  type attr_type =
    | String
    | Shape
    | Int
    | Float
    | Bool
    | List of [ `float | `int | `shape | `type_ ]

  type t =
    { name : string
    ; attr_type : attr_type
    ; has_default_value : bool
    }

  let caml_name t =
    String.uncapitalize t.name

  let caml_type = function
    | String -> "string"
    | Shape -> "Dim.t list"
    | Int -> "int"
    | Float -> "float"
    | Bool -> "bool"
    | List `float -> "float list"
    | List `int -> "int list"
    | List `shape -> "Dim.t list list"
    | List `type_ -> "Type.p list"

  let of_dtype = function
    | "string" -> Some String
    | "shape" -> Some Shape
    | "int" -> Some Int
    | "float" -> Some Float
    | "bool" -> Some Bool
    | "list(float)" -> Some (List `float)
    | "list(int)" -> Some (List `int)
    | "list(shape)" -> Some (List `shape)
    | "list(type)" -> Some (List `type_)
    | _str -> None

  let constr caml_name = function
    | String -> "String " ^ caml_name
    | Shape -> "Shape " ^ caml_name
    | Int -> "Int " ^ caml_name
    | Float -> "Float " ^ caml_name
    | Bool -> "Bool " ^ caml_name
    | List `float -> "List (Float " ^ caml_name ^ ")"
    | List `int -> "List (Int " ^ caml_name ^ ")"
    | List `shape -> "List (Shape " ^ caml_name ^ ")"
    | List `type_ -> "List (Type " ^ caml_name ^ ")"

  let mli t =
    sprintf "%s%s:%s"
      (if t.has_default_value then "?" else "")
      (caml_name t)
      (caml_type t.attr_type)

  let ml_def t =
    sprintf "%s%s"
      (if t.has_default_value then "?" else "~")
      (caml_name t)

  let ml_apply t attribute_var =
    let caml_name = caml_name t in
    let updated_attributes =
      sprintf "(\"%s\", %s) :: %s"
        t.name
        (constr caml_name t.attr_type)
        attribute_var
    in
    if t.has_default_value
    then
      sprintf "match %s with | None -> %s | Some %s -> %s"
        caml_name
        attribute_var
        caml_name
        updated_attributes
    else updated_attributes
end

module Op = struct
  type t =
    { name : string
    ; inputs : Input.t list 
    ; output_type : Type.t
    ; output_type_name : string option
    ; attributes : Attribute.t list
    ; summary : string option
    ; description : string option
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

  let get_attr (attr : Op_def_piqi.Op_def_attr_def.t) =
    Option.bind attr.type_ Attribute.of_dtype
    |> Option.map ~f:(fun attr_type ->
      { Attribute.name = Option.value_exn attr.name
      ; attr_type
      ; has_default_value = Option.is_some attr.default_value
      })

  let create (op : Op_def_piqi.Op_def.t) =
    let name = Option.value_exn op.name in
    try
      let types = extract_types op.attr in
      let inputs =
        List.map op.input_arg ~f:(fun input_arg ->
          let type_name, type_ = read_type types input_arg in
          { Input.name = input_arg.name
          ; type_
          ; type_name
          })
      in
      let output_type_name, output_type =
        match op.output_arg with
        | [] -> None, Type.Fixed (P Unit)
        | _ :: _ :: _ -> raise (Not_supported "multiple outputs")
        | [ output_arg ] -> read_type types output_arg
      in
      Ok
        { name
        ; inputs
        ; output_type
        ; output_type_name
        ; attributes = List.filter_map op.attr ~f:get_attr
        ; summary = op.summary
        ; description = op.description
        }
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

let automatically_generated_file =
  "(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)"

let gen_mli ops =
  let out_channel = Out_channel.create (sprintf "%s.mli" output_file) in
  let p s =
    ksprintf (fun line ->
      Out_channel.output_string out_channel line;
      Out_channel.output_char out_channel '\n') s
  in
  let handle_one_op (op : Op.t) =
    let needs_variable_for_output_type = needs_variable_for_output_type op in
    Option.iter op.summary ~f:(fun summary -> p "(* %s *)" summary);
    Option.iter op.description ~f:(fun description -> p "(* %s *)" description);
    p "val %s" (Op.caml_name op);
    p "  :  ?name:string";
    if needs_variable_for_output_type
    then p "  -> type_ : %s Node.Type.t" (Type.to_string op.output_type);
    List.iter op.attributes ~f:(fun attribute ->
      p "  -> %s" (Attribute.mli attribute));
    List.iter op.inputs ~f:(fun { Input.name = _; type_; type_name = _ } ->
      p "  -> %s Node.t" (Type.to_string type_));
    if List.is_empty op.inputs
    then p "  -> unit";
    p "  -> %s Node.t" (Type.to_string op.output_type);
    p "";
  in
  p "%s" automatically_generated_file;
  p "open Node";
  p "";
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
    List.iter op.attributes ~f:(fun attribute ->
      p "    %s" (Attribute.ml_def attribute));
    List.iteri op.inputs ~f:(fun idx input ->
      let name = Input.caml_name input ~idx in
      p "    (%s : %s t)" name (Type.to_string input.type_));
    if List.is_empty op.inputs
    then p "    ()";
    let output_type_string = output_type_string op in
    p "  =";
    let type_attr =
      match op.output_type_name with
      | Some output_type_name ->
        sprintf " \"%s\", Type (P %s) " output_type_name output_type_string
      | None -> ""
    in
    p "  let attributes = [%s] in" type_attr;
    List.iter op.attributes ~f:(fun attribute ->
      p "  let attributes =";
      p "    %s" (Attribute.ml_apply attribute "attributes");
      p "  in";
    );
    p "  { name = Name.make_fresh ~name";
    p "  ; op_name = Op_name.of_string \"%s\"" op.name;
    p "  ; output_type = %s" output_type_string;
    let inputs =
      List.mapi op.inputs ~f:(fun idx input ->
        sprintf "P %s" (Input.caml_name input ~idx))
      |> String.concat ~sep:"; "
    in
    p "  ; inputs = [ %s ]" inputs;
    p "  ; attributes";
    p "  ; output_name = None";
    p "  }";
    p "";
  in
  p "%s" automatically_generated_file;
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
      | Ok op ->
          if Set.mem do_not_generate_these_ops op.name
          then None
          else Some op
      | Error err -> printf "Error %s\n" err; None)
  in
  gen_mli ops;
  gen_ml ops

let () = run ()
