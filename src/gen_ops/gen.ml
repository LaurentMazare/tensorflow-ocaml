open Base
open Stdio
open Tensorflow_core

exception Not_supported of string

let ops_file = "src/gen_ops/ops.pb"
let output_file = "src/graph/ops_generated"
let do_not_generate_these_ops = Set.of_list (module String) [ "Const" ]
let types_to_string type_ = "`" ^ String.uncapitalize (Operation.Type.to_string type_)

module Type = struct
  type t =
    | Polymorphic of string * [ `allow_only of Operation.Type.p list | `allow_all ]
    | Fixed of Operation.Type.p

  let to_string = function
    | Polymorphic (alpha, `allow_all) -> alpha
    | Polymorphic (alpha, `allow_only types) ->
      List.map types ~f:types_to_string
      |> String.concat ~sep:" | "
      |> fun types -> Printf.sprintf "([< %s ] as %s)" types alpha
    | Fixed type_ -> Printf.sprintf "[ %s ]" (types_to_string type_)
end

module Input = struct
  type t =
    { name : string
    ; type_ : Type.t
    ; type_name : string option
          (* When [number_attr] is present, the input is a list of tensors. *)
    ; number_attr : string option
    }

  let caml_name t =
    match t.name with
    | "begin" -> "begin__"
    | "end" -> "end__"
    | "in" -> "in__"
    | "inputs" -> "inputs__"
    | "method" -> "method__"
    | name -> name

  let caml_comp_name t =
    let name = caml_name t in
    if Option.equal String.equal t.number_attr None
    then name
    else Printf.sprintf "(List.hd_exn %s)" name
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
    ; match_input_length : Input.t option
    }

  let caml_name t =
    match String.uncapitalize t.name with
    | "method" -> "method_"
    | otherwise -> otherwise

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

  let mli t (p : ('a, unit, string, unit) format4 -> 'a) =
    match t.match_input_length with
    | None ->
      p
        "  -> %s%s:%s"
        (if t.has_default_value then "?" else "")
        (caml_name t)
        (caml_type t.attr_type)
    | Some _ -> ()

  let ml_def t (p : ('a, unit, string, unit) format4 -> 'a) =
    match t.match_input_length with
    | None -> p "    %s%s" (if t.has_default_value then "?" else "~") (caml_name t)
    | Some _ -> ()

  let ml_apply t attribute_var =
    let caml_name = caml_name t in
    let updated_attributes =
      let caml_name =
        match t.match_input_length with
        | None -> caml_name
        | Some input -> Printf.sprintf "(List.length %s)" (Input.caml_name input)
      in
      Printf.sprintf
        "(\"%s\", %s) :: %s"
        t.name
        (constr caml_name t.attr_type)
        attribute_var
    in
    if t.has_default_value && Option.is_none t.match_input_length
    then
      Printf.sprintf
        "match %s with | None -> %s | Some %s -> %s"
        caml_name
        attribute_var
        caml_name
        updated_attributes
    else updated_attributes
end

module Op = struct
  type output_type =
    { name : string option
    ; type_ : Type.t
    ; number_attr : string option
    }

  type t =
    { name : string
    ; inputs : Input.t list
    ; output_types : output_type list
    ; attributes : Attribute.t list
    ; summary : string option
    ; description : string option
    }

  let read_type types (arg : Op_def_piqi.op_def_arg_def) =
    match arg.type_attr with
    | Some type_attr ->
      let alpha =
        let type_attr = String.uncapitalize type_attr in
        if String.( = ) type_attr "type" then "'type__" else "'" ^ type_attr
      in
      let type_ =
        match List.Assoc.find ~equal:String.equal types type_attr with
        | None -> Type.Polymorphic (alpha, `allow_all)
        | Some types -> Polymorphic (alpha, `allow_only types)
      in
      Some type_attr, type_
    | None ->
      let raise_not_supported msg =
        let name = Option.value arg.name ~default:"unknown" in
        raise (Not_supported (Printf.sprintf "%s: %s" msg name))
      in
      (match arg.type_ with
      | None -> raise_not_supported "no input/output type"
      | Some dt_type ->
        (match Operation.Type.of_dt_type dt_type with
        | Some p -> None, Fixed p
        | None -> raise_not_supported "unknown input/output type"))

  let extract_types (attrs : Op_def_piqi.op_def_attr_def list) =
    List.filter_map attrs ~f:(fun (attr : Op_def_piqi.op_def_attr_def) ->
        match attr.name, attr.type_ with
        | Some name, Some "type" ->
          let allowed_values =
            match attr.allowed_values with
            | None -> []
            | Some allowed_values ->
              (match allowed_values.list with
              | None -> []
              | Some allowed_values ->
                List.filter_map allowed_values.type_ ~f:Operation.Type.of_dt_type)
          in
          if List.is_empty allowed_values then None else Some (name, allowed_values)
        | _ -> None)

  let get_attr (attr : Op_def_piqi.Op_def_attr_def.t) ~inputs =
    Option.bind attr.type_ ~f:Attribute.of_dtype
    |> Option.map ~f:(fun attr_type ->
           let name = Option.value_exn attr.name in
           let match_input_length =
             List.find inputs ~f:(fun (input : Input.t) ->
                 match input.number_attr with
                 | Some number when String.( = ) number name -> true
                 | _ -> false)
           in
           { Attribute.name
           ; attr_type
           ; has_default_value = Option.is_some attr.default_value
           ; match_input_length
           })

  let create (op : Op_def_piqi.Op_def.t) =
    let name = Option.value_exn op.name in
    try
      let types = extract_types op.attr in
      let inputs =
        List.mapi op.input_arg ~f:(fun idx input_arg ->
            let type_name, type_ = read_type types input_arg in
            let name =
              match input_arg.name with
              | None -> Printf.sprintf "x%d" idx
              | Some name -> name
            in
            { Input.name; type_; type_name; number_attr = input_arg.number_attr })
      in
      let output_types =
        let output_types =
          List.map op.output_arg ~f:(fun output_arg ->
              let name, type_ = read_type types output_arg in
              { name; type_; number_attr = output_arg.number_attr })
        in
        match output_types with
        | [] -> [ { name = None; type_ = Type.Fixed (P Unit); number_attr = None } ]
        | output_types -> output_types
      in
      let has_output_list =
        List.exists output_types ~f:(fun output_type ->
            Option.is_some output_type.number_attr)
      in
      if has_output_list && 1 < List.length output_types
      then raise (Not_supported "output list are only supported for a single output");
      Ok
        { name
        ; inputs
        ; output_types
        ; attributes = List.filter_map op.attr ~f:(get_attr ~inputs)
        ; summary = op.summary
        ; description = op.description
        }
    with
    | Not_supported str -> Error (Printf.sprintf "%s: %s" name str)

  let caml_name t =
    match t.name with
    | "Mod" -> "mod_"
    | otherwise -> String.uncapitalize otherwise
end

let same_input_and_output_type (op : Op.t) ~alpha =
  List.find_map op.inputs ~f:(fun input ->
      match input.type_ with
      | Polymorphic (alpha', _) when String.( = ) alpha alpha' -> Some input
      | _ -> None)

let type_variable ~idx = if idx = 0 then "type_" else Printf.sprintf "type_%d" idx

let output_type_string op output_type ~idx =
  match (output_type : Type.t) with
  | Fixed p -> "Type." ^ Operation.Type.to_string p
  | Polymorphic (alpha, _) ->
    (match same_input_and_output_type op ~alpha with
    | Some input -> Printf.sprintf "(Node.output_type %s)" (Input.caml_comp_name input)
    | None -> type_variable ~idx)

let needs_variable_for_output_type op output_type =
  match (output_type : Type.t) with
  | Fixed _ -> false
  | Polymorphic (alpha, _) -> same_input_and_output_type op ~alpha |> Option.is_none

let automatically_generated_file =
  "(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)"

let p out_channel s =
  Printf.ksprintf
    (fun line ->
      Out_channel.output_string out_channel line;
      Out_channel.output_char out_channel '\n')
    s

let escape_comment s =
  String.substr_replace_all s ~pattern:"{|" ~with_:"{ |"
  |> String.substr_replace_all ~pattern:"|}" ~with_:"| }"
  |> String.tr ~target:'"' ~replacement:'\''

let gen_mli ops =
  let out_channel = Out_channel.create (Printf.sprintf "%s.mli" output_file) in
  let p s = p out_channel s in
  let handle_one_op (op : Op.t) =
    Option.iter op.summary ~f:(fun summary -> p "(* %s *)" (escape_comment summary));
    Option.iter op.description ~f:(fun description ->
        p "(* %s *)" (escape_comment description));
    p "val %s" (Op.caml_name op);
    p "  :  ?name:string";
    List.iteri op.output_types ~f:(fun idx output_type ->
        if needs_variable_for_output_type op output_type.type_
        then
          p "  -> %s:%s Type.t" (type_variable ~idx) (Type.to_string output_type.type_));
    List.iter op.attributes ~f:(fun attribute -> Attribute.mli attribute p);
    p "  -> ?control_inputs:Node.p list";
    List.iter op.inputs ~f:(fun input ->
        let maybe_list = if Option.is_some input.number_attr then " list" else "" in
        p "  -> %s t%s" (Type.to_string input.type_) maybe_list);
    if List.is_empty op.inputs then p "  -> unit";
    p
      "  -> %s"
      (List.map op.output_types ~f:(fun output_type ->
           Printf.sprintf
             "%s t%s"
             (Type.to_string output_type.type_)
             (if Option.is_some output_type.number_attr then " list" else ""))
      |> String.concat ~sep:" * ");
    p ""
  in
  p "%s" automatically_generated_file;
  p "open Node";
  p "";
  p "module Op_names : sig";
  List.iter ops ~f:(fun op -> p "  val %s : Op_name.t" (Op.caml_name op));
  p "end";
  p "";
  List.iter ops ~f:handle_one_op;
  Out_channel.close out_channel

let handle_one_op (op : Op.t) out_channel =
  let p s = p out_channel s in
  p "let %s" (Op.caml_name op);
  p "    ?(name = \"%s\")" op.name;
  List.iteri op.output_types ~f:(fun idx output_type ->
      if needs_variable_for_output_type op output_type.type_
      then p "    ~%s" (type_variable ~idx));
  List.iter op.attributes ~f:(fun attribute -> Attribute.ml_def attribute p);
  p "    ?(control_inputs = [])";
  List.iter op.inputs ~f:(fun input ->
      let name = Input.caml_name input in
      let maybe_list = if Option.is_some input.number_attr then " list" else "" in
      p "    (%s : %s t%s)" name (Type.to_string input.type_) maybe_list);
  if List.is_empty op.inputs then p "    ()";
  p "  =";
  let type_attr =
    List.filter_mapi op.output_types ~f:(fun idx output_type ->
        Option.map output_type.name ~f:(fun output_type_name ->
            output_type_name, output_type_string op output_type.type_ ~idx))
  in
  let type_attr =
    List.fold op.inputs ~init:type_attr ~f:(fun acc (input : Input.t) ->
        match input.type_name with
        | None -> acc
        | Some type_name when List.Assoc.mem ~equal:String.equal acc type_name -> acc
        | Some type_name ->
          let name = Input.caml_comp_name input in
          (type_name, Printf.sprintf "(Node.output_type %s)" name) :: acc)
    |> List.map ~f:(fun (type_name, type_string) ->
           Printf.sprintf " \"%s\", Type (P %s) " type_name type_string)
    |> String.concat ~sep:"; "
  in
  p "  let attributes = [%s] in" type_attr;
  List.iter op.attributes ~f:(fun attribute ->
      p "  let attributes =";
      p "    %s" (Attribute.ml_apply attribute "attributes");
      p "  in");
  p "  let name = Name.of_string name in";
  p "  let op_name = Op_names.%s in" (Op.caml_name op);
  let inputs =
    List.map op.inputs ~f:(fun input ->
        if Option.is_none input.number_attr
        then Printf.sprintf "(`single (P %s))" (Input.caml_name input)
        else
          Printf.sprintf
            "(`multi (List.map ~f:(fun n -> P n) %s))"
            (Input.caml_name input))
    |> String.concat ~sep:"; "
    |> Printf.sprintf "[ %s ]"
  in
  p "  let inputs = %s in" inputs;
  let multiple_outputs = 1 < List.length op.output_types in
  (match op.output_types with
  | [ ({ number_attr = Some number_attr; _ } as output_type) ] ->
    let output_type_string = output_type_string op output_type.type_ ~idx:0 in
    let number_attr =
      List.find_exn op.attributes ~f:(fun attr -> String.( = ) attr.name number_attr)
    in
    let number_value =
      match number_attr.match_input_length with
      | Some input -> Printf.sprintf "(List.length %s)" input.name
      | None -> number_attr.name
    in
    p "  let node =";
    p "    Node.create";
    p "      ~name";
    p "      ~op_name";
    p "      ~output_type:%s" output_type_string;
    p "      ~inputs";
    p "      ~control_inputs";
    p "      ~attributes";
    p "      ~output_idx:None";
    p "  in";
    p "  List.init %s ~f:(fun output_idx ->" number_value;
    p "    set_output_idx node (Some output_idx))"
  | output_types ->
    List.iteri output_types ~f:(fun idx output_type ->
        let output_type_string = output_type_string op output_type.type_ ~idx in
        if 0 < idx then p "  ,";
        p "  Node.create";
        p "    ~name";
        p "    ~op_name";
        p "    ~output_type:%s" output_type_string;
        p "    ~inputs";
        p "    ~control_inputs";
        p "    ~attributes";
        p
          "    ~output_idx:%s"
          (if multiple_outputs then Printf.sprintf "(Some %d)" idx else "None")));
  p ""

let gen_ml ops =
  let out_channel = Out_channel.create (Printf.sprintf "%s.ml" output_file) in
  let p s = p out_channel s in
  p "%s" automatically_generated_file;
  p "open Base";
  p "open Tensorflow_core.Operation";
  p "open Node";
  p "";
  p "module Op_names = struct";
  List.iter ops ~f:(fun op ->
      p "  let %s = Op_name.of_string \"%s\"" (Op.caml_name op) op.name);
  p "end";
  p "";
  List.iter ops ~f:(fun op -> handle_one_op op out_channel);
  Out_channel.close out_channel

let run () =
  let ops =
    Stdio.In_channel.create ops_file
    |> Piqirun.init_from_channel
    |> Op_def_piqi.parse_op_list
    |> fun op_list -> op_list.op
  in
  printf "Found %d ops.\n%!" (List.length ops);
  let ops =
    List.filter_map ops ~f:(fun op ->
        match Op.create op with
        | Ok op -> if Set.mem do_not_generate_these_ops op.name then None else Some op
        | Error err ->
          printf "Error %s\n" err;
          None)
  in
  gen_mli ops;
  gen_ml ops

let () = run ()
