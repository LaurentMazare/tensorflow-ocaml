type t = string

let of_string x = x
let to_string x = x

open Node
open Graph_piqi

let create_attr_value
    ?s
    ?i
    ?f
    ?b
    ?type_
    ?shape
    ?tensor
    ?list
    ?func
    ?placeholder
    ()
  =
  { Attr_value.s
  ;  i
  ;  f
  ;  b
  ;  type_
  ;  shape
  ;  tensor
  ;  list
  ;  func
  ;  placeholder
  }

let default_tensor_proto = default_tensor_proto ()

let tensor_attr ?(float_val = []) ?(double_val = []) ?(int_val = []) ?(int64_val = []) ~shape output_type =
  let tensor =
    { default_tensor_proto with
      dtype = Some (Node.Type.to_dt_type (P output_type))
    ; float_val
    ; double_val
    ; int_val
    ; int64_val
    ; tensor_shape =
      Some
        { dim =
          List.map
          (fun d -> { Tensor_shape_proto_dim.size = Some (Int64.of_int d); name = None })
          shape
        ; unknown_rank = None
        }
    }
  in
  create_attr_value ~tensor ()

let of_attribute (type a) name value (output_type : a Node.Type.t) =
  let value =
    match value with
    | String s -> Some (create_attr_value ~s ())
    | Int i -> Some (create_attr_value ~i:(Int64.of_int i) ())
    | Float f -> Some (create_attr_value ~f ())
    | Bool b -> Some (create_attr_value ~b ())
    | Type type_ ->
      Some (create_attr_value ~type_:(Type.to_dt_type type_) ())
    | Shape shape ->
      let unknown_rank =
        match shape with
        | [] -> Some true
        | _ :: _ -> None
      in
      let dim =
        List.map
          (fun { Node.Dim.size; name } ->
            { Tensor_shape_proto_dim.size = Some (Int64.of_int size)
            ; name
            })
          shape
      in
      let shape =
        { Tensor_shape_proto.dim
        ; unknown_rank
        }
      in
      Some (create_attr_value ~shape ())
    | Tensor_float tensor ->
      let tensor_attr =
        match output_type with
        | Node.Type.Float ->
          tensor_attr ~float_val:tensor.values ~shape:tensor.shape output_type
        | Node.Type.Double ->
          tensor_attr ~double_val:tensor.values ~shape:tensor.shape output_type
        | _ -> tensor_attr ~shape:tensor.shape output_type
      in
      Some tensor_attr
    | Tensor_int tensor ->
      let tensor_attr =
        match output_type with
        | Node.Type.Int32 ->
          let int_val = List.map Int32.of_int tensor.values in
          tensor_attr ~int_val ~shape:tensor.shape output_type
        | Node.Type.Int64 ->
          let int64_val = List.map Int64.of_int tensor.values in
          tensor_attr ~int64_val ~shape:tensor.shape output_type
        | _ -> tensor_attr ~shape:tensor.shape output_type
      in
      Some tensor_attr
    (* TODO *)
    | List _ -> None
  in
  { Node_def_attr_entry.key = Some name
  ; value
  }

let of_nodes ts =
  let nodes = Hashtbl.create 128 in
  let rec walk p =
    let P t = p in
    if Hashtbl.mem nodes t.name
    then ()
    else begin
      let attr = List.map (fun (name, value) -> of_attribute name value t.output_type) t.attributes in
      let node =
        { Node_def.name = Some (Node.Name.to_string t.name)
        ; op = Some t.op_name
        ; input = List.map (fun (P input) -> Node.Name.to_string input.name) t.inputs
        ; device = None
        ; attr
        }
      in
      Hashtbl.add nodes t.name node;
      List.iter walk t.inputs
    end
  in
  List.iter walk ts;
  let nodes = Hashtbl.fold (fun _ v acc -> v :: acc) nodes [] in
  let graph_def =
    { Graph_def.node = nodes
    ; versions =
      Some
        { Version_def.producer = Some (Int32.of_int 8)
        ; min_consumer = None
        ; bad_consumers = []
        }
    ; version = None
    ; library = None
    }
  in
  gen_graph_def graph_def
  |> Piqirun.to_string

let of_node t = of_nodes [ P t ]

let read_file filename =
  let input_channel = open_in filename in
  let size = in_channel_length input_channel in
  let content = Bytes.create size in
  really_input input_channel content 0 size;
  close_in input_channel;
  content
