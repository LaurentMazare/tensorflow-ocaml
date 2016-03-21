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

let create_attr_value_list_value
    ?(s = [])
    ?(i = [])
    ?(f = [])
    ?(b = [])
    ?(type_ = [])
    ?(shape = [])
    ?(tensor = [])
    ()
  =
  { Attr_value_list_value.s; i; f; b; type_; shape; tensor }

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

let shape_attr shape =
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
  { Tensor_shape_proto.dim
  ; unknown_rank
  }

let of_attribute (type a) name value (output_type : a Node.Type.t) =
  let value =
    match value with
    | String s -> create_attr_value ~s ()
    | Int i -> create_attr_value ~i:(Int64.of_int i) ()
    | Float f -> create_attr_value ~f ()
    | Bool b -> create_attr_value ~b ()
    | Type type_ ->
      create_attr_value ~type_:(Type.to_dt_type type_) ()
    | Shape shape ->
      create_attr_value ~shape:(shape_attr shape) ()
    | Tensor_float tensor ->
      begin
        match output_type with
        | Node.Type.Float ->
          tensor_attr ~float_val:tensor.values ~shape:tensor.shape output_type
        | Node.Type.Double ->
          tensor_attr ~double_val:tensor.values ~shape:tensor.shape output_type
        | _ -> tensor_attr ~shape:tensor.shape output_type
      end
    | Tensor_int tensor ->
      begin
        match output_type with
        | Node.Type.Int32 ->
          let int_val = List.map Int32.of_int tensor.values in
          tensor_attr ~int_val ~shape:tensor.shape output_type
        | Node.Type.Int64 ->
          let int64_val = List.map Int64.of_int tensor.values in
          tensor_attr ~int64_val ~shape:tensor.shape output_type
        | _ -> tensor_attr ~shape:tensor.shape output_type
      end
    | List attr_list ->
      let list =
        match attr_list with
        | Float f -> create_attr_value_list_value ~f ()
        | String s -> create_attr_value_list_value ~s ()
        | Int i -> create_attr_value_list_value ~i:(List.map Int64.of_int i) ()
        | Bool b -> create_attr_value_list_value ~b ()
        | Type type_ -> create_attr_value_list_value ~type_:(List.map Type.to_dt_type type_) ()
        | Shape shape -> create_attr_value_list_value ~shape:(List.map shape_attr shape) ()
      in
      create_attr_value ~list ()
  in
  { Node_def_attr_entry.key = Some name
  ; value = Some value
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
  |> Protobuf.of_string

let of_node t = of_nodes [ P t ]
