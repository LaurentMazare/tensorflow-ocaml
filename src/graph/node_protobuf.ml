open Core_kernel.Std
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

let tensor_attr
      ?(float_val = [])
      ?(double_val = [])
      ?(int_val = [])
      ?(int64_val = [])
      ?(string_val = [])
      ~shape
      output_type
  =
  let dim =
    List.map shape ~f:(fun d ->
      { Tensor_shape_proto_dim.size = Some (Int64.of_int d); name = None })
  in
  let tensor =
    { default_tensor_proto with
      dtype = Some (Node.Type.to_dt_type (P output_type))
    ; float_val
    ; double_val
    ; int_val
    ; int64_val
    ; string_val
    ; tensor_shape = Some { dim; unknown_rank = None }
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
    List.map shape ~f:(fun { Node.Dim.size; name } ->
      { Tensor_shape_proto_dim.size = Some (Int64.of_int size)
      ; name
      })
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
          let int_val = List.map tensor.values ~f:Int32.of_int_exn in
          tensor_attr ~int_val ~shape:tensor.shape output_type
        | Node.Type.Int64 ->
          let int64_val = List.map tensor.values ~f:Int64.of_int in
          tensor_attr ~int64_val ~shape:tensor.shape output_type
        | _ -> tensor_attr ~shape:tensor.shape output_type
      end
    | Tensor_string tensor ->
      begin
        match output_type with
        | Node.Type.String ->
          tensor_attr ~string_val:tensor.values ~shape:tensor.shape output_type
        | _ -> tensor_attr ~shape:tensor.shape output_type
      end
    | List attr_list ->
      let list =
        match attr_list with
        | Float f -> create_attr_value_list_value ~f ()
        | String s -> create_attr_value_list_value ~s ()
        | Int i -> create_attr_value_list_value ~i:(List.map i ~f:Int64.of_int) ()
        | Bool b -> create_attr_value_list_value ~b ()
        | Type type_ -> create_attr_value_list_value ~type_:(List.map type_ ~f:Type.to_dt_type) ()
        | Shape shape -> create_attr_value_list_value ~shape:(List.map shape ~f:shape_attr) ()
      in
      create_attr_value ~list ()
  in
  { Node_def_attr_entry.key = Some name
  ; value = Some value
  }

let of_nodes' ?verbose ~already_exported_nodes ts =
  let verbose = Option.is_some verbose in
  let output = ref [] in
  let rec walk p =
    let P t = p in
    if Hash_set.mem already_exported_nodes (Node.id t)
    then ()
    else begin
      if verbose
      then begin
        let inputs =
          if List.is_empty (Node.inputs t)
          then "No inputs"
          else
            List.map (Node.inputs t) ~f:(fun input ->
              Node.packed_name input |> Node.Name.to_string)
            |> String.concat ~sep:", "
            |> sprintf "Inputs: %s"
        in
        printf "Node: %s %s Op: %s %s\n%!"
          (Node.Id.to_string (Node.id t))
          (Node.Name.to_string (Node.name t))
          (Node.Op_name.to_string (Node.op_name t))
          inputs
      end;
      let attr =
        List.map (Node.attributes t) ~f:(fun (name, value) ->
          of_attribute name value (Node.output_type t))
      in
      let input =
        List.map (Node.inputs t) ~f:(fun (P input) ->
          let idx = Option.value_map (Node.output_idx input) ~default:"" ~f:(sprintf ":%d") in
          Node.unique_name input ^ idx)
      in
      let node =
        { Node_def.name = Some (Node.unique_name t)
        ; op = Some (Node.Op_name.to_string (Node.op_name t))
        ; input
        ; device = None
        ; attr
        }
      in
      Hash_set.add already_exported_nodes (Node.id t);
      output := node :: !output;
      List.iter (Node.inputs t) ~f:walk
    end
  in
  List.iter ts ~f:walk;
  match !output with
  | [] -> None
  | node ->
    { Graph_def.node
    ; versions =
      Some
        { Version_def.producer = Some (Int32.of_int_exn 8)
        ; min_consumer = None
        ; bad_consumers = []
        }
    ; version = None
    ; library = None
    }
    |> gen_graph_def
    |> Piqirun.to_string
    |> Protobuf.of_string
    |> Option.some

let of_nodes ?verbose ts =
  of_nodes' ?verbose ~already_exported_nodes:(Node.Id.Hash_set.create ()) ts

let of_node ?verbose t = of_nodes ?verbose [ P t ]
