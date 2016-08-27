open Core_kernel.Std

type t =
  { session : Wrapper.Session_with_graph.t
  ; graph : Wrapper.Graph.t
  ; nodes : Wrapper.Graph.operation Node.Id.Table.t
  }

let create () =
  let graph = Wrapper.Graph.create () in
  match Wrapper.Session_with_graph.create graph with
  | Error status ->
    failwithf "Unable to generate session: %s" (Wrapper.Status.message status) ()
  | Ok session ->
    { session
    ; graph
    ; nodes = Node.Id.Table.create ()
    }

let default_session = lazy (create ())

let add_attribute operation_description ~attr_name attr =
  match (attr : Node.attr) with
  | String str ->
    Wrapper.Graph.set_attr_string operation_description ~attr_name str
  | Type dtype ->
    let dtype = Node.Type.to_data_type dtype in
    Wrapper.Graph.set_attr_type operation_description ~attr_name dtype
  | Tensor_float tensor ->
    let shape =
      match tensor.shape with
      | [] -> [| List.length tensor.values |]
      | shape -> Array.of_list shape
    in
    let tensor = Tensor.create Float32 shape in
    (* TODO: set values... *)
    Wrapper.Graph.set_attr_tensor operation_description ~attr_name (Tensor.P tensor)
    |> Wrapper.Status.ok_exn
  | _ -> ()

let rec build t node =
  let id = Node.packed_id node in
  match Hashtbl.find t.nodes id with
  | Some op -> op
  | None ->
    let Node.P u_node = node in
    (* TODO: [Var.get_init]. *)
    let operation_description =
      Wrapper.Graph.new_operation t.graph
        ~op_name:(Node.op_name u_node |> Node.Op_name.to_string)
        ~name:(Node.unique_name u_node)
    in
    List.iter (Node.inputs u_node) ~f:(function
      | `single input ->
        Wrapper.Graph.add_input
          operation_description
          (build t input)
          ~index:(Node.packed_output_idx input |> Option.value ~default:0)
      | `multi inputs ->
        let inputs =
          List.map inputs ~f:(fun input ->
            let index = Node.packed_output_idx input |> Option.value ~default:0 in
            build t input, index)
        in
        Wrapper.Graph.add_inputs operation_description inputs);
    List.iter (Node.attributes u_node) ~f:(fun (attr_name, attr) ->
      add_attribute operation_description ~attr_name attr);
    let operation =
      Wrapper.Graph.finish_operation operation_description
      |> Wrapper.Status.ok_exn
    in
    Hashtbl.set t.nodes ~key:id ~data:operation;
    operation

let run ?(inputs=[]) ?(outputs=[]) ?(targets=[]) t =
  let inputs, input_tensors = List.unzip inputs in
  let inputs =
    List.map inputs ~f:(fun input ->
      build t input
      |> Wrapper.Graph.create_port ~index:0)
  in
  let outputs = List.map outputs ~f:(build t) in
  let targets =
    List.map targets ~f:(build t) @ outputs
  in
  let outputs = List.map outputs ~f:(fun op -> Wrapper.Graph.create_port op ~index:0) in
  let inputs = List.zip_exn inputs input_tensors in
  (* TODO: Run variable init *)
  Wrapper.Session_with_graph.run t.session ~inputs ~outputs ~targets
  |> Wrapper.Status.ok_exn

module Input = struct
   type t =
   | I : _ Ops.Placeholder.t * (_,_) Tensor.t -> t

  let float
        (node : [ `float ] Ops.Placeholder.t)
        (tensor : (float, Bigarray.float32_elt) Tensor.t)
    =
    I (node, tensor)

  let double
        (node : [ `double ] Ops.Placeholder.t)
        (tensor : (float, Bigarray.float64_elt) Tensor.t)
    =
    I (node, tensor)
 end

module Output = struct
  type _ t =
    | Return : 'a -> 'a t
    | Compute : _ Node.t -> Tensor.p t
    | Both : 'a t * 'b t ->  ('a * 'b) t
    | Map : 'a t * ('a -> 'b) -> 'b t
    | Empty : unit t

  let map t ~f = Map (t, f)
  let return node = Return node
  let both t1 t2 = Both (t1, t2)
  let empty = Empty

  let three t1 t2 t3 =
    both t1 (both t2 t3) |> map ~f:(fun (t1, (t2, t3)) -> t1, t2, t3)

  let four t1 t2 t3 t4 =
    both (both t1 t2) (both t3 t4)
    |> map ~f:(fun ((t1, t2), (t3, t4)) -> t1, t2, t3, t4)

  let five t1 t2 t3 t4 t5 =
    both (both (both t1 t2) (both t3 t4)) t5
    |> map ~f:(fun (((t1, t2), (t3, t4)), t5) -> t1, t2, t3, t4, t5)

  let six t1 t2 t3 t4 t5 t6 =
    both (both (both t1 t2) (both t3 t4)) (both t5 t6)
    |> map ~f:(fun (((t1, t2), (t3, t4)), (t5, t6)) -> t1, t2, t3, t4, t5, t6)

  (* CR-someday noury: this could be just one function with modular implicits *)
  let float (node : [`float] Node.t) : (float, Bigarray.float32_elt) Tensor.t t =
    Compute node
    |> map ~f:(fun (Tensor.P tensor) ->
      match Tensor.kind tensor with
      | Bigarray.Float32 -> (tensor : (float, Bigarray.float32_elt) Tensor.t)
      | _ -> failwith "PANIC: wrong kind in float")

  let double (node : [`double] Node.t) : (float, Bigarray.float64_elt) Tensor.t t =
    Compute node
    |> map ~f:(fun (Tensor.P tensor) ->
      match Tensor.kind tensor with
      | Bigarray.Float64 -> (tensor : (float, Bigarray.float64_elt) Tensor.t)
      | _ -> failwith "PANIC: wrong kind in double")

  let int32 (node : [`int32] Node.t) : (int32, Bigarray.int32_elt) Tensor.t t =
    Compute node
    |> map ~f:(fun (Tensor.P tensor) ->
      match Tensor.kind tensor with
      | Bigarray.Int32 -> (tensor : (int32, Bigarray.int32_elt) Tensor.t)
      | _ -> failwith "PANIC: wrong kind in double")

  let int64 (node : [`int64] Node.t) : (Int64.t, Bigarray.int64_elt) Tensor.t t =
    Compute node
    |> map ~f:(fun (Tensor.P tensor) ->
      match Tensor.kind tensor with
      | Bigarray.Int64 -> (tensor : (Int64.t, Bigarray.int64_elt) Tensor.t)
      | _ -> failwith "PANIC: wrong kind in double")

  (* CR noury: add more output types *)

  let scalar_gen extract node =
    extract node |> map ~f:(fun t ->
      Array.create 0 ~len:(Tensor.num_dims t)
      |> Tensor.get t)

  let scalar_float n = scalar_gen float n
  let scalar_double n = scalar_gen double n
  let scalar_int32 n = scalar_gen int32 n |> map ~f:Int32.to_int_exn
  let scalar_int64 n = scalar_gen int64 n

  let rec build_output
    : type a. a t ->  (Node.p list -> Node.p list) * (Tensor.p list -> a * Tensor.p list) =
    function
    | Return a -> (fun l -> l), (fun l -> a, l)
    | Both (o1, o2) ->
      let l1, k1 = build_output o1 in
      let l2, k2 = build_output o2 in
      (fun l -> l1 (l2 l)),
      (fun l ->
        let a, l = k1 l in
        let b, l = k2 l in
        (a, b), l)
    | Map (o, f) ->
      let l, k = build_output o in
      l, (fun l -> let a, l = k l in f a, l)
    | Empty -> Fn.id, fun l -> (), l
    | Compute node ->
     (fun l -> (P node) :: l),
     function
     | t::l -> t, l
     | [] -> failwith "wrong number of elts in output dispatch"

  let build_output o =
   let f, k = build_output o in
   f [], fun l -> fst (k l)
end

let run ?inputs ?targets ?session output =
  let t =
    match session with
    | None -> Lazy.force default_session
    | Some session -> session
  in
  let inputs =
    Option.map inputs ~f:(List.map ~f:(fun (Input.I (n, t)) ->
      Node.P (Ops.Placeholder.to_node n), Tensor.P t))
  in
  let outputs, k = Output.build_output output in
  k (run ?inputs ?targets ~outputs t)
