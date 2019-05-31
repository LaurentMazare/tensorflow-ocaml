open Base

exception No_derivative_for_op of Node.Op_name.t

(* Return a table mapping 'useful node' names to the number of times they
   appear as input of other useful nodes.
   Nodes are useful in they are on a path between [node] and [with_respect_to]
   that contains only float/double nodes.
*)
let uses_per_node node with_respect_to =
  let uses_per_node = Hashtbl.create (module Node.Id) in
  let rec is_useful node =
    let node_id = Node.packed_id node in
    let current_uses = Hashtbl.find uses_per_node node_id in
    (* The [is_useful] function should be applied recursively to children only once.
       It should also apply to all children, hence the List.map ... |> List.exists below.
    *)
    let is_useful =
      Node.packed_is_real node
      && (Option.is_some current_uses
         || Set.mem with_respect_to node_id
         || List.map (Node.packed_flat_inputs node) ~f:is_useful |> List.exists ~f:Fn.id
         )
    in
    if is_useful
    then
      Hashtbl.set
        uses_per_node
        ~key:node_id
        ~data:(1 + Option.value ~default:0 current_uses);
    is_useful
  in
  ignore (is_useful node : bool);
  uses_per_node

let aggregate_contributions = function
  | [] -> assert false
  | [ input ] -> input
  | Node.P input :: _ as inputs ->
    let output_type = Node.output_type input in
    let inputs =
      List.map inputs ~f:(fun input -> Option.value_exn (Node.extract input output_type))
    in
    (match output_type with
    | Node.Type.Double -> Node.P (Ops.addN inputs)
    | Node.Type.Float -> Node.P (Ops.addN inputs)
    | _ -> failwith "Improper type.")

let aggregate_contributions_multi gradients =
  List.map gradients ~f:(fun (output_idx, gradient) ->
      Option.value_exn output_idx, gradient)
  |> Map.of_alist_multi (module Int)
  |> Map.map ~f:aggregate_contributions

(* Compute the gradients of [node] with respect to [arg] using backpropagation.
   This only works when [node] is a scalar. *)
let gradient node ~with_respect_to =
  let with_respect_to =
    List.map with_respect_to ~f:Node.packed_id |> Set.of_list (module Node.Id)
  in
  let uses_per_node = uses_per_node (P node) with_respect_to in
  let contributions = Hashtbl.create (module Node.Id) in
  let output_gradients = Hashtbl.create (module Node.Id) in
  let rec add_contribution node ~gradient =
    let node_id = Node.packed_id node in
    match Hashtbl.find uses_per_node node_id with
    | None -> ()
    | Some uses ->
      assert (uses > 0);
      Option.iter gradient ~f:(fun gradient ->
          Hashtbl.add_multi
            contributions
            ~key:node_id
            ~data:(Node.packed_output_idx node, gradient));
      let uses = uses - 1 in
      Hashtbl.set uses_per_node ~key:node_id ~data:uses;
      if uses = 0
      then (
        let gradients = Hashtbl.find contributions node_id in
        if Set.mem with_respect_to node_id
        then (
          let gradient =
            Option.map gradients ~f:(fun gradients ->
                List.map gradients ~f:snd |> aggregate_contributions)
          in
          Hashtbl.add_exn output_gradients ~key:node_id ~data:gradient)
        else (
          let op_name = Node.packed_op_name node in
          match gradients with
          | None ->
            List.iter (Node.packed_flat_inputs node) ~f:(add_contribution ~gradient:None)
          | Some gradients ->
            (match Registered_gradients.find op_name with
            | None ->
              (match Registered_gradients.find_multi op_name with
              | None -> raise (No_derivative_for_op op_name)
              | Some fn ->
                let gradients = aggregate_contributions_multi gradients in
                (try
                   List.iter2_exn
                     (fn ~self:node ~gradients)
                     (Node.packed_flat_inputs node)
                     ~f:(fun gradient input -> add_contribution input ~gradient)
                 with
                | exn -> Exn.reraise exn (Node.Op_name.to_string op_name)))
            | Some fn ->
              (try
                 let gradients = List.map gradients ~f:snd in
                 List.iter2_exn
                   (fn ~self:node ~gradient:(aggregate_contributions gradients))
                   (Node.packed_flat_inputs node)
                   ~f:(fun gradient input -> add_contribution input ~gradient)
               with
              | exn -> Exn.reraise exn (Node.Op_name.to_string op_name)))))
  in
  let one =
    Ops.const_float ~shape:[] ~type_:(Node.output_type node) [ 1. ]
    |> Ops.fill (Ops.shape node ~type_:Int32)
  in
  add_contribution (P node) ~gradient:(Some (Node.P one));
  output_gradients

let gradient_caml node ~with_respect_to_float ~with_respect_to_double =
  let pack = List.map ~f:(fun node -> Node.P node) in
  let table =
    gradient
      node
      ~with_respect_to:(pack with_respect_to_float @ pack with_respect_to_double)
  in
  let cast : type a. Node.p -> type_:a Node.Type.t -> a Node.t =
   fun node ~type_ -> Option.value_exn (Node.extract node type_)
  in
  let lookup ~type_ =
    List.map ~f:(fun node ->
        match Hashtbl.find table (Node.id node) with
        | Some (Some gradient) -> cast gradient ~type_
        | Some None | None ->
          (* The node hasn't been reached from the root. *)
          Ops.zerosLike node)
  in
  ( lookup with_respect_to_float ~type_:Node.Type.Float
  , lookup with_respect_to_double ~type_:Node.Type.Double )

let () = Ops_gradients.register_all ()

let gradient_tf node ~with_respect_to_float ~with_respect_to_double =
  let open Tensorflow_core in
  let graph = Node.operation node |> Wrapper.Graph.graph in
  let add_gradient xs ~output_type =
    Wrapper.Graph.add_gradients
      graph
      [ Node.output node ]
      ~xs:(List.map xs ~f:Node.output)
    |> Wrapper.Status.ok_exn
    |> List.map ~f:(Node.create_gradient ~output_type)
  in
  ( add_gradient with_respect_to_float ~output_type:Node.Type.Float
  , add_gradient with_respect_to_double ~output_type:Node.Type.Double )
