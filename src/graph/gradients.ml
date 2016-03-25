open Core_kernel.Std

exception No_derivative_for_op of Node.Op_name.t

(* Return a table mapping 'useful node' names to the number of times they
   appear as input of other useful nodes.
   Nodes are useful in they are on a path between [node] and [with_respect_to]
   that contains only float/double nodes.
*)
let uses_per_node node with_respect_to =
  let uses_per_node = Node.Name.Table.create () in
  let rec is_useful node =
    let node_name = Node.packed_name node in
    let current_uses =
      Hashtbl.find uses_per_node node_name
    in
    (* The [is_useful] function should be applied recursively to children only once.
       It should also apply to all children, hence the List.map ... |> List.exists below.
    *)
    let is_useful =
      Node.packed_is_real node
      &&
        (  Option.is_some current_uses
        || Set.mem with_respect_to node_name
        || List.map (Node.packed_inputs node) ~f:is_useful |> List.exists ~f:Fn.id)
    in
    if is_useful
    then
      Hashtbl.set uses_per_node
        ~key:node_name
        ~data:(1 + Option.value ~default:0 current_uses);
    is_useful
  in
  ignore (is_useful node : bool);
  uses_per_node

let aggregate_contributions = function
  | [] -> assert false
  | [ input ] -> input
  | (Node.P input :: _) as inputs ->
    (* Hack: all the nodes in [inputs] should have the same type however they are packed
       so we cannot use [Ops.addN] directly and build the node manually instead. *)
    let output_type = input.output_type in
    let attributes =
      [ "N", Node.Int (List.length inputs)
      ; "T", Type (P output_type) ]
    in
    Node.P
      { name = Node.Name.make_fresh ~name:"gradient/addN"
      ; op_name = Node.Op_name.of_string "AddN"
      ; output_type
      ; inputs
      ; attributes
      ; output_name = None
      }

(* Compute the gradients of [node] with respect to [arg] using backpropagation.
   This only works when [node] is a scalar. *)
let gradient node ~with_respect_to =
  let with_respect_to =
    List.map with_respect_to ~f:Node.packed_name |> Node.Name.Set.of_list
  in
  let uses_per_node = uses_per_node (P node) with_respect_to in
  let contributions = Node.Name.Table.create () in
  let output_gradients = Node.Name.Table.create () in
  let rec add_contribution node ~gradient =
    let node_name = Node.packed_name node in
    match Hashtbl.find uses_per_node node_name with
    | None -> ()
    | Some uses ->
      assert (uses > 0);
      Option.iter gradient ~f:(fun gradient ->
        Hashtbl.add_multi contributions ~key:node_name ~data:gradient);
      let uses = uses - 1 in
      Hashtbl.set uses_per_node ~key:node_name ~data:uses;
      if uses = 0
      then
        let gradient =
          Option.map (Hashtbl.find contributions node_name) ~f:aggregate_contributions
        in
        if Set.mem with_respect_to node_name
        then Hashtbl.add_exn output_gradients ~key:node_name ~data:gradient
        else
          let op_name = Node.packed_op_name node in
          match Registered_gradients.find op_name with
          | None -> raise (No_derivative_for_op op_name)
          | Some fn ->
            match gradient with
            | None ->
              List.iter (Node.packed_inputs node) ~f:(add_contribution ~gradient:None)
            | Some gradient ->
              try
                List.iter2_exn
                  (fn ~self:node ~gradient)
                  (Node.packed_inputs node)
                  ~f:(fun gradient input -> add_contribution input ~gradient)
              with
              | exn -> Exn.reraise exn (Node.Op_name.to_string op_name)
  in 
  let one =
    Ops_m.const_float
      ~shape:[]
      ~type_:node.output_type
      [ 1. ]
    |> Ops.fill (Ops.shape node)
  in
  add_contribution (P node) ~gradient:(Some (Node.P one));
  output_gradients

let gradient node ~with_respect_to_float ~with_respect_to_double =
  let pack = List.map ~f:(fun node -> Node.P node) in
  let table =
    gradient node
      ~with_respect_to:(pack with_respect_to_float @ pack with_respect_to_double)
  in
  let cast : type a . Node.p -> type_: a Node.Type.t -> a Node.t =
    fun node ~type_ -> Option.value_exn (Node.extract node type_)
  in
  let lookup ~type_ =
    List.map ~f:(fun node ->
      match Hashtbl.find table node.Node.name with
      | Some (Some gradient) -> cast gradient ~type_
      | Some None | None -> (* The node hasn't been reached from the root. *)
        Ops.zerosLike node
    )
  in
  lookup with_respect_to_float ~type_:Node.Type.Float,
  lookup with_respect_to_double ~type_:Node.Type.Double

let () = Ops_gradients.register_all ()
