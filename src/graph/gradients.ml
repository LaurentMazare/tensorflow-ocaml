open Core.Std

exception No_derivative_for_op of string

let registered_gradients = String.Table.create ()

type t =
  { f : 'a .
          (  self:([< `float | `double] as 'a) Node.t
          -> gradient:'a Node.t
          -> Node.p option list)
  }

let register_gradient op t =
  let f ~self:(Node.P self) ~gradient:(Node.P gradient) =
    match self.output_type, gradient.output_type with
    | Node.Type.Double, Node.Type.Double -> t.f ~self ~gradient
    | Node.Type.Float, Node.Type.Float -> t.f ~self ~gradient
    | _, _ ->
      failwithf "Inconsistent types %s" op ()
  in
  Hashtbl.set registered_gradients ~key:op ~data:f

(* Return a table mapping 'useful node' names to the number of times they
   appear as input of other useful nodes.
*)
let uses_per_node node with_respect_to =
  let uses_per_node = Node.Name.Table.create () in
  let rec is_useful node =
    let node_name = Node.packed_name node in
    let current_uses =
      Hashtbl.find uses_per_node node_name
    in
    let is_useful =
      Node.packed_is_real node
      &&
        (  Option.is_some current_uses
        || Set.mem with_respect_to node_name
        || List.exists (Node.packed_inputs node) ~f:is_useful)
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
    let output_type = input.output_type in
    let attributes =
      [ "N", Node.Int (List.length inputs)
      ; "T", Type (P output_type) ]
    in
    Node.P
      { name = Node.Name.make_fresh ~name:"gradient/addN"
      ; op_name = "addN"
      ; output_type
      ; inputs
      ; attributes
      ; output_name = None
      }

(* Compute the gradients of [node] with respect to [arg] using backpropagation. *)
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
      Hashtbl.add_multi contributions ~key:node_name ~data:gradient;
      let uses = uses - 1 in
      Hashtbl.set uses_per_node ~key:node_name ~data:uses;
      if uses = 0
      then begin
        let gradient =
          Hashtbl.find_exn contributions node_name |> aggregate_contributions
        in
        if Set.mem with_respect_to node_name
        then Hashtbl.add_exn output_gradients ~key:node_name ~data:gradient
        else
          let op_name = Node.packed_op_name node in
          match Hashtbl.find registered_gradients op_name with
          | None -> raise (No_derivative_for_op op_name)
          | Some fn ->
            List.iter2_exn
              (fn ~self:node ~gradient)
              (Node.packed_inputs node)
              ~f:(fun gradient input ->
                Option.iter gradient ~f:(fun gradient ->
                  add_contribution input ~gradient))
      end
  in 
  let scalar_one =
    Ops_m.const_float
      ~type_:node.output_type
      ~shape:[ 1 ]
      [ 1. ]
  in
  add_contribution (P node) ~gradient:(Node.P scalar_one);
  output_gradients
