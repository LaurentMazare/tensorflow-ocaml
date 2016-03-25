open Core_kernel.Std
(* An higher level view of a session *)

(* CR-soon noury: the whole renaming of fresh variables and export can be done in one
   pass but we need to think more. *)

(* the structure of variable to initialize might be slightly tricky:
   some variables migh depend on other to be already initialised in order to be
   initialised.
   I think it can even be a nice way to make sure the size of multiple layers matches.
   (use the parameter of one layer to initialise the parameter of the next) *)
module Variable_initialisation =
struct
  (* Initialisation will run each list in order *)
  type t = Node.p list list
end
(* There is no uninitialised variables because I think we can initialize
them last minute when someone call run, as there is no extend graph *)
type t =
{ session : Wrapper.Session.t;
  (* The nodes already in the graph of the session,
     with their name there *)
  exported_nodes : Node.p Node.Id.Table.t;
  (* The names already present on the server, with the number of times
     it has been used *)
  names : int String.Table.t;
  (* To manage variable initialisation, each unitialised variable has a height.
     If a variable init depends of no initialised variable,
     its height is 0.
     If it depends of unitialised variable v1 ... vn, its height is max(height(vi)) + 1
     Initialisation can be done one level after one level *)
  uninitialised_variables : Node.p list Int.Table.t;
  current_table : (Node.p * int) Node.Id.Table.t
}

let rec choose_name t node =
  let base_name = Node.Name.to_string (Node.packed_name node) in
  match Hashtbl.find t.names  base_name with
  | None ->
    Hashtbl.set t.names ~key:base_name ~data:1;
    Node.Name.of_string base_name
  | Some i ->
    Hashtbl.set t.names ~key:base_name ~data:(i + 1);
    let name = sprintf "%s-%i" base_name i in
    if Hashtbl.mem t.names name
    (* Our new name conflict with a base name, so we try again with another number *)
    then choose_name t node
    else Node.Name.of_string name

(* returns a graph with nodes freshly renamed.
   computes the unitialised variable.
   returns what would be the height of a variable just above.
*)
let rec prepare_node t node =
  let id = Node.packed_id node in
  match
   Hashtbl.find t.exported_nodes id
  with
  | Some h -> (h, 0) (* already exported before starting this run *)
  | None ->
  match Hashtbl.find t.current_table id with
  | Some res -> res (* already exported this round *)
  | None ->
    let Node.P u_node = node in
    let rev_inputs, height =
    List.fold u_node.inputs ~init:([], 0)
      ~f:(fun (rev_inputs, height) input ->
          let input, h = prepare_node t input in
          input::rev_inputs, max h height)
    in
    let node =
      Node.P
      { u_node with
        name = choose_name t node
      ; inputs = List.rev rev_inputs
      }
   in
   let h =
    if Node.Op_name.to_string (Node.packed_op_name node) = "Var"
    then
      begin
       (*CR noury: should generate the code for variable initialisation *)
        Hashtbl.add_multi t.uninitialised_variables ~key:height ~data:node ;
        height + 1
      end
    else height
   in
   Hashtbl.set t.current_table ~key:id ~data:(node,h);
   (node, h)
;;

let prepare_graph t l =
  let l =
    List.fold l ~init:[]
      ~f:(fun acc node ->
        fst (prepare_node t node)::acc)
  in
  Hashtbl.clear t.current_table;
  let rec build_variables i =
    match Hashtbl.find t.uninitialised_variables i with
    | None -> []
    | l ->
     l::build_variables (i + 1)
  in
  let _uninitialised_variables = build_variables 0 in
  Hashtbl.clear t.uninitialised_variables;
  let protobuf = Node_protobuf.of_nodes' t.exported_nodes l in
  Wrapper.Session.extend_graph t.session protobuf





