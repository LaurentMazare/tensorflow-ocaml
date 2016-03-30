open Core_kernel.Std
(* An higher level view of a session *)

(* CR-soon noury: the whole renaming of fresh variables and export can be done in one
   pass but we need to think more. *)

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

let create () =
  match Wrapper.Session.create () with
  | Error status ->
    failwithf "Unable to generate session: %s" (Wrapper.Status.message status) ()
  | Ok session ->
    { session;
      exported_nodes = Node.Id.Table.create ();
      names = String.Table.create ();
      uninitialised_variables = Int.Table.create ();
      current_table = Node.Id.Table.create()
    }

let default_session = lazy (create ())

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
  let choose_correct_output (Node.P n) =
    Node.P { n with output_idx = Node.packed_output_idx node }
  in
  let id = Node.packed_id node in
  match Hashtbl.find t.exported_nodes id with
  | Some h -> (choose_correct_output h, 0) (* already exported before starting this run *)
  | None ->
    match Hashtbl.find t.current_table id with
    | Some (node, h) -> choose_correct_output node, h (* already exported this round *)
    | None ->
      let Node.P u_node = node in
      let rev_inputs, height =
        List.fold u_node.inputs ~init:([], 0)
          ~f:(fun (rev_inputs, height) input ->
              let input, h = prepare_node t input in
              input::rev_inputs, max h height)
      in
      let res =
        Node.P
        { u_node with
          name = choose_name t node
        ; inputs = List.rev rev_inputs
        }
      in
      let h =
       match Var.get_init node with
       | None -> height
       | Some assign ->
         Hashtbl.set t.current_table ~key:id ~data:(res, 0);
         let assign, h = prepare_node t assign in
         Hashtbl.add_multi t.uninitialised_variables ~key:h ~data:assign;
         h + 1
      in
      Hashtbl.set t.current_table ~key:id ~data:(res, h);
      (res, h)
;;

type node_names =
  { inputs : string list
  ; targets : string list
  ; outputs : string list
  ; variables_to_initialize : string list list
  }

let prepare_graph t ~inputs ~targets ~outputs =
  let prep l =
    List.map l ~f:(fun node -> fst (prepare_node t node))
  in
  let inputs  = prep inputs in
  let targets = prep targets in
  let outputs = prep outputs in
  Hashtbl.clear t.current_table;
  let rec build_variables i =
    match Hashtbl.find t.uninitialised_variables i with
    | None -> []
    | Some l -> l::build_variables (i + 1)
  in
  let uninitialised_variables = build_variables 0 in
  Hashtbl.clear t.uninitialised_variables;
  let all_nodes_to_export =
    List.concat uninitialised_variables @ List.concat [ inputs; outputs; targets ]
  in
  let protobuf =
    Node_protobuf.of_nodes' ~already_exported_nodes:t.exported_nodes all_nodes_to_export
  in
  Wrapper.Session.(extend_graph t.session protobuf |> ok_exn);
  let node_names = List.map ~f:(fun x -> Node.packed_name x |> Node.Name.to_string) in
  { inputs = node_names inputs
  ; targets = node_names targets
  ; outputs = node_names outputs
  ; variables_to_initialize = List.map ~f:node_names uninitialised_variables
  }

let run ?(inputs=[]) ?(outputs=[]) ?(targets=[]) t =
  let inputs, input_tensors = List.unzip inputs in
  let { inputs; targets; outputs; variables_to_initialize } =
    prepare_graph t ~inputs ~targets ~outputs
  in
  (* add outputs to targets *)
  let targets =
    List.fold outputs ~init:(String.Set.of_list targets) ~f:Set.add
    |> Set.to_list
  in
  let inputs = List.zip_exn inputs input_tensors in
  (* Run variable init *)
  List.iter variables_to_initialize ~f:(fun targets ->
    Wrapper.Session.run t.session ~inputs:[] ~outputs:[] ~targets
    |> Wrapper.Session.ok_exn
    |> fun tensor_list -> assert (List.is_empty tensor_list));
  Wrapper.Session.(run t.session ~inputs ~outputs ~targets |> ok_exn)

module Input =
 struct
   type t =
   | I : _ Node.t * (_,_) Tensor.t -> t

  let float (node : [`float ] Node.t)  (tensor : (float, Bigarray.float32_elt) Tensor.t)  =
    I (node, tensor)

  let double (node : [`double ] Node.t)  (tensor : (float, Bigarray.float64_elt) Tensor.t)  =
    I (node, tensor)
 end

module Target =
struct
  type t = Node.p
end

let target node = Node.P node

module Output =
struct
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

  (* CR-someday noury: this could be just one function with modular implicits *)
  let float (node : [`float] Node.t) : (float, Bigarray.float32_elt) Tensor.t t =
    Compute node
    |> map
    ~f:(fun (Tensor.P tensor) ->
      match Bigarray.Genarray.kind tensor with
      | Bigarray.Float32 -> (tensor : (float, Bigarray.float32_elt) Tensor.t)
      | _ -> failwith "PANIC: wrong kind in float")

  let double (node : [`double] Node.t) : (float, Bigarray.float64_elt) Tensor.t t =
    Compute node
    |> map
    ~f:(fun (Tensor.P tensor) ->
      match Bigarray.Genarray.kind tensor with
      | Bigarray.Float64 -> (tensor : (float, Bigarray.float64_elt) Tensor.t)
      | _ -> failwith "PANIC: wrong kind in double")

  (* CR noury: add more output types *)

  let scalar_float node =
    float node |> map ~f:(fun t ->
      Array.create 0 ~len:(Bigarray.Genarray.num_dims t)
      |> Bigarray.Genarray.get t)

  let scalar_double node =
    double node |> map ~f:(fun t ->
      Array.create 0 ~len:(Bigarray.Genarray.num_dims t)
      |> Bigarray.Genarray.get t)

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
      l, (fun l -> let a, l = (k l) in f a, l)
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
    Option.map inputs
     ~f:(List.map ~f:(fun (Input.I (n, t)) -> Node.P n, Tensor.P t))
  in
  let outputs, k = Output.build_output output in
  k (run ?inputs ?targets ~outputs t)
