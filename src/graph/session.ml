open Core_kernel.Std
(* An higher level view of a session *)

(* CR-soon noury: the whole renaming of fresh variables and export can be done in one
   pass but we need to think more. *)

(* There is no uninitialised variables because I think we can initialize
them last minute when someone call run, as there is no extend graph *)
type t =
  { session : Wrapper.Session.t
  (* The nodes already in the graph of the session, with their name there *)
  ; exported_nodes : Node.Id.Hash_set.t
  (* To manage variable initialisation, each unitialised variable has a height.
     If a variable init depends of no initialised variable, its height is 0.
     If it depends of unitialised variable v1 ... vn, its height is max(height(vi)) + 1
     Initialisation can be done one level after one level *)
  ; uninitialised_variables : Node.p list Int.Table.t
}

let create () =
  match Wrapper.Session.create () with
  | Error status ->
    failwithf "Unable to generate session: %s" (Wrapper.Status.message status) ()
  | Ok session ->
    { session
    ; exported_nodes = Node.Id.Hash_set.create ()
    ; uninitialised_variables = Int.Table.create ()
    }

let default_session = lazy (create ())

(* [walk t node] walks through the graph and store the unitialized variables. *)
let rec walk t node ~current_table =
  let id = Node.packed_id node in
  if Hash_set.mem t.exported_nodes id
  then 0 (* already exported before starting this run *)
  else
    Hashtbl.find_or_add current_table id ~default:(fun () ->
      let Node.P u_node = node in
      match Var.get_init u_node with
      | None ->
        List.fold (Node.inputs u_node) ~init:0 ~f:(fun acc_height input ->
          max acc_height (walk t input ~current_table))
      | Some init ->
        let h = walk t (Node.P init) ~current_table in
        let assign = Node.P (Ops.assign u_node init) in
        Hashtbl.add_multi t.uninitialised_variables ~key:h ~data:assign;
        h + 1)

type node_names =
  { inputs : string list
  ; targets : string list
  ; outputs : string list
  ; variables_to_initialize : string list list
  }

let prepare_graph t ~inputs ~targets ~outputs =
  let current_table = Node.Id.Table.create () in
  let prep =
    List.iter ~f:(fun node -> ignore (walk t node ~current_table : int))
  in
  prep inputs;
  prep targets;
  prep outputs;
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
  Option.iter protobuf ~f:(fun protobuf ->
    Wrapper.Session.(extend_graph t.session protobuf |> ok_exn));
  let node_names = List.map ~f:(fun (Node.P x) -> Node.unique_name x) in
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
    Option.map inputs
     ~f:(List.map ~f:(fun (Input.I (n, t)) -> Node.P n, Tensor.P t))
  in
  let outputs, k = Output.build_output output in
  k (run ?inputs ?targets ~outputs t)
