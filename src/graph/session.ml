open Base
open Tensorflow_core

let run ?(inputs = []) ?(outputs = []) ?(targets = []) () =
  let cmp (Node.P n1, _) (Node.P n2, _) = Node.Id.compare (Node.id n1) (Node.id n2) in
  if List.contains_dup ~compare:cmp inputs
  then failwith "Session.run: duplicate entry in [inputs].";
  let inputs =
    List.map inputs ~f:(fun (input, input_tensor) ->
        Node.packed_output input, input_tensor)
  in
  let targets = List.map (targets @ outputs) ~f:Node.packed_operation in
  let outputs = List.map outputs ~f:Node.packed_output in
  let session = Node.Session.default_session () in
  (* [variable_initializations] is topologically sorted. *)
  List.iter (Node.Session.get_and_reset_variable_initializations ()) ~f:(fun init_op ->
      Wrapper.Session.run session ~inputs ~outputs:[] ~targets:[ init_op ]
      |> Wrapper.Status.ok_exn
      |> fun l -> assert (List.is_empty l));
  Wrapper.Session.run session ~inputs ~outputs ~targets |> Wrapper.Status.ok_exn

module Input = struct
  type t = I : _ Ops.Placeholder.t * (_, _) Tensor.t -> t

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

  let bool
      (node : [ `bool ] Ops.Placeholder.t)
      (tensor : (int, Bigarray.int8_unsigned_elt) Tensor.t)
    =
    I (node, tensor)
end

module Output = struct
  type _ t =
    | Return : 'a -> 'a t
    | Compute : _ Node.t -> Tensor.p t
    | Both : 'a t * 'b t -> ('a * 'b) t
    | Map : 'a t * ('a -> 'b) -> 'b t
    | Empty : unit t

  let map t ~f = Map (t, f)
  let return node = Return node
  let both t1 t2 = Both (t1, t2)
  let empty = Empty
  let three t1 t2 t3 = both t1 (both t2 t3) |> map ~f:(fun (t1, (t2, t3)) -> t1, t2, t3)

  let four t1 t2 t3 t4 =
    both (both t1 t2) (both t3 t4) |> map ~f:(fun ((t1, t2), (t3, t4)) -> t1, t2, t3, t4)

  let five t1 t2 t3 t4 t5 =
    both (both (both t1 t2) (both t3 t4)) t5
    |> map ~f:(fun (((t1, t2), (t3, t4)), t5) -> t1, t2, t3, t4, t5)

  let six t1 t2 t3 t4 t5 t6 =
    both (both (both t1 t2) (both t3 t4)) (both t5 t6)
    |> map ~f:(fun (((t1, t2), (t3, t4)), (t5, t6)) -> t1, t2, t3, t4, t5, t6)

  (* TODO-someday noury: this could be just one function with modular implicits *)
  let float (node : [ `float ] Node.t) : (float, Bigarray.float32_elt) Tensor.t t =
    Compute node
    |> map ~f:(fun (Tensor.P tensor) ->
           match Tensor.kind tensor with
           | Bigarray.Float32 -> (tensor : (float, Bigarray.float32_elt) Tensor.t)
           | _ -> failwith "PANIC: wrong kind in float")

  let double (node : [ `double ] Node.t) : (float, Bigarray.float64_elt) Tensor.t t =
    Compute node
    |> map ~f:(fun (Tensor.P tensor) ->
           match Tensor.kind tensor with
           | Bigarray.Float64 -> (tensor : (float, Bigarray.float64_elt) Tensor.t)
           | _ -> failwith "PANIC: wrong kind in double")

  let int32 (node : [ `int32 ] Node.t) : (int32, Bigarray.int32_elt) Tensor.t t =
    Compute node
    |> map ~f:(fun (Tensor.P tensor) ->
           match Tensor.kind tensor with
           | Bigarray.Int32 -> (tensor : (int32, Bigarray.int32_elt) Tensor.t)
           | _ -> failwith "PANIC: wrong kind in double")

  let int64 (node : [ `int64 ] Node.t) : (Int64.t, Bigarray.int64_elt) Tensor.t t =
    Compute node
    |> map ~f:(fun (Tensor.P tensor) ->
           match Tensor.kind tensor with
           | Bigarray.Int64 -> (tensor : (Int64.t, Bigarray.int64_elt) Tensor.t)
           | _ -> failwith "PANIC: wrong kind in double")

  (* TODO noury: add more output types *)

  let scalar_gen extract node =
    extract node
    |> map ~f:(fun t -> Array.create 0 ~len:(Tensor.num_dims t) |> Tensor.get t)

  let scalar_float n = scalar_gen float n
  let scalar_double n = scalar_gen double n
  let scalar_int32 n = scalar_gen int32 n |> map ~f:Int32.to_int_exn
  let scalar_int64 n = scalar_gen int64 n

  let rec build_output
      : type a.
        a t -> (Node.p list -> Node.p list) * (Tensor.p list -> a * Tensor.p list)
    = function
    | Return a -> (fun l -> l), fun l -> a, l
    | Both (o1, o2) ->
      let l1, k1 = build_output o1 in
      let l2, k2 = build_output o2 in
      ( (fun l -> l1 (l2 l))
      , fun l ->
          let a, l = k1 l in
          let b, l = k2 l in
          (a, b), l )
    | Map (o, f) ->
      let l, k = build_output o in
      ( l
      , fun l ->
          let a, l = k l in
          f a, l )
    | Empty -> Fn.id, fun l -> (), l
    | Compute node ->
      ( (fun l -> P node :: l)
      , (function
        | t :: l -> t, l
        | [] -> failwith "wrong number of elts in output dispatch") )

  let build_output o =
    let f, k = build_output o in
    f [], fun l -> fst (k l)
end

let run ?inputs ?targets output =
  let inputs =
    Option.map
      inputs
      ~f:
        (List.map ~f:(fun (Input.I (n, t)) ->
             Node.P (Ops.Placeholder.to_node n), Tensor.P t))
  in
  let outputs, k = Output.build_output output in
  k (run ?inputs ?targets ~outputs ())

module Vars = struct
  let set input_fn var_and_tensors =
    let inputs, targets =
      List.map var_and_tensors ~f:(fun (var, tensor) ->
          let dims = Tensor.dims tensor |> Array.to_list in
          let placeholder = Ops.placeholder dims ~type_:(Node.output_type var) in
          let assign = Ops.assign var (Ops.Placeholder.to_node placeholder) in
          input_fn placeholder tensor, Node.P assign)
      |> List.unzip
    in
    run ~inputs ~targets Output.empty

  let set_float = set Input.float
  let set_double = set Input.double
end
