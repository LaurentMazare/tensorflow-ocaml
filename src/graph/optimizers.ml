open Base

type 'a optimizer =
  learning_rate:[ `float ] Node.t
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> 'a Node.t
  -> Node.p list

(* Collect float and double variables below a given node.
   Using this is an overapproximation as we would only need the variables that
   can be reached from the node via a 'derivable' path. *)
let get_all_vars node =
  let processed_nodes = Hash_set.create (module Node.Id) in
  (* Using references here make the following code quite consise. *)
  let varsf = ref ([] : [ `float ] Node.t list) in
  let varsd = ref ([] : [ `double ] Node.t list) in
  let rec vars (Node.P node) =
    if not (Hash_set.mem processed_nodes (Node.id node))
    then (
      Hash_set.add processed_nodes (Node.id node);
      if Node.Op_name.( = ) (Node.op_name node) Ops.Op_names.variable
      then (
        match Node.output_type node with
        | Node.Type.Float -> varsf := node :: !varsf
        | Node.Type.Double -> varsd := node :: !varsd
        | _ -> ())
      else List.iter (Node.flat_inputs node) ~f:vars)
  in
  vars (Node.P node);
  !varsf, !varsd

let check_var (type a) (node : a Node.t) =
  if Node.Op_name.( <> ) (Node.op_name node) Ops.Op_names.variable
  then
    Printf.failwithf
      "Node %s is not a variable."
      (Node.Name.to_string (Node.name node))
      ()

type t =
  { apply :
      'a. gradient:([< `float | `double ] as 'a) Node.t -> var:'a Node.t
      -> type_:'a Node.Type.t -> 'a Node.t
  }

let general_minimizer t ?varsf ?varsd target =
  let varsf, varsd =
    match varsf, varsd with
    | Some varsf, Some varsd -> varsf, varsd
    | Some varsf, None -> varsf, snd (get_all_vars target)
    | None, Some varsd -> fst (get_all_vars target), varsd
    | None, None -> get_all_vars target
  in
  let gradsf, gradsd =
    match Caml.Sys.getenv_opt "WITH_TF_BACKPROP" with
    | None | Some "" | Some "false" ->
      Gradients.gradient_caml
        target
        ~with_respect_to_float:varsf
        ~with_respect_to_double:varsd
    | Some "true" | Some _ ->
      Gradients.gradient_tf
        target
        ~with_respect_to_float:varsf
        ~with_respect_to_double:varsd
  in
  let apply gradients vars =
    List.map2_exn gradients vars ~f:(fun gradient var ->
        check_var var;
        let gradient = Ops.reshape gradient (Ops.shape32 var) in
        Node.P (t.apply ~gradient ~var ~type_:(Node.output_type var)))
  in
  let gdf = if not (List.is_empty varsf) then apply gradsf varsf else [] in
  let gdd = if not (List.is_empty varsd) then apply gradsd varsd else [] in
  gdf @ gdd

let maybe_cast node ~type_ =
  match Node.extract (Node.P node) type_ with
  | Some node -> node
  | None -> Ops.cast node ~type_

let gradient_descent_minimizer ~learning_rate ?varsf ?varsd target =
  let apply ~gradient ~var ~type_ =
    Ops.applyGradientDescent var (maybe_cast learning_rate ~type_) gradient
  in
  general_minimizer { apply } ?varsf ?varsd target

let momentum_minimizer ~momentum ~learning_rate ?varsf ?varsd target =
  let apply ~gradient ~var ~type_ =
    let accum = Var.create (Node.shape var) ~type_ ~init:(Ops.zerosLike var) in
    Ops.applyMomentum
      var
      accum
      (maybe_cast learning_rate ~type_)
      gradient
      (maybe_cast momentum ~type_)
  in
  general_minimizer { apply } ?varsf ?varsd target

let adam_minimizer
    ?(beta1 = Ops.f 0.9)
    ?(beta2 = Ops.f 0.999)
    ?(epsilon = Ops.f 1e-8)
    ~learning_rate
    ?varsf
    ?varsd
    target
  =
  let apply ~gradient ~var ~type_ =
    let create_var () = Var.create (Node.shape var) ~type_ ~init:(Ops.zerosLike var) in
    let create_scalar_var () =
      Var.create [] ~type_ ~init:(Ops.scalar ~empty_shape:() ~type_ 0.)
    in
    Ops.applyAdam
      var
      (create_var ()) (* m *)
      (create_var ()) (* v *)
      (create_scalar_var ()) (* beta1_power *)
      (create_scalar_var ()) (* beta2_power *)
      (maybe_cast learning_rate ~type_)
      (maybe_cast beta1 ~type_)
      (maybe_cast beta2 ~type_)
      (maybe_cast epsilon ~type_)
      gradient
  in
  general_minimizer { apply } ?varsf ?varsd target

let adagrad_minimizer ?(init = Ops.f 0.1) ~learning_rate ?varsf ?varsd target =
  let apply ~gradient ~var ~type_ =
    let var_shape = Ops.shape32 var in
    let init = Ops.fill var_shape (maybe_cast init ~type_) in
    let accum = Var.create (Node.shape var) ~type_ ~init in
    Ops.applyAdagrad var accum (maybe_cast learning_rate ~type_) gradient
  in
  general_minimizer { apply } ?varsf ?varsd target

let rmsprop_minimizer
    ?(decay = Ops.f 0.9)
    ?(momentum = Ops.f 0.)
    ?(epsilon = Ops.f 1e-10)
    ~learning_rate
    ?varsf
    ?varsd
    target
  =
  let apply ~gradient ~var ~type_ =
    let rms_var = Var.create (Node.shape var) ~type_ ~init:(Ops.zerosLike var) in
    let momentum_var = Var.create (Node.shape var) ~type_ ~init:(Ops.zerosLike var) in
    Ops.applyRMSProp
      var
      rms_var
      momentum_var
      (maybe_cast learning_rate ~type_)
      (maybe_cast decay ~type_)
      (maybe_cast momentum ~type_)
      (maybe_cast epsilon ~type_)
      gradient
  in
  general_minimizer { apply } ?varsf ?varsd target
