open Core_kernel.Std

let check_var (type a) (node : a Node.t) =
  if Node.Op_name.(<>) node.Node.op_name Ops.Op_names.variable
  then
    failwithf "Node %s is not a variable." (Node.Name.to_string node.name) ()

let get_shape var =
  Option.value_exn (Node.get_shape var)
  |> List.map ~f:(fun { Node.Dim.size; name = _ } -> size)

type t =
  { apply
    :  'a .gradient:([< `float | `double] as 'a) Node.t
    -> var:'a Node.t
    -> type_:'a Node.Type.t
    -> 'a Node.t
  }

let general_minimizer t ?(varsf = []) ?(varsd = []) target =
  let gradsf, gradsd =
    Gradients.gradient target
      ~with_respect_to_float:varsf
      ~with_respect_to_double:varsd
  in
  let apply gradients vars =
    List.map2_exn gradients vars ~f:(fun gradient var ->
      check_var var;
      let gradient = Ops.reshape gradient (Ops.shape var) in
      Node.P (t.apply ~gradient ~var ~type_:var.output_type))
  in
  let gdf =
    if not (List.is_empty varsf)
    then apply gradsf varsf
    else []
  in
  let gdd =
    if not (List.is_empty varsd)
    then apply gradsd varsd
    else []
  in
  gdf @ gdd

let maybe_cast node ~type_ =
  match Node.extract (Node.P node) type_ with
  | Some node -> node
  | None -> Ops.cast node ~type_

let gradient_descent_minimizer ~alpha ?varsf ?varsd target =
  let apply ~gradient ~var ~type_ =
    Ops.applyGradientDescent var (maybe_cast alpha ~type_) gradient
  in
  general_minimizer { apply } ?varsf ?varsd target

let momentum_minimizer ~alpha ~momentum ?varsf ?varsd target =
  let apply ~gradient ~var ~type_ =
    let accum =
      Var.create (get_shape var) ~type_ ~init:(Ops.zerosLike var)
    in
    Ops.applyMomentum var accum (maybe_cast alpha ~type_) gradient (maybe_cast momentum ~type_)
  in
  general_minimizer { apply } ?varsf ?varsd target

let adam_minimizer ~alpha ?beta1 ?beta2 ?epsilon ?varsf ?varsd target =
  let beta1 = match beta1 with | Some b -> b | None -> Ops.f 0.9 in
  let beta2 = match beta2 with | Some b -> b | None -> Ops.f 0.999 in
  let epsilon = match epsilon with | Some b -> b | None -> Ops.f 1e-8 in
  let apply ~gradient ~var ~type_ =
    let create_var () =
      Var.create (get_shape var) ~type_ ~init:(Ops.zerosLike var)
    in
    let create_scalar_var () =
      Var.create [] ~type_
        ~init:(Ops.scalar ~empty_shape:() ~type_ 0.)
    in
    Ops.applyAdam
      var
      (create_var ()) (* m *)
      (create_var ()) (* v *)
      (create_scalar_var ()) (* beta1_power *)
      (create_scalar_var ()) (* beta2_power *)
      (maybe_cast alpha ~type_)
      (maybe_cast beta1 ~type_)
      (maybe_cast beta2 ~type_)
      (maybe_cast epsilon ~type_)
      gradient
  in
  general_minimizer { apply } ?varsf ?varsd target

let adagrad_minimizer ~alpha ?init ?varsf ?varsd target =
  let init = match init with | Some b -> b | None -> Ops.f 0.1 in
  let apply ~gradient ~var ~type_ =
    let var_shape = Ops.shape var in
    let init = Ops.fill var_shape (maybe_cast init ~type_) in
    let accum = Var.create (get_shape var) ~type_ ~init in
    Ops.applyAdagrad var accum (maybe_cast alpha ~type_) gradient
  in
  general_minimizer { apply } ?varsf ?varsd target

let rmsprop_minimizer ~alpha ?decay ?momentum ?epsilon ?varsf ?varsd target =
  let decay = match decay with | Some b -> b | None -> Ops.f 0.9 in
  let momentum = match momentum with | Some b -> b | None -> Ops.f 0. in
  let epsilon = match epsilon with | Some b -> b | None -> Ops.f 1e-10 in
  let apply ~gradient ~var ~type_ =
    let rms_var =
      Var.create (get_shape var) ~type_ ~init:(Ops.zerosLike var)
    in
    let momentum_var = Var.create (get_shape var) ~type_ ~init:(Ops.zerosLike var) in
    Ops.applyRMSProp
      var
      rms_var
      momentum_var
      (maybe_cast alpha ~type_)
      (maybe_cast decay ~type_)
      (maybe_cast momentum ~type_)
      (maybe_cast epsilon ~type_)
      gradient
  in
  general_minimizer { apply } ?varsf ?varsd target
