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
    -> alpha:'a Node.t
    -> 'a Node.t
  }

let general_minimizer t ~alpha ?(varsf = []) ?(varsd = []) target =
  let gradsf, gradsd =
    Gradients.gradient target
      ~with_respect_to_float:varsf
      ~with_respect_to_double:varsd
  in
  let apply gradients vars ~alpha =
    List.map2_exn gradients vars ~f:(fun gradient var ->
      check_var var;
      let gradient = Ops.reshape gradient (Ops.shape var) in
      Node.P (t.apply ~gradient ~var ~alpha))
  in
  let gdf =
    if not (List.is_empty varsf)
    then apply gradsf varsf ~alpha
    else []
  in
  let gdd =
    if not (List.is_empty varsd)
    then apply gradsd varsd ~alpha:(Ops.cast alpha ~type_:Double)
    else []
  in
  gdf @ gdd

let gradient_descent_minimizer ~alpha ?varsf ?varsd target =
  let apply ~gradient ~var ~alpha = Ops.applyGradientDescent var alpha gradient in
  general_minimizer { apply } ~alpha ?varsf ?varsd target

let momentum_minimizer ~alpha ~momentum ?varsf ?varsd target =
  let apply ~gradient ~var ~alpha =
    let var_shape = Ops.shape var in
    let accum =
      Var.create (get_shape var) ~type_:var.output_type
        ~init:(Ops.fill var_shape (Ops.scalar ~empty_shape:() ~type_:var.output_type 0.))
    in
    let momentum = Ops.scalar ~empty_shape:() ~type_:var.output_type momentum in
    Ops.applyMomentum var accum alpha gradient momentum
  in
  general_minimizer { apply } ~alpha ?varsf ?varsd target

let adam_minimizer ~alpha ?(beta1=0.9) ?(beta2=0.999) ?(epsilon=1e-8) ?varsf ?varsd target =
  let apply ~gradient ~var ~alpha =
    let var_shape = Ops.shape var in
    let create_var () =
      Var.create (get_shape var) ~type_:var.output_type
        ~init:(Ops.fill var_shape (Ops.scalar ~empty_shape:() ~type_:var.output_type 0.))
    in
    let create_scalar_var () =
      Var.create [] ~type_:var.output_type
        ~init:(Ops.scalar ~empty_shape:() ~type_:var.output_type 0.)
    in
    Ops.applyAdam
      var
      (create_var ()) (* m *)
      (create_var ()) (* v *)
      (create_scalar_var ()) (* beta1_power *)
      (create_scalar_var ()) (* beta2_power *)
      alpha
      (Ops.scalar ~empty_shape:() ~type_:var.output_type beta1)
      (Ops.scalar ~empty_shape:() ~type_:var.output_type beta2)
      (Ops.scalar ~empty_shape:() ~type_:var.output_type epsilon)
      gradient
  in
  general_minimizer { apply } ~alpha ?varsf ?varsd target

let adagrad_minimizer ~alpha ?(init=0.1) ?varsf ?varsd target =
  let apply ~gradient ~var ~alpha =
    let var_shape = Ops.shape var in
    let init =
      Ops.fill var_shape (Ops.scalar ~empty_shape:() ~type_:var.output_type init)
    in
    let accum = Var.create (get_shape var) ~type_:var.output_type ~init in
    Ops.applyAdagrad var accum alpha gradient
  in
  general_minimizer { apply } ~alpha ?varsf ?varsd target
