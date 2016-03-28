open Core_kernel.Std

let check_var (type a) (node : a Node.t) =
  if Node.Op_name.(<>) node.Node.op_name Ops.Op_names.variable
  then
    failwithf "Node %s is not a variable." (Node.Name.to_string node.name) ()

let get_shape var =
  Option.value_exn (Node.get_shape var)
  |> List.map ~f:(fun { Node.Dim.size; name = _ } -> size)

let gradient_descent_minimizer ~alpha ?(varsf = []) ?(varsd = []) target =
  let gradsf, gradsd =
    Gradients.gradient target
      ~with_respect_to_float:varsf
      ~with_respect_to_double:varsd
  in
  let apply_gradient_descent grads vars ~alpha =
    List.map2_exn grads vars ~f:(fun grad var ->
      check_var var;
      let grad = Ops.reshape grad (Ops.shape var) in
      Node.P (Ops.applyGradientDescent var alpha grad))
  in
  let gdf =
    if not (List.is_empty varsf)
    then apply_gradient_descent gradsf varsf ~alpha
    else []
  in
  let gdd =
    if not (List.is_empty varsd)
    then apply_gradient_descent gradsd varsd ~alpha:(Ops.cast alpha ~type_:Double)
    else []
  in
  gdf @ gdd

let momentum_minimizer ~alpha ~momentum ?(varsf = []) ?(varsd = []) target =
  let gradsf, gradsd =
    Gradients.gradient target
      ~with_respect_to_float:varsf
      ~with_respect_to_double:varsd
  in
  let apply_momentum grads vars ~alpha ~momentum =
    List.map2_exn grads vars ~f:(fun grad var ->
      let var_shape = Ops.shape var in
      let accum =
        Var.create (get_shape var) ~type_:var.output_type
          ~init:(Ops.fill var_shape (Ops.scalar ~empty_shape:() ~type_:var.output_type 0.))
      in
      let grad = Ops.reshape grad var_shape in
      Node.P (Ops.applyMomentum var accum alpha grad momentum))
  in
  let gdf =
    if not (List.is_empty varsf)
    then apply_momentum gradsf varsf ~alpha ~momentum
    else []
  in
  let gdd =
    if not (List.is_empty varsd)
    then
      apply_momentum gradsd varsd
        ~alpha:(Ops.cast alpha ~type_:Double)
        ~momentum:(Ops.cast momentum ~type_:Double)
    else []
  in
  gdf @ gdd

let adam_minimizer ~alpha ?beta1 ?beta2 ?epsilon ?(varsf = []) ?(varsd = []) target =
  let beta1 =
    match beta1 with
    | None -> Ops.f 0.9
    | Some v -> v
  in
  let beta2 =
    match beta2 with
    | None -> Ops.f 0.999
    | Some v -> v
  in
  let epsilon =
    match epsilon with
    | None -> Ops.f 1e-8
    | Some v -> v
  in
  let gradsf, gradsd =
    Gradients.gradient target
      ~with_respect_to_float:varsf
      ~with_respect_to_double:varsd
  in
  let apply_adam grads vars ~alpha ~beta1 ~beta2 ~epsilon =
    List.map2_exn grads vars ~f:(fun grad var ->
      let var_shape = Ops.shape var in
      let create_var () =
        Var.create (get_shape var) ~type_:var.output_type
          ~init:(Ops.fill var_shape (Ops.scalar ~empty_shape:() ~type_:var.output_type 0.))
      in
      let create_scalar_var () =
        Var.create [] ~type_:var.output_type
          ~init:(Ops.scalar ~empty_shape:() ~type_:var.output_type 0.)
      in
      let grad = Ops.reshape grad var_shape in
      let adam =
        Ops.applyAdam
          var
          (create_var ()) (* m *)
          (create_var ()) (* v *)
          (create_scalar_var ()) (* beta1_power *)
          (create_scalar_var ()) (* beta2_power *)
          alpha
          beta1
          beta2
          epsilon
          grad
      in
      Node.P adam)
  in
  let gdf =
    if not (List.is_empty varsf)
    then apply_adam gradsf varsf ~alpha ~beta1 ~beta2 ~epsilon
    else []
  in
  let gdd =
    if not (List.is_empty varsd)
    then
      apply_adam gradsd varsd
        ~alpha:(Ops.cast alpha ~type_:Double)
        ~beta1:(Ops.cast beta1 ~type_:Double)
        ~beta2:(Ops.cast beta2 ~type_:Double)
        ~epsilon:(Ops.cast epsilon ~type_:Double)
    else []
  in
  gdf @ gdd
