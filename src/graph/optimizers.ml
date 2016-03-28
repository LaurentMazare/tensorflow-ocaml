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
