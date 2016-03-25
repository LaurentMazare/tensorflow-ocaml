open Core_kernel.Std

let gradient_descent_minimizer ~alpha ?(varsf = []) ?(varsd = []) target =
  let gradsf, gradsd =
    Gradients.gradient target
      ~with_respect_to_float:varsf
      ~with_respect_to_double:varsd
  in
  let apply_gradient_descent grads vars ~alpha =
    List.map2_exn grads vars ~f:(fun grad var ->
      let grad = Ops.reshape grad (Ops.shape var) in
      Node.P (Ops.applyGradientDescent var alpha grad))
  in
  let gdf =
    if not (List.is_empty varsf)
    then apply_gradient_descent gradsf varsf ~alpha:(Ops_m.f alpha)
    else []
  in
  let gdd =
    if not (List.is_empty varsd)
    then apply_gradient_descent gradsd varsd ~alpha:(Ops_m.d alpha)
    else []
  in
  gdf @ gdd
