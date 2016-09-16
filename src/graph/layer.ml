
let batch_normalization t ~testing ~shape =
  let type_ = Node.output_type t in
  let zero = Ops.const_float ~type_ [ 0. ] in
  let one = Ops.const_float ~type_ [ 1. ] in
  let beta = Var.create shape ~type_ ~init:zero in
  let gamma = Var.create shape ~type_ ~init:one in
  let batch_moments = Ops.moments t ~dims:[ 0 ] in
  let beta =
    let update_beta =
      Ops.assign beta batch_moments.mean
    in
    Ops.select
      testing
      beta
      (Ops.identity ~control_inputs:[ Node.P update_beta ] beta)
  in
  let gamma =
    let update_gamma =
      Ops.assign gamma batch_moments.variance
    in
    Ops.select
      testing
      gamma
      (Ops.identity ~control_inputs:[ Node.P update_gamma ] gamma)
  in
  Ops.normalize t { mean = beta; variance = gamma }

