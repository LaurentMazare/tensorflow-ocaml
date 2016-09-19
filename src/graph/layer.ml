open Core_kernel.Std

let batch_normalization ?(decay = 0.99) t ~testing ~dims ~feature_count =
  let type_ = Node.output_type t in
  let zero = Ops.const_float ~type_ (List.init feature_count ~f:(fun _ -> 0.)) in
  let one = Ops.const_float ~type_ (List.init feature_count ~f:(fun _ -> 1.)) in
  let one_minus_decay = Ops.scalar ~type_ (1. -. decay) in
  let beta = Var.create [ feature_count ] ~type_ ~init:zero in
  let gamma = Var.create [ feature_count ] ~type_ ~init:one in
  let batch_moments = Ops.moments t ~dims:(List.init dims ~f:Fn.id) in
  let beta_with_update =
    let update_beta =
      (* EWMA update. *)
      Ops.assignSub
        beta
        Ops.(one_minus_decay * (beta - batch_moments.mean))
    in
    Ops.identity ~control_inputs:[ Node.P update_beta ] beta
  in
  let gamma_with_update =
      (* EWMA update. *)
    let update_gamma =
      Ops.assignSub
        gamma
        Ops.(one_minus_decay * (gamma - batch_moments.variance))
    in
    Ops.identity ~control_inputs:[ Node.P update_gamma ] gamma
  in
  let beta = Ops.cond testing ~if_true:beta ~if_false:beta_with_update in
  let gamma = Ops.cond testing ~if_true:gamma ~if_false:gamma_with_update in
  Ops.normalize t { mean = beta; variance = gamma }
