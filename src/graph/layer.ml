open Base
open Float.O_dot

let batch_normalization ?(decay = 0.9) t ~update_moments ~dims ~feature_count =
  let type_ = Node.output_type t in
  let zero = Ops.const_float ~type_ (List.init feature_count ~f:(fun _ -> 0.)) in
  let one = Ops.const_float ~type_ (List.init feature_count ~f:(fun _ -> 1.)) in
  let one_minus_decay = Ops.scalar ~type_ (1. -. decay) in
  let beta = Var.create [ feature_count ] ~type_ ~init:zero in
  let gamma = Var.create [ feature_count ] ~type_ ~init:one in
  let batch_moments = Ops.moments t ~dims:(List.init dims ~f:Fn.id) in
  let beta_with_update ~control_inputs =
    (* EWMA update. *)
    Ops.assignSub
      beta
      Ops.(one_minus_decay * (beta - batch_moments.mean))
      ~control_inputs
  in
  let gamma_with_update ~control_inputs =
    (* EWMA update. *)
    Ops.assignSub
      gamma
      Ops.(one_minus_decay * (gamma - batch_moments.variance))
      ~control_inputs
  in
  let beta, gamma =
    match update_moments with
    | `always ->
      Ops.identity beta ~control_inputs:[ Node.P (beta_with_update ~control_inputs:[]) ],
      Ops.identity gamma ~control_inputs:[ Node.P (gamma_with_update ~control_inputs:[]) ]
    | `not_in_testing testing ->
      let beta ~control_inputs:_ = beta in
      let gamma ~control_inputs:_ = gamma in
      Ops.cond_with_control_inputs testing ~if_true:beta ~if_false:beta_with_update,
      Ops.cond_with_control_inputs testing ~if_true:gamma ~if_false:gamma_with_update
  in
  Ops.normalize t { mean = beta; variance = gamma }
