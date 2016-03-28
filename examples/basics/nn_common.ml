open Core_kernel.Std
module H = Helper

let one_layer ~samples ~size_xs ~size_ys ~xs ~ys ~hidden_nodes ~epochs =
  let xs = List.concat xs in
  let ys = List.concat ys in
  let xs = Ops_m.cf ~shape:[samples; size_xs] xs in
  let y  = Ops_m.cf ~shape:[samples; size_ys] ys in
  let w1 = Var.normalf [ size_xs; hidden_nodes ] ~stddev:0.1 in
  let b1 = Var.f [ hidden_nodes ] 0. in
  let w2 = Var.normalf [ hidden_nodes; size_ys ] ~stddev:0.1 in
  let b2 = Var.f [ size_ys ] 0. in
  let y_ = Ops_m.(Ops.sigmoid (xs *^ w1 + b1) *^ w2 + b2) in
  let err = Ops_m.(Ops.square (y_ - y) |> reduce_mean) in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:(Ops_m.f 0.05) ~varsf:[ w1; w2; b1; b2 ] err
  in
  let session = Session.create () in
  let results = ref [] in
  let print_err n =
    let err, y_ =
      Session.run session Session.Output.(both (float err) (float y_))
    in
    H.print_tensors [ Tensor.P err ] ~names:[ sprintf "err %d" n ];
    results := (n, Tensor.to_float_list (Tensor.P y_)) :: !results
  in
  for i = 0 to epochs do
    Session.run session Session.Output.empty ~targets:gd;
    if i % (epochs / 5) = 0 then print_err i
  done;
  !results
