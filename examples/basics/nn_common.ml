open Base
open Tensorflow
module O = Ops

let one_layer ~samples ~size_xs ~size_ys ~xs ~ys ~hidden_nodes ~epochs =
  let xs = List.concat xs in
  let ys = List.concat ys in
  let xs = O.cf ~shape:[ samples; size_xs ] xs in
  let y = O.cf ~shape:[ samples; size_ys ] ys in
  let w1 = Var.normalf [ size_xs; hidden_nodes ] ~stddev:0.1 in
  let b1 = Var.f [ hidden_nodes ] 0. in
  let w2 = Var.normalf [ hidden_nodes; size_ys ] ~stddev:0.1 in
  let b2 = Var.f [ size_ys ] 0. in
  let y_ = O.((sigmoid ((xs *^ w1) + b1) *^ w2) + b2) in
  let err = O.(square (y_ - y) |> reduce_mean) in
  let gd =
    Optimizers.gradient_descent_minimizer
      ~learning_rate:(O.f 0.05)
      ~varsf:[ w1; w2; b1; b2 ]
      err
  in
  let results = ref [] in
  let print_err n =
    let err, y_ = Session.run Session.Output.(both (float err) (float y_)) in
    Tensor.print (Tensor.P err);
    results := (n, Tensor.to_float_list (Tensor.P y_)) :: !results
  in
  for i = 0 to epochs do
    Session.run Session.Output.empty ~targets:gd;
    if i % (epochs / 5) = 0 then print_err i
  done;
  !results
