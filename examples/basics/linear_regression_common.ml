open Base
open Tensorflow_core
open Tensorflow
module O = Ops

let run ~samples ~size_xs ~size_ys ~xs ~ys =
  let xs = List.concat xs in
  let ys = List.concat ys in
  let xs = O.cf ~shape:[ samples; size_xs ] xs in
  let y = O.cf ~shape:[ samples; size_ys ] ys in
  let w = Var.f [ size_xs; size_ys ] 0. in
  let b = Var.f [ size_ys ] 0. in
  let y_ = O.((xs *^ w) + b) in
  let err = O.(square (y_ - y) |> reduce_mean) in
  let gd =
    Optimizers.gradient_descent_minimizer ~learning_rate:(O.f 0.04) ~varsf:[ w; b ] err
  in
  let results = ref [] in
  let print_err n =
    let err, y_ = Session.run Session.Output.(both (float err) (float y_)) in
    Tensor.print (Tensor.P err);
    results := (n, Tensor.to_float_list (Tensor.P y_)) :: !results
  in
  for i = 0 to 2000 do
    Session.run Session.Output.empty ~targets:gd;
    if i % 400 = 0 then print_err i
  done;
  !results
