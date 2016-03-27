open Core_kernel.Std
module H = Helper

let run ~samples ~size_xs ~size_ys ~xs ~ys =
  let xs = List.concat xs in
  let ys = List.concat ys in
  let xs = Ops_m.cf ~shape:[samples; size_xs] xs in
  let y  = Ops_m.cf ~shape:[samples; size_ys] ys in
  let w = Var.f [ size_xs; size_ys ] 0. in
  let b = Var.f [ size_ys ] 0. in
  let y_ = Ops_m.(xs *^ w + b) in
  let err = Ops_m.(Ops.square (y_ - y) |> reduce_mean) in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:0.04 ~varsf:[ w; b ] err
  in
  let session = Session.create () in
  let results = ref [] in
  let print_err n =
    let err, y_ = Session.run session Session.Output.(both (float err) (float y_)) in
    H.print_tensors [ Tensor.P err ] ~names:[ sprintf "err %d" n ];
    results := (n, Tensor.to_float_list (Tensor.P y_)) :: !results
  in
  for i = 0 to 2000 do
    Session.run session Session.Output.empty ~targets:gd;
    if i % 400 = 0 then print_err i
  done;
  !results
