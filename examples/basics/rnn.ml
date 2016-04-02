(* See http://colah.github.io/posts/2015-08-Understanding-LSTMs
   for a simple description of LSTM networks.
*)
open Core_kernel.Std
open Tensorflow

let lstm ~size_c ~size_x ~size_y x_and_ys =
  let create_vars () = Var.f [ size_c+size_x; size_c ] 0., Var.f [ size_c ] 0. in
  let zero = Ops.f ~shape:[ 1; size_c ] 0. in
  let wf, bf = create_vars () in
  let wi, bi = create_vars () in
  let wC, bC = create_vars () in
  let wo, bo = create_vars () in
  let wy, by = Var.f [ size_c; size_y ] 0., Var.f [ size_y ] 0. in
  let one_lstm ~h ~x ~c =
    let open Ops in
    let h_and_x = concat one32 [ h; x ] in
    let c =
      sigmoid (h_and_x *^ wf + bf) * c
      + sigmoid (h_and_x *^ wi + bi) * tanh (sigmoid (h_and_x *^ wC + bC))
    in
    let h = sigmoid (h_and_x *^ wo + bo) * tanh c in
    h, c
  in
  let err =
    List.fold x_and_ys ~init:([], zero, zero) ~f:(fun (errs, h, c) (x, y) ->
      let h, c = one_lstm ~h ~x ~c in
      let err = Ops.(h *^ wy + by - y) in
      err :: errs, h, c)
    |> fun (errs, _, _) -> Ops.concat Ops.one32 errs
    |> Ops.square
    |> Ops.reduce_mean
  in
  err, one_lstm

let epochs = 100
let size_c = 20
let steps = 100
let step_size = 0.05

let fit_1d fn =
  let x_and_ys =
    List.init steps ~f:(fun x ->
      let x = float x *. step_size in
      let xs = Ops.const_float ~type_:Float ~shape:[ 1; 1 ] [ fn x ] in
      let ys = Ops.const_float ~type_:Float ~shape:[ 1; 1 ] [ fn (x +. step_size) ] in
      xs, ys
    )
  in
  let err, one_lstm = lstm ~size_c ~size_x:1 ~size_y:1 x_and_ys in
  let gd = Optimizers.gradient_descent_minimizer err ~alpha:(Ops.f 0.4) in
  for i = 1 to epochs do
    let err =
      Session.run
        ~inputs:[]
        ~targets:(ignore gd; [])
        (Session.Output.scalar_float err);
    in
    printf "%d %f\n%!" i err;
  done;
  ignore (one_lstm, ())

let () =
  fit_1d sin
