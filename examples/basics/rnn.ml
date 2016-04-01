(* See http://colah.github.io/posts/2015-08-Understanding-LSTMs
   for a simple description of LSTM networks.
*)
open Core_kernel.Std
open Tensorflow

let lstm ~size_c ~size_x ~size_y x_and_ys =
  let create_vars () = Var.f [ size_c+size_x; size_c ] 0., Var.f [ size_c ] 0. in
  let zero = Ops.f ~shape:[ size_c ] 0. in
  let wf, bf = create_vars () in
  let wi, bi = create_vars () in
  let wC, bC = create_vars () in
  let wo, bo = create_vars () in
  let wy, by = Var.f [ size_c; size_y ] 0., Var.f [ size_y ] 0. in
  let one_lstm ~h ~x ~c =
    let open Ops in
    let h_and_x = concat zero32 [ h; x ] in
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
    |> fun (errs, _, _) -> List.rev errs
    |> Ops.concat Ops.zero32
    |> Ops.square
    |> Ops.reduce_mean
  in
  err, one_lstm

let () =
  ignore (lstm, ())
