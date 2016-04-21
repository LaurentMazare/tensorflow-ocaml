(* See http://colah.github.io/posts/2015-08-Understanding-LSTMs
   for a simple description of LSTM networks.
*)
open Core_kernel.Std

let lstm ~size_c ~size_x =
  let create_vars () =
    Var.normalf [ size_c+size_x; size_c ] ~stddev:0.1, Var.f [ size_c ] 0.
  in
  let wf, bf = create_vars () in
  let wi, bi = create_vars () in
  let wC, bC = create_vars () in
  let wo, bo = create_vars () in
  Staged.stage (fun ~h ~x ~c ->
    let open Ops in
    let h_and_x = concat one32 [ h; x ] in
    let c =
      sigmoid (h_and_x *^ wf + bf) * c
      + sigmoid (h_and_x *^ wi + bi) * tanh (sigmoid (h_and_x *^ wC + bC))
    in
    let h = sigmoid (h_and_x *^ wo + bo) * tanh c in
    `h h, `c c)
