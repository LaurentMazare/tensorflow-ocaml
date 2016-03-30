open Core_kernel.Std
open Tensorflow
module O = Ops

(* This should reach ~97% accuracy. *)
let image_dim = Mnist_helper.image_dim
let label_count = Mnist_helper.label_count
let hidden_nodes = 128
let epochs = 1000

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let xs = O.placeholder [] ~type_:Float in
  let ys = O.placeholder [] ~type_:Float in
  let w1 = Var.normalf [ image_dim; hidden_nodes ] ~stddev:0.1 in
  let b1 = Var.f [ hidden_nodes ] 0. in
  let w2 = Var.normalf [ hidden_nodes; label_count ] ~stddev:0.1 in
  let b2 = Var.f [ label_count ] 0. in
  let ys_ = O.(relu (xs *^ w1 + b1) *^ w2 + b2) |> O.softmax in
  let cross_entropy = O.(neg (reduce_mean (ys * log ys_))) in
  let accuracy =
    O.(equal (argMax ys_ one32) (argMax ys one32))
    |> O.cast ~type_:Float
    |> O.reduce_mean
  in
  let gd =
    Optimizers.momentum_minimizer cross_entropy
      ~alpha:(O.f 0.6) ~momentum:0.9 ~varsf:[ w1; w2; b1; b2 ]
  in
  let train_inputs = Session.Input.[ float xs train_images; float ys train_labels ] in
  let validation_inputs =
    Session.Input.[ float xs test_images; float ys test_labels ]
  in
  let print_err n =
    let accuracy =
      Session.run
        ~inputs:validation_inputs
        (Session.Output.scalar_float accuracy)
    in
    printf "epoch %d, accuracy %.2f%%\n%!" n (100. *. accuracy)
  in
  for i = 1 to epochs do
    if i % 50 = 0 then print_err i;
    Session.run
      ~inputs:train_inputs
      ~targets:gd
      Session.Output.empty;
  done
