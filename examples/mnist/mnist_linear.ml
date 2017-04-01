open Base
open Float.O_dot
open Tensorflow
module O = Ops

(* This should reach ~92% accuracy. *)
let image_dim = Mnist_helper.image_dim
let label_count = Mnist_helper.label_count
let epochs = 300

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let xs = O.placeholder [] ~type_:Float in
  let ys = O.placeholder [] ~type_:Float in
  let ys_node = O.Placeholder.to_node ys in
  let w = Var.f [ image_dim; label_count ] 0. in
  let b = Var.f [ label_count ] 0. in
  let ys_ = O.(Placeholder.to_node xs *^ w + b) |> O.softmax in
  let cross_entropy = O.cross_entropy ~ys:ys_node ~y_hats:ys_ `mean in
  let accuracy =
    O.(equal (argMax ys_ O.one32) (argMax ys_node O.one32))
    |> O.cast ~type_:Float
    |> O.reduce_mean
  in
  let gd =
    Optimizers.gradient_descent_minimizer ~learning_rate:(O.f 8.) cross_entropy
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
    Stdio.printf "epoch %d, accuracy %.2f%%\n%!" n (100. *. accuracy)
  in
  for i = 1 to epochs do
    if i % 50 = 0 then print_err i;
    Session.run
      ~inputs:train_inputs
      ~targets:gd
      Session.Output.empty;
  done
