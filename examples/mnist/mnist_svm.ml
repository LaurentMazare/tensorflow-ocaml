open Base
open Float.O_dot
open Tensorflow

let image_dim = Mnist_helper.image_dim
let label_count = Mnist_helper.label_count
let epochs = 10000
let lambda = 10.0

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let xs = Ops.placeholder [] ~type_:Float in
  let ys = Ops.placeholder [] ~type_:Float in
  let ys_node = Ops.Placeholder.to_node ys in
  let w = Var.f [ image_dim; label_count ] 0. in
  let b = Var.f [ label_count ] 0. in
  let ys_ = Ops.((Placeholder.to_node xs *^ w) - b) in
  let accuracy =
    Ops.equal
      (Ops.argMax ~type_:Int32 ys_ Ops.one32)
      (Ops.argMax ~type_:Int32 ys_node Ops.one32)
    |> Ops.cast ~type_:Float
    |> Ops.reduce_mean
  in
  (* ys in {-1 ; + 1} *)
  let ys' = Ops.((f 2.0 * ys_node) - f 1.0) in
  let distance_from_margin = Ops.(f 1.0 - (ys' * ys_)) in
  let hinge_loss = Ops.relu distance_from_margin |> Ops.reduce_mean in
  let square_norm = Ops.(f lambda * reduce_sum (w * w)) in
  let gd =
    Optimizers.gradient_descent_minimizer
      ~learning_rate:(Ops.f 0.1)
      ~varsf:[ w; b ]
      Ops.(square_norm + hinge_loss)
  in
  let train_inputs = Session.Input.[ float xs train_images; float ys train_labels ] in
  let validation_inputs = Session.Input.[ float xs test_images; float ys test_labels ] in
  let print_err n =
    let vaccuracy =
      Session.run ~inputs:validation_inputs (Session.Output.scalar_float accuracy)
    in
    let taccuracy =
      Session.run ~inputs:train_inputs (Session.Output.scalar_float accuracy)
    in
    Stdio.printf
      "epoch %d, accuracy %.2f%% train accuracy %.2f%%\n%!"
      n
      (100. *. vaccuracy)
      (100. *. taccuracy)
  in
  for i = 1 to epochs do
    if i % 100 = 0 then print_err i;
    Session.run ~inputs:train_inputs ~targets:gd Session.Output.empty
  done
