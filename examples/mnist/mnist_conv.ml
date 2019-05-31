open Base
open Float.O_dot
open Tensorflow_core
open Tensorflow
module O = Ops

let image_dim = Mnist_helper.image_dim
let label_count = Mnist_helper.label_count

let scalar_tensor f =
  let array = Tensor.create1 Bigarray.float32 1 in
  Tensor.set array [| 0 |] f;
  array

let batch_size = 512
let epochs = 5000

let () =
  let mnist = Mnist_helper.read_files () in
  let keep_prob = O.placeholder [] ~type_:Float in
  let xs = O.placeholder [ -1; image_dim ] ~type_:Float in
  let ys = O.placeholder [ -1; label_count ] ~type_:Float in
  let ys_ =
    O.Placeholder.to_node xs
    |> Layer.reshape ~shape:[ -1; 28; 28; 1 ]
    |> Layer.conv2d ~ksize:(5, 5) ~strides:(1, 1) ~output_dim:32
    |> Layer.max_pool ~ksize:(2, 2) ~strides:(2, 2)
    |> Layer.conv2d ~ksize:(5, 5) ~strides:(1, 1) ~output_dim:64
    |> Layer.max_pool ~ksize:(2, 2) ~strides:(2, 2)
    |> Layer.flatten
    |> Layer.linear ~output_dim:1024 ~activation:Relu
    |> O.dropout ~keep_prob:(O.Placeholder.to_node keep_prob)
    |> Layer.linear ~output_dim:10 ~activation:Softmax
  in
  let cross_entropy = O.cross_entropy ~ys:(O.Placeholder.to_node ys) ~y_hats:ys_ `sum in
  let gd = Optimizers.adam_minimizer ~learning_rate:(O.f 1e-5) cross_entropy in
  let one = scalar_tensor 1. in
  let predict images =
    Session.run
      (Session.Output.float ys_)
      ~inputs:Session.Input.[ float xs images; float keep_prob one ]
  in
  let print_err n =
    let test_accuracy =
      Mnist_helper.batch_accuracy mnist `test ~batch_size:1024 ~predict
    in
    let train_accuracy =
      Mnist_helper.batch_accuracy mnist `train ~batch_size:1024 ~predict ~samples:5000
    in
    Stdio.printf
      "epoch %d, train: %.2f%% valid: %.2f%%\n%!"
      n
      (100. *. train_accuracy)
      (100. *. test_accuracy)
  in
  let half = scalar_tensor 0.5 in
  for batch_idx = 1 to epochs do
    let batch_images, batch_labels =
      Mnist_helper.train_batch mnist ~batch_size ~batch_idx
    in
    if batch_idx % 100 = 0 then print_err batch_idx;
    Session.run
      ~inputs:
        Session.Input.
          [ float xs batch_images; float ys batch_labels; float keep_prob half ]
      ~targets:gd
      Session.Output.empty
  done
