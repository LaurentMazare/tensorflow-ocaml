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
  let xs = O.placeholder [-1; image_dim] ~type_:Float in
  let ys = O.placeholder [-1; label_count] ~type_:Float in

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
  let accuracy =
    O.(equal (argMax ~type_:Int32 ys_ one32) (argMax ~type_:Int32 (O.Placeholder.to_node ys) one32))
    |> O.cast ~type_:Float
    |> O.reduce_mean
  in
  let gd = Optimizers.adam_minimizer ~learning_rate:(O.f 1e-5) cross_entropy in
  let validation_inputs =
    let one = scalar_tensor 1. in
    let validation_images = Tensor.sub_left mnist.test_images 0 1024 in
    let validation_labels = Tensor.sub_left mnist.test_labels 0 1024 in
    Session.Input.
      [ float xs validation_images; float ys validation_labels; float keep_prob one ]
  in
  let print_err n ~train_inputs =
    let vaccuracy =
      Session.run
        ~inputs:validation_inputs
        (Session.Output.scalar_float accuracy)
    in
    let taccuracy =
      Session.run
        ~inputs:train_inputs
        (Session.Output.scalar_float accuracy)
    in
    Stdio.printf "epoch %d, vaccuracy %.2f%% taccuracy: %.2f%%\n%!" n (100. *. vaccuracy) (100. *. taccuracy)
  in
  for batch_idx = 1 to epochs do
    let batch_images, batch_labels =
      Mnist_helper.train_batch mnist ~batch_size ~batch_idx
    in
    let batch_inputs =
      let half = scalar_tensor 0.5 in
      Session.Input.[ float xs batch_images; float ys batch_labels; float keep_prob half ]
    in
    if batch_idx % 25 = 0 then print_err batch_idx ~train_inputs:batch_inputs;
    Session.run
      ~inputs:batch_inputs
      ~targets:gd
      Session.Output.empty;
  done
