open Core_kernel.Std
open Tensorflow
module O = Ops

(* ResNet model for the mnist dataset.
   This is mostly a work in progess for now.

   Reference:
   - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
*)
let label_count = Mnist_helper.label_count
let epochs = 10000
let batch_size = 1024

let hidden_nodes = 64

let scalar_tensor f =
  let array = Tensor.create1 Bigarray.float32 1 in
  Tensor.set array [| 0 |] f;
  array

let conv2d x ~filter ~stride:s ~kernel_size:k =
  let w = Var.normalf [ k; k; 1; filter ] ~stddev:0.1 in
  let b = Var.f [ filter ] 0. in
  let conv = O.conv2D x w ~strides:[ 1; s; s; 1 ] ~padding:"SAME" in
  O.add conv b

let avg_pool x ~stride:s ~kernel_size:k =
  O.avgPool x ~ksize:[ 1; k; k; 1 ] ~strides:[ 1; s; s; 1 ] ~padding:"SAME"

let basic_block input_layer ~filter ~stride =
  let layer = conv2d input_layer ~filter ~stride ~kernel_size:3 in
  let layer = O.relu layer in
  let layer = conv2d layer ~filter ~stride:1 ~kernel_size:1 in
  let shortcut = conv2d input_layer ~filter ~stride ~kernel_size:1 in
  let layer = O.add layer shortcut in
  O.relu layer

let build_model ~xs =
  let layer = O.reshape xs (O.const_int ~type_:Int32 [ -1; 28; 28; 1 ]) in
  (* 3x3 convolution + max-pool. *)
  let layer = conv2d layer ~filter:16 ~kernel_size:3 ~stride:1 in
  let layer = O.relu layer in

  let layer = basic_block layer ~filter:16 ~stride:1 in
  let layer = basic_block layer ~filter:32 ~stride:2 in
  let layer = basic_block layer ~filter:64 ~stride:2 in

  let layer = avg_pool ~stride:1 ~kernel_size:8 layer in
  (* Final dense layer. *)
  let output_dim = 7*7*64 in
  let layer = O.reshape layer (O.const_int ~type_:Int32 [ -1; output_dim ]) in
  let w1 = Var.normalf [ output_dim; hidden_nodes ] ~stddev:0.1 in
  let b1 = Var.f [ hidden_nodes ] 0. in
  let w2 = Var.normalf [ hidden_nodes; label_count ] ~stddev:0.1 in
  let b2 = Var.f [ label_count ] 0. in
  O.(relu (layer *^ w1 + b1) *^ w2 + b2) |> O.softmax

let () =
  let keep_prob = O.placeholder [] ~type_:Float in
  let mnist = Mnist_helper.read_files () in
  let xs = O.placeholder [] ~type_:Float in
  let ys = O.placeholder [] ~type_:Float in
  let ys_node = O.Placeholder.to_node ys in
  let ys_ = build_model ~xs:(O.Placeholder.to_node xs) in
  let cross_entropy = O.(neg (reduce_mean (ys_node * log ys_))) in
  let accuracy =
    O.(equal (argMax ys_ one32) (argMax ys_node one32))
    |> O.cast ~type_:Float
    |> O.reduce_mean
  in
  let gd = Optimizers.adam_minimizer ~learning_rate:(O.f 1e-4) cross_entropy in
  let validation_inputs =
    let one = scalar_tensor 1. in
    let validation_images = Tensor.sub_left mnist.test_images 0 1024 in
    let validation_labels = Tensor.sub_left mnist.test_labels 0 1024 in
    Session.Input.
      [ float xs validation_images; float ys validation_labels; float keep_prob one ]
  in
  let print_err n ~train_inputs =
    let vaccuracy, vcross_entropy =
      Session.run
        ~inputs:validation_inputs
        Session.Output.(both (scalar_float accuracy) (scalar_float cross_entropy))
    in
    let taccuracy, tcross_entropy =
      Session.run
        ~inputs:train_inputs
        Session.Output.(both (scalar_float accuracy) (scalar_float cross_entropy))
    in
    printf "epoch %d, train: %.2f%% (%8f) valid: %.2f%% (%8f)\n%!"
      n
      (100. *. taccuracy)
      tcross_entropy
      (100. *. vaccuracy)
      vcross_entropy
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
