open Base
open Tensorflow
module O = Ops

(* ResNet model for the mnist dataset.
   This is mostly a work in progess for now. Batch normalization is not included.

   Reference:
     - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
*)
let label_count = Mnist_helper.label_count
let epochs = 100_000
let batch_size = 128

let depth = 20
(* Number of hidden nodes in the final layer. *)
let hidden_nodes = 64

let conv2d x ~out_features ~in_features ~stride:s ~kernel_size:k =
  let w = Var.normalf [ k; k; in_features; out_features ] ~stddev:0.1 in
  let b = Var.f [ out_features ] 0. in
  let conv = O.conv2D x w ~strides:[ 1; s; s; 1 ] ~padding:"SAME" in
  O.add conv b

let avg_pool x ~stride:s =
  O.avgPool x ~ksize:[ 1; s; s; 1 ] ~strides:[ 1; s; s; 1 ] ~padding:"VALID"

let basic_block input_layer ~out_features ~in_features ~stride ~testing =
  ignore testing;
  let shortcut =
    if stride = 1
    then input_layer
    else
      let half_diff = (out_features - in_features) / 2 in
      O.pad (avg_pool input_layer ~stride)
        (O.const_int ~shape:[ 4; 2 ] ~type_:Int32 [ 0; 0; 0; 0; 0; 0; half_diff; half_diff ])
  in
  conv2d input_layer ~out_features ~in_features ~stride ~kernel_size:3
  |> Layer.batch_normalization ~update_moments:`always ~dims:3 ~feature_count:out_features
  |> O.relu
  |> conv2d ~out_features ~in_features:out_features ~stride:1 ~kernel_size:1
  |> O.add shortcut
  |> Layer.batch_normalization ~update_moments:`always ~dims:3 ~feature_count:out_features
  (* No ReLU after the add as per http://torch.ch/blog/2016/02/04/resnets.html *)

let block_stack layer ~out_features ~in_features ~stride ~testing =
  let depth = (depth - 2) / 6 in
  let layer = basic_block layer ~out_features ~in_features ~stride ~testing in
  List.init (depth - 1) ~f:Fn.id
  |> List.fold ~init:layer ~f:(fun layer _idx ->
    basic_block layer ~out_features ~in_features:out_features ~stride:1 ~testing)

let build_model ~xs ~testing =
  let layer = O.reshape xs (O.const_int ~type_:Int32 [ -1; 28; 28; 1 ]) in
  (* 3x3 convolution + max-pool. *)
  let layer = conv2d layer ~out_features:16 ~in_features:1 ~kernel_size:3 ~stride:1 in
  let layer = O.relu layer in

  let layer = block_stack layer ~out_features:16 ~in_features:16 ~stride:1 ~testing in
  let layer = block_stack layer ~out_features:32 ~in_features:16 ~stride:2 ~testing in
  let layer = block_stack layer ~out_features:64 ~in_features:32 ~stride:2 ~testing in

  let layer = O.reduce_mean layer ~dims:[ 1; 2 ] in
  (* Final dense layer. *)
  let output_dim = 64 in
  let layer = O.reshape layer (O.const_int ~type_:Int32 [ -1; output_dim ]) in
  let w1 = Var.normalf [ output_dim; hidden_nodes ] ~stddev:0.1 in
  let b1 = Var.f [ hidden_nodes ] 0. in
  let w2 = Var.normalf [ hidden_nodes; label_count ] ~stddev:0.1 in
  let b2 = Var.f [ label_count ] 0. in
  O.(relu (layer *^ w1 + b1) *^ w2 + b2) |> O.softmax

let () =
  let mnist = Mnist_helper.read_files () in
  let xs = O.placeholder [] ~type_:Float in
  let ys = O.placeholder [] ~type_:Float in
  let testing = O.placeholder [] ~type_:Bool in
  let ys_node = O.Placeholder.to_node ys in
  let ys_ =
    build_model ~xs:(O.Placeholder.to_node xs) ~testing:(O.Placeholder.to_node testing)
  in
  let cross_entropy = O.cross_entropy ~ys:ys_node ~y_hats:ys_ `mean in
  let accuracy =
    O.(equal (argMax ys_ one32) (argMax ys_node one32))
    |> O.cast ~type_:Float
    |> O.reduce_mean
  in
  let learning_rate = O.placeholder [] ~type_:Float in
  let gd =
    Optimizers.adam_minimizer cross_entropy
      ~learning_rate:(O.Placeholder.to_node learning_rate)
  in
  let true_tensor = Tensor.create Bigarray.int8_unsigned [||] in
  Tensor.set true_tensor [||] 1;
  let false_tensor = Tensor.create Bigarray.int8_unsigned [||] in
  Tensor.set false_tensor [||] 0;
  let validation_inputs =
    let validation_images = Tensor.sub_left mnist.test_images 0 1024 in
    let validation_labels = Tensor.sub_left mnist.test_labels 0 1024 in
    Session.Input.
      [ float xs validation_images
      ; float ys validation_labels
      ; bool testing true_tensor
      ]
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
  let learning_rate_tensor = Tensor.create Bigarray.float32 [||] in
  Tensor.set learning_rate_tensor [||] 1e-3;
  for batch_idx = 1 to epochs do
    if batch_idx = 2_000
    then Tensor.set learning_rate_tensor [||] 5e-4;
    let batch_images, batch_labels =
      Mnist_helper.train_batch mnist ~batch_size ~batch_idx
    in
    let batch_inputs =
      Session.Input.
        [ float xs batch_images
        ; float ys batch_labels
        ; bool testing false_tensor
        ; float learning_rate learning_rate_tensor
        ]
    in
    if batch_idx % 25 = 0 then print_err batch_idx ~train_inputs:batch_inputs;
    Session.run
      ~inputs:batch_inputs
      ~targets:gd
      Session.Output.empty;
  done
