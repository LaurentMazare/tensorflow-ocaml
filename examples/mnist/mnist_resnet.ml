open Base
open Tensorflow
module O = Ops
module Tensor = Tensorflow_core.Tensor

(* ResNet model for the mnist dataset.

   Reference:
     - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
*)
let image_dim = Mnist_helper.image_dim
let label_count = Mnist_helper.label_count
let epochs = 100_000
let batch_size = 128

let depth = 20

let conv2d x ~out_features ~in_features ~stride:s ~kernel_size:k =
  let w = Var.normalf [ k; k; in_features; out_features ] ~stddev:0.1 in
  let b = Var.f [ out_features ] 0. in
  let conv = O.conv2D x w ~strides:[ 1; s; s; 1 ] ~padding:"SAME" in
  O.add conv b

let avg_pool x ~stride:s =
  O.avgPool x ~ksize:[ 1; s; s; 1 ] ~strides:[ 1; s; s; 1 ] ~padding:"VALID"

let basic_block input_layer ~out_features ~in_features ~stride ~is_training ~update_ops_store =
  let shortcut =
    if stride = 1
    then input_layer
    else
      let half_diff = (out_features - in_features) / 2 in
      O.pad (avg_pool input_layer ~stride)
        (O.ci32 ~shape:[ 4; 2 ] [ 0; 0; 0; 0; 0; 0; half_diff; half_diff ])
  in
  conv2d input_layer ~out_features ~in_features ~stride ~kernel_size:3
  |> Layer.batch_norm ~is_training ~update_ops_store
  |> O.relu
  |> conv2d ~out_features ~in_features:out_features ~stride:1 ~kernel_size:1
  |> O.add shortcut
  |> Layer.batch_norm ~is_training ~update_ops_store
  (* No ReLU after the add as per http://torch.ch/blog/2016/02/04/resnets.html *)

let block_stack layer ~out_features ~in_features ~stride ~is_training ~update_ops_store =
  let depth = (depth - 2) / 6 in
  let layer =
    basic_block layer ~out_features ~in_features ~stride ~is_training ~update_ops_store
  in
  List.init (depth - 1) ~f:Fn.id
  |> List.fold ~init:layer ~f:(fun layer _idx ->
    basic_block layer
      ~update_ops_store ~out_features ~in_features:out_features ~stride:1 ~is_training)

let build_model ~xs ~is_training ~update_ops_store =
  O.reshape xs (O.ci32 [ -1; 28; 28; 1 ])
  (* 3x3 convolution + max-pool. *)
  |> conv2d ~out_features:16 ~in_features:1 ~kernel_size:3 ~stride:1
  |> O.relu

  |> block_stack ~out_features:16 ~in_features:16 ~stride:1 ~is_training ~update_ops_store
  |> block_stack ~out_features:32 ~in_features:16 ~stride:2 ~is_training ~update_ops_store
  |> block_stack ~out_features:64 ~in_features:32 ~stride:2 ~is_training ~update_ops_store

  |> O.reduce_mean ~dims:[ 1; 2 ]
  (* Final dense layer. *)
  |> Layer.linear ~output_dim:64 ~activation:Relu
  |> Layer.linear ~output_dim:label_count ~activation:Softmax

let () =
  let mnist = Mnist_helper.read_files () in
  let xs = O.placeholder [ -1; image_dim ] ~type_:Float in
  let ys = O.placeholder [ -1; label_count ] ~type_:Float in
  let is_training = O.placeholder [] ~type_:Bool in
  let ys_node = O.Placeholder.to_node ys in
  let update_ops_store = Layer.Update_ops_store.create () in
  let ys_ =
    build_model
      ~xs:(O.Placeholder.to_node xs)
      ~is_training:(O.Placeholder.to_node is_training)
      ~update_ops_store
  in
  let cross_entropy = O.cross_entropy ~ys:ys_node ~y_hats:ys_ `mean in
  let gd =
    Optimizers.adam_minimizer cross_entropy
      ~learning_rate:(O.f 1e-4)
    @ Layer.Update_ops_store.ops update_ops_store
  in
  let true_tensor = Tensor.create Bigarray.int8_unsigned [||] in
  Tensor.set true_tensor [||] 1;
  let false_tensor = Tensor.create Bigarray.int8_unsigned [||] in
  Tensor.set false_tensor [||] 0;
  let predict images =
    Session.run (Session.Output.float ys_)
      ~inputs:Session.Input.[ float xs images; bool is_training false_tensor ]
  in
  let print_err n =
    let test_accuracy =
      Mnist_helper.batch_accuracy mnist `test ~batch_size:1024 ~predict
    in
    let train_accuracy =
      Mnist_helper.batch_accuracy mnist `train ~batch_size:1024 ~predict ~samples:5000
    in
    Stdio.printf "epoch %d, train: %.2f%% valid: %.2f%%\n%!"
      n (100. *. train_accuracy) (100. *. test_accuracy)
  in
  for batch_idx = 1 to epochs do
    let batch_images, batch_labels =
      Mnist_helper.train_batch mnist ~batch_size ~batch_idx
    in
    let batch_inputs =
      Session.Input.
        [ float xs batch_images
        ; float ys batch_labels
        ; bool is_training true_tensor
        ]
    in
    if batch_idx % 100 = 0 then print_err batch_idx;
    Session.run
      ~inputs:batch_inputs
      ~targets:gd
      Session.Output.empty;
  done
