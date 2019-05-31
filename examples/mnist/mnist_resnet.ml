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

let basic_block input_layer ~output_dim ~stride ~is_training ~update_ops_store =
  let shortcut =
    if stride = 1
    then input_layer
    else (
      let in_features = Node.shape input_layer |> List.last_exn in
      let half_diff = (output_dim - in_features) / 2 in
      O.pad
        (O.avgPool
           input_layer
           ~padding:"VALID"
           ~ksize:[ 1; stride; stride; 1 ]
           ~strides:[ 1; stride; stride; 1 ])
        (O.ci32 ~shape:[ 4; 2 ] [ 0; 0; 0; 0; 0; 0; half_diff; half_diff ]))
  in
  Layer.conv2d input_layer ~ksize:(3, 3) ~strides:(stride, stride) ~output_dim
  |> Layer.batch_norm ~is_training ~update_ops_store
  |> O.relu
  |> Layer.conv2d ~output_dim ~ksize:(1, 1) ~strides:(1, 1)
  |> O.add shortcut
  |> Layer.batch_norm ~is_training ~update_ops_store

(* No ReLU after the add as per http://torch.ch/blog/2016/02/04/resnets.html *)

let block_stack layer ~output_dim ~stride ~is_training ~update_ops_store =
  let depth = (depth - 2) / 6 in
  let layer = basic_block layer ~output_dim ~stride ~is_training ~update_ops_store in
  List.init (depth - 1) ~f:Fn.id
  |> List.fold ~init:layer ~f:(fun layer _idx ->
         basic_block layer ~update_ops_store ~output_dim ~stride:1 ~is_training)

let build_model ~xs ~is_training ~update_ops_store =
  O.reshape xs (O.ci32 [ -1; 28; 28; 1 ])
  (* 3x3 convolution + max-pool. *)
  |> Layer.conv2d ~output_dim:16 ~ksize:(3, 3) ~strides:(1, 1)
  |> O.relu
  |> block_stack ~output_dim:16 ~stride:1 ~is_training ~update_ops_store
  |> block_stack ~output_dim:32 ~stride:2 ~is_training ~update_ops_store
  |> block_stack ~output_dim:64 ~stride:2 ~is_training ~update_ops_store
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
    Optimizers.adam_minimizer cross_entropy ~learning_rate:(O.f 1e-4)
    @ Layer.Update_ops_store.ops update_ops_store
  in
  let true_tensor = Tensor.create Bigarray.int8_unsigned [||] in
  Tensor.set true_tensor [||] 1;
  let false_tensor = Tensor.create Bigarray.int8_unsigned [||] in
  Tensor.set false_tensor [||] 0;
  let predict images =
    Session.run
      (Session.Output.float ys_)
      ~inputs:Session.Input.[ float xs images; bool is_training false_tensor ]
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
  Checkpointing.loop
    ~start_index:1
    ~end_index:epochs
    ~save_vars_from:gd
    ~checkpoint_base:"tf-resnet"
    (fun ~index:batch_idx ->
      let batch_images, batch_labels =
        Mnist_helper.train_batch mnist ~batch_size ~batch_idx
      in
      if batch_idx % 100 = 0 then print_err batch_idx;
      Session.run
        ~inputs:
          Session.Input.
            [ float xs batch_images; float ys batch_labels; bool is_training true_tensor ]
        ~targets:gd
        Session.Output.empty)
