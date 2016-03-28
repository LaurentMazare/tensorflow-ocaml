open Core_kernel.Std
module O = Ops

let scalar_tensor f =
  let array = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 1 |] in
  Bigarray.Genarray.set array [| 0 |] f;
  array

let batch_size = 512
let epochs = 5000

let conv2d x w =
  O.conv2D x w ~strides:[ 1; 1; 1; 1 ] ~padding:"SAME"

let max_pool_2x2 x =
  O.maxPool x ~ksize:[ 1; 2; 2; 1 ] ~strides:[ 1; 2; 2; 1 ] ~padding:"SAME"

let () =
  let { Mnist.train_images; train_labels; test_images; test_labels } =
    Mnist.read_files ()
  in
  let train_size = (Bigarray.Genarray.dims train_images).(0) in
  let keep_prob = O.placeholder [] ~type_:Float in
  let xs = O.placeholder [] ~type_:Float in
  let ys = O.placeholder [] ~type_:Float in

  let x_image = O.reshape xs (O.const_int ~type_:Int32 [ -1; 28; 28; 1 ]) in
  let w_conv1 = Var.normalf [ 5; 5; 1; 32 ] ~stddev:0.1 in
  let b_conv1 = Var.f [ 32 ] 0. in
  let h_conv1 = O.add (conv2d x_image w_conv1) b_conv1 in
  let h_pool1 = max_pool_2x2 h_conv1 in

  let w_conv2 = Var.normalf [ 5; 5; 32; 64 ] ~stddev:0.1 in
  let b_conv2 = Var.f [ 64 ] 0. in
  let h_conv2 = O.add (conv2d h_pool1 w_conv2) b_conv2 |> O.relu in
  let h_pool2 = max_pool_2x2 h_conv2 in

  let w_fc1 = Var.normalf [ 7*7*64; 1024 ] ~stddev:0.1 in
  let b_fc1 = Var.f [ 1024 ] 0. in
  let h_pool2_flat = O.reshape h_pool2 (O.const_int ~type_:Int32 [ -1; 7*7*64 ]) in
  let h_fc1 = O.(relu (h_pool2_flat *^ w_fc1 + b_fc1)) in
  let h_fc1_dropout = O.dropout h_fc1 ~keep_prob in

  let w_fc2 = Var.normalf [ 1024; 10 ] ~stddev:0.1 in
  let b_fc2 = Var.f [ 10 ] 0. in

  let ys_ = O.(h_fc1_dropout *^ w_fc2 + b_fc2) |> O.softmax in
  let cross_entropy = O.(neg (reduce_mean (ys * O.log ys_))) in
  let accuracy =
    O.(equal (argMax ys_ one32) (argMax ys one32))
    |> O.cast ~type_:Float
    |> O.reduce_mean
  in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:(O.f 0.02)
      ~varsf:[ w_conv1; b_conv1; w_conv2; b_conv2; w_fc1; b_fc1; w_fc2; b_fc2 ]
      cross_entropy
  in
  let validation_inputs =
    let one = scalar_tensor 1. in
    Session.Input.[ float xs test_images; float ys test_labels; float keep_prob one ]
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
    printf "epoch %d, vaccuracy %.2f%% taccuracy: %.2f%%\n%!" n (100. *. vaccuracy) (100. *. taccuracy)
  in
  for batch_idx = 1 to epochs do
    let start_batch = batch_size * batch_idx in
    (* Stochastic gradient descent... *)
    let start_batch = if start_batch + batch_size >= train_size then 0 else start_batch in
    let batch_images =
      Mnist.slice2 (Bigarray.array2_of_genarray train_images) start_batch batch_size
      |> Bigarray.genarray_of_array2
    in
    let batch_labels =
      Mnist.slice2 (Bigarray.array2_of_genarray train_labels) start_batch batch_size
      |> Bigarray.genarray_of_array2
    in
    let batch_inputs =
      let half = scalar_tensor 0.5 in
      Session.Input.[ float xs batch_images; float ys batch_labels; float keep_prob half ]
    in
    print_err batch_idx ~train_inputs:batch_inputs;
    Session.run
      ~inputs:batch_inputs
      ~targets:gd
      Session.Output.empty;
  done
