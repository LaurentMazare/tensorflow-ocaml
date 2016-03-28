open Core_kernel.Std

let train_size = 10000
let validation_size = 1000
let image_dim = Mnist.image_dim
let label_count = Mnist.label_count
let hidden_nodes = 64
let epochs = 5000

let () =
  let { Mnist.train_images; train_labels; validation_images; validation_labels } =
    Mnist.read_files ~train_size ~validation_size ()
  in
  let xs = Ops_m.placeholder [] ~type_:Float in
  let ys = Ops_m.placeholder [] ~type_:Float in
  let w1 = Var.normalf [ image_dim; hidden_nodes ] ~stddev:0.1 in
  let b1 = Var.f [ hidden_nodes ] 0. in
  let w2 = Var.normalf [ hidden_nodes; label_count ] ~stddev:0.1 in
  let b2 = Var.f [ label_count ] 0. in
  let ys_ = Ops_m.(Ops.sigmoid (xs *^ w1 + b1) *^ w2 + b2) |> Ops.softmax in
  let cross_entropy = Ops.neg Ops_m.(reduce_mean (ys * Ops.log ys_)) in
  let accuracy =
    Ops.equal (Ops.argMax ys_ Ops_m.one32) (Ops.argMax ys Ops_m.one32)
    |> Ops.cast ~type_:Float
    |> Ops_m.reduce_mean
  in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:(Ops_m.f 0.8) ~varsf:[ w1; w2; b1; b2 ]
      cross_entropy
  in
  let session = Session.create () in
  let train_inputs = Session.Input.[ float xs train_images; float ys train_labels ] in
  let validation_inputs =
    Session.Input.[ float xs validation_images; float ys validation_labels ]
  in
  let print_err n =
    let accuracy =
      Session.run
        session
        ~inputs:validation_inputs
        (Session.Output.scalar_float accuracy)
    in
    printf "epoch %d, accuracy %.2f%%\n%!" n (100. *. accuracy)
  in
  for i = 1 to epochs do
    if i % 100 = 0 then print_err i;
    Session.run
      session
      ~inputs:train_inputs
      ~targets:gd
      Session.Output.empty;
  done
