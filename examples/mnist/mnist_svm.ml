open Core_kernel.Std

let train_size = 1000
let validation_size = 1000
let image_dim = Mnist.image_dim
let label_count = Mnist.label_count
let epochs = 10000
let lambda = 0.001

let () =
  let { Mnist.train_images; train_labels; validation_images; validation_labels } =
    Mnist.read_files ~train_size ~validation_size ()
  in
  let xs = Ops_m.placeholder [] ~type_:Float in
  let ys = Ops_m.placeholder [] ~type_:Float in
  let w = Var.f [ image_dim; label_count ] 0. in
  let b = Var.f [ label_count ] 0. in
  let ys_ = Ops_m.(xs *^ w - b) in
  let accuracy =
    Ops.equal (Ops.argMax ys_ Ops_m.one32) (Ops.argMax ys Ops_m.one32)
    |> Ops.cast ~type_:Float
    |> Ops_m.reduce_mean
  in

  (* ys in {-1 ; + 1} *)
  let ys' = Ops_m.(f 2.0 * ys - f 1.0) in
  let distance_from_margin =
    Ops_m.(f 1.0 - ys' * ys_ )
  in
  let hinge_loss =
     Ops.relu distance_from_margin |> Ops_m.reduce_mean
  in
  let square_norm =
    Ops_m.(f lambda * reduce_sum (w * w))
  in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:(Ops_m.f 0.02) ~varsf:[ w; b ]
      Ops_m.(square_norm + hinge_loss)
  in
  let train_inputs = Session.Input.[ float xs train_images; float ys train_labels ] in
  let validation_inputs =
    Session.Input.[ float xs validation_images; float ys validation_labels ]
  in
  let print_err n =
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
    printf "epoch %d, accuracy %.2f%% train accuracy %.2f%%\n%!"
      n
      (100. *. vaccuracy)
      (100. *. taccuracy)
  in
  for i = 1 to epochs do
    if i % 100 = 0 then print_err i;
    Session.run
      ~inputs:train_inputs
      ~targets:gd
      Session.Output.empty;
  done
