open Base
open Float.O_dot
open Tensorflow
module O = Ops

(* This should reach ~92% accuracy. *)
let image_dim = Mnist_helper.image_dim
let label_count = Mnist_helper.label_count
let epochs = 300

let () =
  let mnist = Mnist_helper.read_files () in
  let xs = O.placeholder [ -1; image_dim ] ~type_:Float in
  let ys = O.placeholder [ -1; label_count ] ~type_:Float in
  let ys_ =
    Layer.linear (O.Placeholder.to_node xs) ~activation:Softmax ~output_dim:label_count
  in
  let cross_entropy = O.cross_entropy ~ys:(O.Placeholder.to_node ys) ~y_hats:ys_ `mean in
  let gd = Optimizers.gradient_descent_minimizer ~learning_rate:(O.f 8.) cross_entropy in
  let print_err n =
    let accuracy =
      Mnist_helper.batch_accuracy mnist `test ~batch_size:1024 ~predict:(fun images ->
          Session.run
            (Session.Output.float ys_)
            ~inputs:Session.Input.[ float xs images ])
    in
    Stdio.printf "epoch %d, accuracy %.2f%%\n%!" n (100. *. accuracy)
  in
  for i = 1 to epochs do
    if i % 50 = 0 then print_err i;
    Session.run
      ~inputs:Session.Input.[ float xs mnist.train_images; float ys mnist.train_labels ]
      ~targets:gd
      Session.Output.empty
  done
