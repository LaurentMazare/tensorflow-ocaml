open Base
open Float.O_dot
open Tensorflow
open Tensorflow_fnn

(* This should reach ~92% accuracy. *)
let image_dim = Mnist_helper.image_dim
let label_count = Mnist_helper.label_count
let epochs = 300

let () =
  let mnist = Mnist_helper.read_files () in
  let input, input_id = Fnn.input ~shape:(D1 image_dim) in
  let model = Fnn.dense label_count input |> Fnn.softmax |> Fnn.Model.create Float in
  Fnn.Model.fit
    model
    ~loss:(Fnn.Loss.cross_entropy `mean)
    ~optimizer:(Fnn.Optimizer.gradient_descent ~learning_rate:8.)
    ~epochs
    ~input_id
    ~xs:mnist.train_images
    ~ys:mnist.train_labels;
  let test_accuracy =
    Mnist_helper.batch_accuracy mnist `test ~batch_size:1024 ~predict:(fun images ->
        Fnn.Model.predict model [ input_id, images ])
  in
  Stdio.printf "Accuracy: %.2f%%\n%!" (100. *. test_accuracy)
