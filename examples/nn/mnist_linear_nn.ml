open Core_kernel.Std
open Tensorflow

(* This should reach ~92% accuracy. *)
let image_dim = Mnist_helper.image_dim
let label_count = Mnist_helper.label_count
let epochs = 300

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let model =
    Nn.input ~shape:(D1 image_dim)
    |> Nn.dense ~shape:label_count
    |> Nn.softmax
    |> Model.create
  in
  Model.fit model
    ~loss:(Model.Loss.cross_entropy `mean)
    ~optimizer:(Model.Optimizer.gradient_descent ~learning_rate:8.)
    ~epochs
    ~xs:train_images
    ~ys:train_labels;
  let test_results = Model.evaluate model test_images in
  printf "Accuracy: %.2f%%\n%!" (100. *. Mnist_helper.accuracy test_results test_labels)
