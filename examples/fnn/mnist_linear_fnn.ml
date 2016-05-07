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
  let input, input_name = Fnn.input ~shape:(D1 image_dim) in
  let model =
    Staged.unstage (Fnn.dense label_count) input
    |> Fnn.softmax
    |> Fnn.Model.create ~type_:Float
  in
  Fnn.Model.fit model [ input_name, test_images ] Float
    ~loss:(Fnn.Loss.cross_entropy `mean)
    ~optimizer:(Fnn.Optimizer.gradient_descent ~learning_rate:8.)
    ~epochs
    ~xs:train_images
    ~ys:train_labels;
  let test_results = Fnn.Model.predict model [ input_name, test_images ] Float in
  printf "Accuracy: %.2f%%\n%!" (100. *. Mnist_helper.accuracy test_results test_labels)

