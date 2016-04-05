open Core_kernel.Std
open Tensorflow
module O = Ops

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
    |> snd (* TODO: remove this *)
    |> Nn.dense ~shape:label_count
    |> Nn.softmax
    |> Nn.Model.create
  in
  Nn.Model.fit model
    ~loss:Cross_entropy
    ~optimizer:(Gradient_descent 0.8)
    ~epochs
    ~xs:train_images
    ~ys:train_labels;
  ignore (test_images, test_labels)
