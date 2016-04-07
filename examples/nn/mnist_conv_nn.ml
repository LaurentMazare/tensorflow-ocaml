open Core_kernel.Std
open Tensorflow

let label_count = Mnist_helper.label_count
let epochs = 300

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let model =
    Nn.input ~shape:(D1 (28*28))
    |> Nn.reshape ~shape:(D3 (28, 28, 1))
    |> Nn.conv2d ~filter:(5, 5) ~out_channels:32 ~strides:(1, 1) ~padding:`same
    |> Nn.max_pool ~ksize:(1, 2, 2, 1) ~strides:(1, 2, 2, 1) ~padding:`same
    |> Nn.conv2d ~filter:(5, 5) ~out_channels:64 ~strides:(1, 1) ~padding:`same
    |> Nn.max_pool ~ksize:(1, 2, 2, 1) ~strides:(1, 2, 2, 1) ~padding:`same
    |> Nn.flatten
    |> Nn.dense ~shape:1024
    |> Nn.relu
    |> Nn.dense ~shape:label_count
    |> Nn.softmax
    |> Model.create
  in
  Model.fit model
    ~loss:(Model.Loss.cross_entropy `sum)
    ~optimizer:(Model.Optimizer.adam ~alpha:1e-5 ())
    ~epochs
    ~xs:train_images
    ~ys:train_labels;
  let test_results = Model.evaluate model test_images in
  printf "Accuracy: %.2f%%\n%!" (100. *. Mnist_helper.accuracy test_results test_labels)
