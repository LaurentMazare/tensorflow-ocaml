open Core_kernel.Std
open Tensorflow

let label_count = Mnist_helper.label_count
let epochs = 2000
let batch_size = 256

let conv2d =
  Nn.conv2d ~w_init:(`normal 0.1) ~filter:(5, 5) ~strides:(1, 1) ~padding:`same
let max_pool = Nn.max_pool ~filter:(2, 2) ~strides:(2, 2) ~padding:`same

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let model =
    Nn.input ~shape:(D1 (28*28))
    |> Nn.reshape ~shape:(D3 (28, 28, 1))
    |> conv2d ~out_channels:32
    |> max_pool
    |> conv2d ~out_channels:64
    |> max_pool
    |> Nn.flatten
    |> Nn.dense ~w_init:(`normal 0.1) ~shape:1024
    |> Nn.relu
    |> Nn.dense ~w_init:(`normal 0.1) ~shape:label_count
    |> Nn.softmax
    |> Model.create
  in
  let on_epoch epoch ~err ~loss:_ =
    if epoch % 50 = 0
    then begin
      let test_results = Model.evaluate ~batch_size model test_images in
      printf "Epoch: %6d/%-6d   Training Loss: %8.2f  Valid Acc: %8.2f %%\n%!"
        epoch epochs err (100. *. Mnist_helper.accuracy test_results test_labels)
    end;
    `do_nothing
  in
  Model.fit model
    ~loss:(Model.Loss.cross_entropy `sum)
    ~optimizer:(Model.Optimizer.adam ~alpha:1e-4 ())
    ~epochs
    ~on_epoch
    ~batch_size
    ~xs:train_images
    ~ys:train_labels;
  let test_results = Model.evaluate ~batch_size model test_images in
  printf "Accuracy: %.2f%%\n%!" (100. *. Mnist_helper.accuracy test_results test_labels)
