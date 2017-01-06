open Core_kernel.Std
open Tensorflow

let label_count = Mnist_helper.label_count
let image_dim = Mnist_helper.image_dim
let epochs = 2000
let batch_size = 256

let vgg19 () =
  let block iter ~out_channels x =
    List.init iter ~f:Fn.id
    |> List.fold ~init:x ~f:(fun acc _idx ->
      Fnn.conv2d () acc
        ~w_init:(`normal 0.1) ~filter:(3, 3) ~strides:(1, 1) ~padding:`same ~out_channels
      |> Fnn.relu)
    |> Fnn.max_pool ~filter:(2, 2) ~strides:(2, 2) ~padding:`same
  in
  let input, input_id = Fnn.input ~shape:(D1 (224*224*3)) in
  let model =
    Fnn.reshape input ~shape:(D3 (224, 224, 3))
    |> block 2 ~out_channels:64
    |> block 2 ~out_channels:128
    |> block 4 ~out_channels:256
    |> block 4 ~out_channels:512
    |> block 4 ~out_channels:512
    |> Fnn.flatten
    |> Fnn.dense ~w_init:(`normal 0.1) 4096
    |> Fnn.relu
    |> Fnn.dense ~w_init:(`normal 0.1) 4096
    |> Fnn.relu
    |> Fnn.dense ~w_init:(`normal 0.1) 1000
    |> Fnn.softmax
    |> Fnn.Model.create Float
  in
  input_id, model

let () =
  let { Mnist_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let input_id, model = vgg19 () in
  Fnn.Model.fit model
    ~loss:(Fnn.Loss.cross_entropy `sum)
    ~optimizer:(Fnn.Optimizer.adam ~learning_rate:1e-4 ())
    ~epochs
    ~batch_size
    ~input_id
    ~xs:train_images
    ~ys:train_labels;
  let test_results = Fnn.Model.predict model [ input_id, test_images ] in
  printf "Accuracy: %.2f%%\n%!" (100. *. Mnist_helper.accuracy test_results test_labels)

