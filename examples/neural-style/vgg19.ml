open Base
open Float.O_dot
open Tensorflow_core
open Tensorflow_fnn

let img_size = 224

let vgg19 () =
  let block iter ~block_idx ~out_channels x =
    List.init iter ~f:Fn.id
    |> List.fold ~init:x ~f:(fun acc idx ->
           Fnn.conv2d
             ()
             acc
             ~name:(Printf.sprintf "conv%d_%d" block_idx (idx + 1))
             ~w_init:(`normal 0.1)
             ~filter:(3, 3)
             ~strides:(1, 1)
             ~padding:`same
             ~out_channels
           |> Fnn.relu)
    |> Fnn.max_pool ~filter:(2, 2) ~strides:(2, 2) ~padding:`same
  in
  let input, input_id = Fnn.input ~shape:(D3 (img_size, img_size, 3)) in
  let model =
    Fnn.reshape input ~shape:(D3 (img_size, img_size, 3))
    |> block 2 ~block_idx:1 ~out_channels:64
    |> block 2 ~block_idx:2 ~out_channels:128
    |> block 4 ~block_idx:3 ~out_channels:256
    |> block 4 ~block_idx:4 ~out_channels:512
    |> block 4 ~block_idx:5 ~out_channels:512
    |> Fnn.flatten
    |> Fnn.dense ~name:"fc6" ~w_init:(`normal 0.1) 4096
    |> Fnn.relu
    |> Fnn.dense ~name:"fc7" ~w_init:(`normal 0.1) 4096
    |> Fnn.relu
    |> Fnn.dense ~name:"fc8" ~w_init:(`normal 0.1) 1000
    |> Fnn.softmax
    |> Fnn.Model.create Float
  in
  input_id, model

let () =
  let { Image.tensor; _ } = Image.load "input.jpg" in
  let input_id, model = vgg19 () in
  Fnn.Model.load model ~filename:(Caml.Sys.getcwd () ^ "/vgg19.cpkt");
  let results = Fnn.Model.predict model [ input_id, tensor ] in
  let pr, category =
    List.init 1000 ~f:(fun i -> Tensor.get results [| 0; i |], i + 1)
    |> List.reduce_exn ~f:Caml.max
  in
  Stdio.printf "%d: %.2f%%\n" category (100. *. pr)
