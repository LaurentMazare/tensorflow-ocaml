open Core_kernel.Std
open Tensorflow

let img_size = 224

let vgg19 () =
  let block iter ~block_idx ~out_channels x =
    List.init iter ~f:Fn.id
    |> List.fold ~init:x ~f:(fun acc idx ->
      Fnn.conv2d () acc
        ~name:(sprintf "conv%d_%d" block_idx (idx+1))
        ~w_init:(`normal 0.1) ~filter:(3, 3) ~strides:(1, 1) ~padding:`same ~out_channels
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

let load_image ~filename =
  let image_handle = Magick.read_image ~filename in
  let width = Magick.get_image_width image_handle in
  let height = Magick.get_image_height image_handle in
  let min_edge = min width height in
  let image_handle =
    Magick.Fun.crop () image_handle
      ~x:((width-min_edge) / 2)
      ~y:((height-min_edge) / 2)
      ~width:min_edge
      ~height:min_edge
    |> Magick.Fun.resize () ~width:img_size ~height:img_size ~blur:1. ~filter:Cubic
  in
  let tensor = Tensor.create3 Float32 img_size img_size 3 in
  let normalize x ~channel =
    let mean =
      match channel with
      | `blue -> 103.939
      | `green -> 116.779
      | `red -> 123.68
    in
    float x /. 65535. *. 255. -. mean
  in
  for i = 0 to img_size - 1 do
    for j = 0 to img_size - 1 do
      let red, green, blue, _alpha = Magick.Imper.acquire_pixel image_handle j i in
      Tensor.set tensor [| i; j; 0 |] (normalize blue ~channel:`blue);
      Tensor.set tensor [| i; j; 1 |] (normalize green ~channel:`green);
      Tensor.set tensor [| i; j; 2 |] (normalize red ~channel:`red);
    done;
  done;
  tensor

let () =
  let input_tensor = load_image ~filename:"input.jpg" in
  let input_id, model = vgg19 () in
  Fnn.Model.load model ~filename:(Sys.getcwd () ^ "/vgg19.cpkt");
  let results = Fnn.Model.predict model [ input_id, input_tensor ] in
  let pr, category =
    List.init 1000 ~f:(fun i ->
      Tensor.get results [| 0; i |], i+1)
    |> List.reduce_exn ~f:Pervasives.max
  in
  printf "%d: %.2f%%\n" category (100. *. pr)

