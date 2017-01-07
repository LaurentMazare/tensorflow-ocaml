open Core_kernel.Std
open Tensorflow

let img_size = 224

let vgg19 () =
  let block iter ~out_channels x =
    List.init iter ~f:Fn.id
    |> List.fold ~init:x ~f:(fun acc _idx ->
      Fnn.conv2d () acc
        ~w_init:(`normal 0.1) ~filter:(3, 3) ~strides:(1, 1) ~padding:`same ~out_channels
      |> Fnn.relu)
    |> Fnn.max_pool ~filter:(2, 2) ~strides:(2, 2) ~padding:`same
  in
  let input, input_id = Fnn.input ~shape:(D1 (img_size*img_size*3)) in
  let model =
    Fnn.reshape input ~shape:(D3 (img_size, img_size, 3))
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

let load_image ~filename =
  let image_handle = Magick.read_image ~filename in
  Magick.Imper.resize image_handle ~width:img_size ~height:img_size ~blur:0.1 ~filter:Cubic;
  let raw = Magick.Imper.get_raw_without_alpha image_handle in
  let tensor = Tensor.create3 Float32 img_size img_size 3 in
  for i = 0 to img_size - 1 do
    for j = 0 to img_size - 1 do
      let red, green, blue = raw.(i).(j) in
      Tensor.set tensor [| i; j; 0 |] (float blue -. 103.939);
      Tensor.set tensor [| i; j; 1 |] (float green -. 116.779);
      Tensor.set tensor [| i; j; 2 |] (float red -. 123.68);
    done;
  done;
  tensor

let () =
  let input_tensor = load_image ~filename:"input.png" in
  let input_id, model = vgg19 () in
  let results = Fnn.Model.predict model [ input_id, input_tensor ] in
  for i = 0 to 999 do
    printf "%d -> %f\n" i (Tensor.get results [| 0; i |] )
  done

