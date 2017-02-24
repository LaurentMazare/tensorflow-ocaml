open Core_kernel.Std
open Tensorflow

let img_size = 224

type nn =
  { input_id : Fnn.Input_id.t
  ; model : (Fnn._1d, [ `float ], Tensor.float32_elt) Fnn.Model.t
  ; style_layer_ids : Fnn.Id.t list
  }

let vgg19 () =
  let style_layer_ids = ref [] in
  let block iter ~block_idx ~out_channels x =
    List.init iter ~f:Fn.id
    |> List.fold ~init:x ~f:(fun acc idx ->
      let conv2d =
        Fnn.conv2d () acc
          ~name:(sprintf "conv%d_%d" block_idx (idx+1))
          ~w_init:(`normal 0.1) ~filter:(3, 3) ~strides:(1, 1) ~padding:`same ~out_channels
      in
      let relu = Fnn.relu conv2d in
      style_layer_ids := Fnn.id relu :: !style_layer_ids;
      relu)
    |> Fnn.max_pool ~filter:(2, 2) ~strides:(2, 2) ~padding:`same
  in
  let input, input_id = Fnn.input ~shape:(D3 (img_size, img_size, 3)) in
  let var = Fnn.var input in
  let model =
    Fnn.reshape var ~shape:(D3 (img_size, img_size, 3))
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
  { input_id
  ; model
  ; style_layer_ids = !style_layer_ids
  }

let imagenet_mean = function
  | `blue -> 103.939
  | `green -> 116.779
  | `red -> 123.68

let normalize x ~channel =
  float x -. imagenet_mean channel

let unnormalize x ~channel =
  min 255 (int_of_float (x +. imagenet_mean channel))

let load_image ~filename =
  let image = OImages.load filename [] in
  let image = OImages.rgb24 image in
  let width = image # width in
  let height = image # height in
  let min_edge = min width height in
  let image =
    image # sub
      ((width-min_edge) / 2)
      ((height-min_edge) / 2)
      min_edge
      min_edge
  in
  let image = image # resize None img_size img_size in
  let tensor = Tensor.create3 Float32 img_size img_size 3 in
  for i = 0 to img_size - 1 do
    for j = 0 to img_size - 1 do
      let { Color.r; g; b } = image # get j i in
      Tensor.set tensor [| i; j; 0 |] (normalize b ~channel:`blue);
      Tensor.set tensor [| i; j; 1 |] (normalize g ~channel:`green);
      Tensor.set tensor [| i; j; 2 |] (normalize r ~channel:`red);
    done;
  done;
  tensor

let save_image tensor ~filename =
  let total_size = Array.fold (Tensor.dims tensor) ~init:1 ~f:( * ) in
  if total_size <> img_size * img_size * 3
  then failwith "Incorrect tensor size";
  let image = new OImages.rgb24 img_size img_size in
  for i = 0 to img_size - 1 do
    for j = 0 to img_size - 1 do
      let b = Tensor.get tensor [| i; j; 0 |] |> unnormalize ~channel:`blue in
      let g = Tensor.get tensor [| i; j; 1 |] |> unnormalize ~channel:`green in
      let r = Tensor.get tensor [| i; j; 2 |] |> unnormalize ~channel:`red in
      image # set j i { Color.r; g; b }
    done;
  done;
  image # save filename None []

let () =
  let input_tensor = load_image ~filename:"input.jpg" in
  save_image input_tensor ~filename:"cropped.jpg";
  let { input_id; model; style_layer_ids = _ } = vgg19 () in
  Fnn.Model.load model ~filename:(Sys.getcwd () ^ "/vgg19.cpkt")
    ~inputs:[ input_id, input_tensor ];
  let results = Fnn.Model.predict model [ input_id, input_tensor ] in
  let pr, category =
    List.init 1000 ~f:(fun i ->
      Tensor.get results [| 0; i |], i+1)
    |> List.reduce_exn ~f:Pervasives.max
  in
  printf "%d: %.2f%%\n" category (100. *. pr)


