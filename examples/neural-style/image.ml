open Base
open Tensorflow_core

type t =
  { tensor : (float, Bigarray.float32_elt) Tensor.t
  ; width : int
  ; height : int
  }

let imagenet_mean = function
  | `blue -> 103.939
  | `green -> 116.779
  | `red -> 123.68

let normalize x ~channel = Float.of_int x -. imagenet_mean channel
let unnormalize x ~channel = min 255 (Int.of_float (x +. imagenet_mean channel)) |> max 0

let load filename =
  let image =
    match Stb_image.load ~channels:3 filename with
    | Ok image -> image
    | Error (`Msg msg) -> Printf.failwithf "unable to load %s: %s" filename msg ()
  in
  let img_w = Stb_image.width image in
  let img_h = Stb_image.height image in
  let data = Stb_image.data image in
  let tensor = Tensor.create3 Float32 img_h img_w 3 in
  for i = 0 to img_h - 1 do
    for j = 0 to img_w - 1 do
      let idx = (3 * img_w * i) + (3 * j) in
      Tensor.set tensor [| i; j; 0 |] (normalize data.{idx + 0} ~channel:`red);
      Tensor.set tensor [| i; j; 1 |] (normalize data.{idx + 1} ~channel:`green);
      Tensor.set tensor [| i; j; 2 |] (normalize data.{idx + 2} ~channel:`blue)
    done
  done;
  { tensor; width = img_w; height = img_h }

let save tensor filename =
  let img_w, img_h =
    match Tensor.dims tensor with
    | [| img_h; img_w; 3 |] -> img_w, img_h
    | _ -> failwith "Improper tensor dimensions"
  in
  let data = Bigarray.Array1.create Int8_unsigned C_layout (3 * img_w * img_h) in
  for i = 0 to img_h - 1 do
    for j = 0 to img_w - 1 do
      let idx = (3 * img_w * i) + (3 * j) in
      data.{idx + 0} <- Tensor.get tensor [| i; j; 0 |] |> unnormalize ~channel:`red;
      data.{idx + 1} <- Tensor.get tensor [| i; j; 1 |] |> unnormalize ~channel:`green;
      data.{idx + 2} <- Tensor.get tensor [| i; j; 2 |] |> unnormalize ~channel:`blue
    done
  done;
  Stb_image_write.png filename ~w:img_w ~h:img_h data ~c:3
