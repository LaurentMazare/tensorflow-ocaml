(* The readers implemented here are very inefficient as they read bytes one at a time. *)
open Base
open Tensorflow_core
module In_channel = Stdio.In_channel

let image_dim = 28 * 28
let label_count = 10

let one_hot labels =
  let nsamples = Bigarray.Array1.dim labels in
  let one_hot = Tensor.create2 Float32 nsamples label_count in
  for idx = 0 to nsamples - 1 do
    for lbl = 0 to 9 do
      Tensor.set one_hot [| idx; lbl |] 0.
    done;
    let lbl = labels.{idx} |> Int32.to_int_exn in
    Tensor.set one_hot [| idx; lbl |] 1.
  done;
  one_hot

let read_int32_be in_channel =
  let b1 = Option.value_exn (In_channel.input_byte in_channel) in
  let b2 = Option.value_exn (In_channel.input_byte in_channel) in
  let b3 = Option.value_exn (In_channel.input_byte in_channel) in
  let b4 = Option.value_exn (In_channel.input_byte in_channel) in
  b4 + (256 * (b3 + (256 * (b2 + (256 * b1)))))

let read_images filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2051
  then Printf.failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let rows = read_int32_be in_channel in
  let columns = read_int32_be in_channel in
  let data =
    Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout samples (rows * columns)
  in
  for sample = 0 to samples - 1 do
    for idx = 0 to (rows * columns) - 1 do
      let v = Option.value_exn (In_channel.input_byte in_channel) in
      data.{sample, idx} <- Float.(of_int v / 255.)
    done
  done;
  In_channel.close in_channel;
  data

let read_labels filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2049
  then Printf.failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let data = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout samples in
  for sample = 0 to samples - 1 do
    let v = Option.value_exn (In_channel.input_byte in_channel) |> Int32.of_int_exn in
    data.{sample} <- v
  done;
  In_channel.close in_channel;
  data

type float32_tensor = (float, Bigarray.float32_elt) Tensor.t

type t =
  { train_images : float32_tensor
  ; train_labels : float32_tensor
  ; test_images : float32_tensor
  ; test_labels : float32_tensor
  }

let read_files
    ?(train_image_file = "data/train-images-idx3-ubyte")
    ?(train_label_file = "data/train-labels-idx1-ubyte")
    ?(test_image_file = "data/t10k-images-idx3-ubyte")
    ?(test_label_file = "data/t10k-labels-idx1-ubyte")
    ()
  =
  let train_images = read_images train_image_file in
  let train_labels = read_labels train_label_file in
  let test_images = read_images test_image_file in
  let test_labels = read_labels test_label_file in
  let train_images =
    Bigarray.genarray_of_array2 train_images |> Tensor.of_bigarray ~scalar:false
  in
  let test_images =
    Bigarray.genarray_of_array2 test_images |> Tensor.of_bigarray ~scalar:false
  in
  { train_images
  ; train_labels = one_hot train_labels
  ; test_images
  ; test_labels = one_hot test_labels
  }

let train_batch { train_images; train_labels; _ } ~batch_size ~batch_idx =
  let train_size = (Tensor.dims train_images).(0) in
  let start_batch = Int.( % ) (batch_size * batch_idx) (train_size - batch_size) in
  let batch_images = Tensor.sub_left train_images start_batch batch_size in
  let batch_labels = Tensor.sub_left train_labels start_batch batch_size in
  batch_images, batch_labels

(** [accuracy label1 label2] returns the proportion of labels that are equal between
    [label1] and [label2].
*)
let accuracy ys ys' =
  let ys = Bigarray.array2_of_genarray (Tensor.to_bigarray ys) in
  let ys' = Bigarray.array2_of_genarray (Tensor.to_bigarray ys') in
  let nsamples = Bigarray.Array2.dim1 ys in
  let res = ref 0. in
  let find_best_idx ys n =
    let best_idx = ref 0 in
    for l = 1 to label_count - 1 do
      let v = ys.{n, !best_idx} in
      let v' = ys.{n, l} in
      if Float.( > ) v' v then best_idx := l
    done;
    !best_idx
  in
  for n = 0 to nsamples - 1 do
    let idx = find_best_idx ys n in
    let idx' = find_best_idx ys' n in
    res := Float.( + ) !res (if idx = idx' then 1. else 0.)
  done;
  Float.(!res / of_int nsamples)

let batch_accuracy ?samples t train_or_test ~batch_size ~predict =
  let images, labels =
    match train_or_test with
    | `train -> t.train_images, t.train_labels
    | `test -> t.test_images, t.test_labels
  in
  let dataset_samples = (Tensor.dims labels).(0) in
  let samples =
    Option.value_map samples ~default:dataset_samples ~f:(Int.min dataset_samples)
  in
  let rec loop start_index sum_accuracy =
    if samples <= start_index
    then sum_accuracy /. Float.of_int samples
    else (
      let batch_size = Int.min batch_size (samples - start_index) in
      let images = Tensor.sub_left images start_index batch_size in
      let predicted_labels = predict images in
      let labels = Tensor.sub_left labels start_index batch_size in
      let batch_accuracy = accuracy predicted_labels labels in
      loop
        (start_index + batch_size)
        (sum_accuracy +. (batch_accuracy *. Float.of_int batch_size)))
  in
  loop 0 0.
