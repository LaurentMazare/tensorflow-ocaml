(* The readers implemented here are very inefficient as they read bytes one at a time. *)
open Core_kernel.Std

let image_dim = 28 * 28
let label_count = 10

let slice2 data start_idx n =
  let data = Bigarray.array2_of_genarray data in
  let dim2 = Bigarray.Array2.dim2 data in
  let slice =
    Bigarray.Array2.create (Bigarray.Array2.kind data) Bigarray.c_layout n dim2
  in
  for i = 0 to n - 1 do
    for j = 0 to dim2 - 1 do
      Bigarray.Array2.set slice i j (Bigarray.Array2.get data (start_idx + i) j)
    done;
  done;
  Bigarray.genarray_of_array2 slice

let one_hot labels =
  let nsamples = Bigarray.Array1.dim labels in
  let one_hot =
    Bigarray.Genarray.create
      Bigarray.float32
      Bigarray.c_layout
      [| nsamples; label_count |]
  in
  for idx = 0 to nsamples - 1 do
    for lbl = 0 to 9 do
      Bigarray.Genarray.set one_hot [| idx; lbl |] 0.
    done;
    let lbl = Bigarray.Array1.get labels idx |> Int32.to_int_exn in
    Bigarray.Genarray.set one_hot [| idx; lbl |] 1.
  done;
  one_hot

let read_int32_be in_channel =
  let b1 = Option.value_exn (In_channel.input_byte in_channel) in
  let b2 = Option.value_exn (In_channel.input_byte in_channel) in
  let b3 = Option.value_exn (In_channel.input_byte in_channel) in
  let b4 = Option.value_exn (In_channel.input_byte in_channel) in
  b4 + 256 * (b3 + 256 * (b2 + 256 * b1))

let read_images filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2051
  then failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let rows = read_int32_be in_channel in
  let columns = read_int32_be in_channel in
  let data =
    Bigarray.Array2.create Bigarray.float32 Bigarray.c_layout samples (rows * columns)
  in
  for sample = 0 to samples - 1 do
    for idx = 0 to rows * columns - 1 do
      let v = Option.value_exn (In_channel.input_byte in_channel) in
      Bigarray.Array2.set data sample idx (float v /. 255.);
    done;
  done;
  In_channel.close in_channel;
  data

let read_labels filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2049
  then failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let data = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout samples in
  for sample = 0 to samples - 1 do
    let v = Option.value_exn (In_channel.input_byte in_channel) |> Int32.of_int_exn in
    Bigarray.Array1.set data sample v;
  done;
  In_channel.close in_channel;
  data

type float32_genarray =
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Genarray.t

type t =
  { train_images : float32_genarray
  ; train_labels : float32_genarray
  ; test_images : float32_genarray
  ; test_labels : float32_genarray
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
  { train_images = Bigarray.genarray_of_array2 train_images
  ; train_labels = one_hot train_labels
  ; test_images = Bigarray.genarray_of_array2 test_images
  ; test_labels = one_hot test_labels
  }

let train_batch { train_images; train_labels; _ } ~batch_size ~batch_idx =
  let train_size = (Bigarray.Genarray.dims train_images).(0) in
  let start_batch = (batch_size * batch_idx) mod (train_size - batch_size) in
  let batch_images = slice2 train_images start_batch batch_size in
  let batch_labels = slice2 train_labels start_batch batch_size in
  batch_images, batch_labels

