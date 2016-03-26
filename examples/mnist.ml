(* The readers implemented here are very inefficient as they read bytes one at a time. *)
open Core_kernel.Std

let read_int32_be in_channel =
  let b1 = Option.value_exn (In_channel.input_byte in_channel) in
  let b2 = Option.value_exn (In_channel.input_byte in_channel) in
  let b3 = Option.value_exn (In_channel.input_byte in_channel) in
  let b4 = Option.value_exn (In_channel.input_byte in_channel) in
  b4 + 256 * (b3 + 256 * (b2 + 256 * b1))

let read_images ?nsamples filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2051
  then failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let samples = min samples (Option.value nsamples ~default:samples) in
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

let read_labels ?nsamples filename =
  let in_channel = In_channel.create filename in
  let magic_number = read_int32_be in_channel in
  if magic_number <> 2049
  then failwithf "Incorrect magic number in %s: %d" filename magic_number ();
  let samples = read_int32_be in_channel in
  let samples = min samples (Option.value nsamples ~default:samples) in
  let data = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout samples in
  for sample = 0 to samples - 1 do
    let v = Option.value_exn (In_channel.input_byte in_channel) |> Int32.of_int_exn in
    Bigarray.Array1.set data sample v;
  done;
  In_channel.close in_channel;
  data
