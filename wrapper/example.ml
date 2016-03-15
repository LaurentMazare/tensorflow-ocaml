open Tensor
open Ctypes

let read_file filename =
  let lines = ref [] in
  let chan = open_in filename in
  try
    while true; do
      lines := input_line chan :: !lines
    done;
    assert false
  with End_of_file ->
    close_in chan;
    String.concat "\n" (List.rev !lines)

let char_list_of_string s =
  let list = ref [] in
  for i = 0 to String.length s - 1 do
    list := s.[i] :: !list
  done;
  List.rev !list

let () =
  let vector = Tensor.create1d 10 in
  Printf.printf ">> %d %d %d\n%!"
    (tf_numdims vector) (tf_dim vector 0) (tf_tensorbytesize vector |> Unsigned.Size_t.to_int);
  let session_options = Session_options.create () in
  let status = Status.create () in
  tf_setstatus status 9 "test-message";
  Printf.printf "%d %s\n%!" (tf_getcode status) (tf_message status);
  let session = Session.create session_options status in
  Printf.printf "%d %s\n%!" (tf_getcode status) (tf_message status);
  let simple_pbtxt = read_file "test.pbtxt" in
  let carray = char_list_of_string simple_pbtxt |> Ctypes.CArray.of_list char in
  tf_extendgraph
    session
    (Ctypes.CArray.start carray |> to_voidp)
    (String.length simple_pbtxt |> Unsigned.Size_t.of_int)
    status;
  Printf.printf "%d %s\n%!" (tf_getcode status) (tf_message status);
  let output_tensors = Ctypes.CArray.make tf_tensor 1 in
  tf_run
    session
    Ctypes.CArray.(of_list string [] |> start)
    Ctypes.CArray.(of_list tf_tensor [] |> start)
    0
    Ctypes.CArray.(of_list string [ "add" ] |> start)
    (Ctypes.CArray.start output_tensors)
    1
    Ctypes.CArray.(of_list string [ "add" ] |> start)
    1
    status;
  Printf.printf "%d %s\n%!" (tf_getcode status) (tf_message status);
  let output_tensor = Ctypes.CArray.get output_tensors 0 in
  let data =
    Ctypes.CArray.from_ptr (tf_tensordata output_tensor |> from_voidp float) 1
  in
  Printf.printf "%f\n%!" (Ctypes.CArray.get data 0)

