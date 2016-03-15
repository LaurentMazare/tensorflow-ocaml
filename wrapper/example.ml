open Wrapper
module CArray = Ctypes.CArray

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
  let vector = Tensor.create1d Ctypes.float 10 in
  Printf.printf ">> %d %d %d\n%!"
    (Tensor.num_dims vector) (Tensor.dim vector 0) (Tensor.byte_size vector);
  let session_options = Session_options.create () in
  let status = Status.create () in
  Status.set status 9 "test-message";
  Printf.printf "%d %s\n%!" (Status.code status) (Status.message status);
  let session = Session.create session_options status in
  Printf.printf "%d %s\n%!" (Status.code status) (Status.message status);
  let simple_pbtxt = read_file "test.pbtxt" in
  let carray = char_list_of_string simple_pbtxt |> CArray.of_list Ctypes.char in
  Session.extend_graph
    session
    carray
    (String.length simple_pbtxt)
    status;
  Printf.printf "%d %s\n%!" (Status.code status) (Status.message status);
  let output_tensors =
    Session.run
      session
      ~inputs:[]
      ~outputs:[ "add" ]
      ~targets:[ "add" ]
      status
  in
  Printf.printf "%d %s\n%!" (Status.code status) (Status.message status);
  match output_tensors with
  | [ output_tensor ] ->
    let data = Tensor.data output_tensor Ctypes.float 1 in
    Printf.printf "%f\n%!" (CArray.get data 0)
  | [] | _ :: _ :: _ -> assert false

