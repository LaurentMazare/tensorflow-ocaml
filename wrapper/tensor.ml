open Ctypes
open Foreign

(* TF_TENSOR *)
type tf_tensor = unit ptr
let tf_tensor : tf_tensor typ = ptr void

type tf_datatype =
  | TF_FLOAT
  | TF_DOUBLE
  | TF_INT32
  | TF_UINT8
  | TF_INT16
  | TF_INT8
  | TF_STRING
  | TF_COMPLEX
  | TF_INT64
  | TF_BOOL
  | TF_QINT8
  | TF_QUINT8
  | TF_QINT32
  | TF_BFLOAT16
  | TF_QINT16
  | TF_QUINT16
  | TF_UINT16

module Bindings (S : Cstubs.Types.TYPE) = struct
  let tf_datatype =
    S.enum "TF_DataType"
      [ TF_FLOAT, S.constant "TF_FLOAT" S.int64_t
      ]
end

type tf_code =
  | TF_OK
  | TF_CANCELLED
  | TF_UNKNOWN
  | TF_INVALID_ARGUMENT
  | TF_DEADLINE_EXCEEDED
  | TF_NOT_FOUND
  | TF_ALREADY_EXISTS
  | TF_PERMISSION_DENIED
  | TF_UNAUTHENTICATED
  | TF_RESOURCE_EXHAUSTED
  | TF_FAILED_PRECONDITION
  | TF_ABORTED
  | TF_OUT_OF_RANGE
  | TF_UNIMPLEMENTED
  | TF_INTERNAL
  | TF_UNAVAILABLE
  | TF_DATA_LOSS

let tf_newtensor =
  foreign "TF_NewTensor"
    (int
    @-> ptr int64_t
    @-> int
    @-> ptr char
    @-> size_t
    @-> funptr (ptr void @-> int @-> ptr void @-> returning void)
    @-> ptr void
    @-> returning tf_tensor)

let tf_deletetensor =
  foreign "TF_DeleteTensor" (tf_tensor @-> returning void)

let tf_numdims =
  foreign "TF_NumDims" (tf_tensor @-> returning int)

let tf_dim =
  foreign "TF_Dim" (tf_tensor @-> int @-> returning int)

let tf_tensorbytesize =
  foreign "TF_TensorByteSize" (tf_tensor @-> returning size_t)

let tf_tensordata =
  foreign "TF_TensorData" (tf_tensor @-> returning (ptr void))

module Tensor = struct
  (* TODO: actually store references to data at top-level and only remove them in [deallocate]. *)
  let deallocate _ _ _ = ()

  let create1d elts =
    let elt_size = sizeof float in
    let size = elts * elt_size in
    let data = Ctypes.CArray.make char size in
    tf_newtensor 1
      (Ctypes.CArray.of_list int64_t [ Int64.of_int elts ] |> Ctypes.CArray.start)
      1
      (Ctypes.CArray.start data)
      (Unsigned.Size_t.of_int size)
      deallocate
      null,
    data
end

(* TF_STATUS *)
type tf_status = unit ptr
let tf_status : tf_status typ = ptr void

let tf_newstatus =
  foreign "TF_NewStatus" (void @-> returning tf_status)

let tf_deletestatus =
  foreign "TF_DeleteStatus" (tf_status @-> returning void)

let tf_setstatus =
  foreign "TF_SetStatus" (tf_status @-> int @-> string @-> returning void)

let tf_getcode =
  foreign "TF_GetCode" (tf_status @-> returning int)

let tf_message =
  foreign "TF_Message" (tf_status @-> returning string)

module Status = struct
  let create () =
    let status = tf_newstatus () in
    Gc.finalise tf_deletestatus status;
    status
end

(* TF_SESSIONOPTIONS *)
type tf_sessionoptions = unit ptr
let tf_sessionoptions : tf_sessionoptions typ = ptr void

let tf_newsessionoptions =
  foreign "TF_NewSessionOptions" (void @-> returning tf_sessionoptions)

let tf_settarget =
  foreign "TF_SetTarget" (tf_sessionoptions @-> string @-> returning void)

let tf_setconfig =
  foreign "TF_SetConfig"
    (tf_sessionoptions
    @-> ptr void
    @-> size_t
    @-> tf_status
    @-> returning tf_sessionoptions)

let tf_deletesessionoptions =
  foreign "TF_DeleteSessionOptions" (tf_sessionoptions @-> returning void)

module Session_options = struct
  let create () =
    let session_options = tf_newsessionoptions () in
    Gc.finalise tf_deletesessionoptions session_options;
    session_options
end

(* TF_SESSION *)
type tf_session = unit ptr
let tf_session : tf_session typ = ptr void

let tf_newsession =
  foreign "TF_NewSession" (tf_sessionoptions @-> tf_status @-> returning tf_session)

let tf_closesession =
  foreign "TF_CloseSession" (tf_session @-> tf_status @-> returning void)

let tf_deletesession =
  foreign "TF_DeleteSession" (tf_session @-> tf_status @-> returning void)

let tf_extendgraph =
  foreign "TF_ExtendGraph" (tf_session @-> ptr char @-> size_t @-> tf_status @-> returning void)

let tf_run =
  foreign "TF_Run"
    (tf_session
    (* Input tensors *)
    @-> ptr string
    @-> ptr tf_tensor
    @-> int
    (* Output tensors *)
    @-> ptr string
    @-> ptr tf_tensor
    @-> int
    (* Target nodes *)
    @-> ptr string
    @-> int
    (* Output status *)
    @-> tf_status
    @-> returning void)

module Session = struct
  let create session_options status =
    let session = tf_newsession session_options status in
    Gc.finalise
      (fun session ->
        let status = Status.create () in
        tf_deletesession session status)
      session;
    session
end

(* API END *)

(* Example code. TODO: move in a separate file. *)
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
  let vector, data = Tensor.create1d 10 in
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
    (Ctypes.CArray.start carray)
    (String.length simple_pbtxt |> Unsigned.Size_t.of_int)
    status;
  Printf.printf "%d %s\n%!" (tf_getcode status) (tf_message status);
  for i = 0 to 39 do
    Printf.printf "%d " (Ctypes.CArray.get data i |> Char.code)
  done;
  Printf.printf "\n%!";
  tf_run
    session
    Ctypes.CArray.(of_list string [] |> start)
    Ctypes.CArray.(of_list tf_tensor [] |> start)
    0
    Ctypes.CArray.(of_list string [ "add" ] |> start)
    Ctypes.CArray.(of_list tf_tensor [ vector ] |> start)
    1
    Ctypes.CArray.(of_list string [ "add" ] |> start)
    1
    status;
  Printf.printf "%d %s\n%!" (tf_getcode status) (tf_message status);
  for i = 0 to 39 do
    Printf.printf "%d " (Ctypes.CArray.get data i |> Char.code)
  done;
  Printf.printf "\n%!"
