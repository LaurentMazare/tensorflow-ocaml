open Ctypes
open Foreign
let verbose = false

(* TF_TENSOR *)
type tf_tensor = unit ptr
let tf_tensor : tf_tensor typ = ptr void

type data_type =
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
  | Unknown of int

let data_type_to_int = function
  | TF_FLOAT -> 1
  | TF_DOUBLE -> 2
  | TF_INT32 -> 3
  | TF_UINT8 -> 4
  | TF_INT16 -> 5
  | TF_INT8 -> 6
  | TF_STRING -> 7
  | TF_COMPLEX -> 8
  | TF_INT64 -> 9
  | TF_BOOL -> 10
  | TF_QINT8 -> 11
  | TF_QUINT8 -> 12
  | TF_QINT32 -> 13
  | TF_BFLOAT16 -> 14
  | TF_QINT16 -> 15
  | TF_QUINT16 -> 16
  | TF_UINT16 -> 17
  | Unknown n -> n

let int_of_data_type = function
  | 1 -> TF_FLOAT
  | 2 -> TF_DOUBLE
  | 3 -> TF_INT32
  | 4 -> TF_UINT8
  | 5 -> TF_INT16
  | 6 -> TF_INT8
  | 7 -> TF_STRING
  | 8 -> TF_COMPLEX
  | 9 -> TF_INT64
  | 10 -> TF_BOOL
  | 11 -> TF_QINT8
  | 12 -> TF_QUINT8
  | 13 -> TF_QINT32
  | 14 -> TF_BFLOAT16
  | 15 -> TF_QINT16
  | 16 -> TF_QUINT16
  | 17 -> TF_UINT16
  | n -> Unknown n

let tf_newtensor =
  foreign "TF_NewTensor"
    (int
    @-> ptr int64_t (* dims *)
    @-> int         (* num dims *)
    @-> ptr void    (* data *)
    @-> size_t      (* len *)
    @-> funptr (ptr void @-> int @-> ptr void @-> returning void) (* deallocator *)
    @-> ptr void    (* deallocator arg *)
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

let tf_tensortype =
  foreign "TF_TensorType" (tf_tensor @-> returning int)

module Tensor = struct
  let fresh_id =
    let cnt = ref 0 in
    fun () -> incr cnt; !cnt

  type t =
    { tensor : tf_tensor
    ; mutable handled_by_ocaml : bool
    ; id : int
    (* Keep a reference to the data array to avoid it being GCed. *)
    ; data : char CArray.t option
    }

  (* Keep references to the allocated data until they have been deallocated. *)
  let live_tensors = Hashtbl.create 1024

  let deallocate _ _ id =
    let id = raw_address_of_ptr id |> Nativeint.to_int in
    if verbose
    then Printf.printf "Deallocating tensor %d\n%!" id;
    Hashtbl.remove live_tensors id

  let add_finaliser t =
    Gc.finalise
      (fun t ->
        if t.handled_by_ocaml
        then tf_deletetensor t.tensor)
      t;
    t

  let create1d typ elts =
    let elt_size = sizeof typ in
    let size = elts * elt_size in
    let data = CArray.make char size in
    let id = fresh_id () in
    let tensor =
      tf_newtensor 1
        (CArray.of_list int64_t [ Int64.of_int elts ] |> CArray.start)
        1
        (CArray.start data |> to_voidp)
        (Unsigned.Size_t.of_int size)
        deallocate
        (Nativeint.of_int id |> ptr_of_raw_address)
    in
    let t =
      { tensor
      ; handled_by_ocaml = true
      ; id
      ; data = Some data
      }
    in
    Hashtbl.add live_tensors id t;
    add_finaliser t

  let of_c_tensor tensor =
    let id = fresh_id () in
    add_finaliser
      { tensor
      ; handled_by_ocaml = true
      ; id
      ; data = None
      }

  let assert_handled_by_ocaml t =
    if not t.handled_by_ocaml
    then failwith "This tensor is not handled by ocaml anymore."

  let num_dims t =
    assert_handled_by_ocaml t;
    tf_numdims t.tensor

  let dim t =
    assert_handled_by_ocaml t;
    tf_dim t.tensor

  let byte_size t =
    assert_handled_by_ocaml t;
    tf_tensorbytesize t.tensor |> Unsigned.Size_t.to_int

  let data t typ len =
    assert_handled_by_ocaml t;
    CArray.from_ptr (tf_tensordata t.tensor |> Ctypes.from_voidp typ) len

  let data_type t =
    assert_handled_by_ocaml t;
    tf_tensortype t.tensor
    |> int_of_data_type
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
  type t = tf_status

  type code =
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
    | Unknown of int
  
  (* CAUTION: this has to stay in sync with [tensor_c_api.h], maybe we should generate
     some stubs to assert this at compile time. *)
  let code_of_int = function
    | 0  -> TF_OK
    | 1  -> TF_CANCELLED
    | 2  -> TF_UNKNOWN
    | 3  -> TF_INVALID_ARGUMENT
    | 4  -> TF_DEADLINE_EXCEEDED
    | 5  -> TF_NOT_FOUND
    | 6  -> TF_ALREADY_EXISTS
    | 7  -> TF_PERMISSION_DENIED
    | 16 -> TF_UNAUTHENTICATED
    | 8  -> TF_RESOURCE_EXHAUSTED
    | 9  -> TF_FAILED_PRECONDITION
    | 10 -> TF_ABORTED
    | 11 -> TF_OUT_OF_RANGE
    | 12 -> TF_UNIMPLEMENTED
    | 13 -> TF_INTERNAL
    | 14 -> TF_UNAVAILABLE
    | 15 -> TF_DATA_LOSS
    | n  -> Unknown n

  let create () =
    let status = tf_newstatus () in
    Gc.finalise tf_deletestatus status;
    status

  let set = tf_setstatus

  let code t = tf_getcode t |> code_of_int

  let message = tf_message
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
  type t = tf_sessionoptions

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
  foreign "TF_ExtendGraph" (tf_session @-> string @-> size_t @-> tf_status @-> returning void)

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
  type t = tf_session

  type 'a result =
    | Ok of 'a
    | Error of Status.t

  let result_or_error status v =
    match Status.code status with
    | TF_OK -> Ok v
    | _ -> Error status

  let create ?session_options () =
    let session_options =
      match session_options with
      | None -> Session_options.create ()
      | Some session_options -> session_options
    in
    let status = Status.create () in
    let session = tf_newsession session_options status in
    Gc.finalise
      (fun session ->
        tf_closesession session status;
        tf_deletesession session status)
      session;
    result_or_error status session

  let extend_graph t protobuf =
    let status = Status.create () in
    let protobuf = Protobuf.to_string protobuf in
    tf_extendgraph
      t
      protobuf
      (String.length protobuf |> Unsigned.Size_t.of_int)
      status;
    result_or_error status ()

  let run t ~inputs ~outputs ~targets =
    let status = Status.create () in
    let input_names, input_tensors = List.split inputs in
    let input_tensors =
      List.map
        (fun (tensor : Tensor.t) ->
          (* The memory will be handled by the C++ side. *)
          tensor.handled_by_ocaml <- false;
          tensor.tensor)
        input_tensors
    in
    let output_len = List.length outputs in
    let output_tensors = CArray.make tf_tensor output_len in
    tf_run
      t
      CArray.(of_list string input_names |> start)
      CArray.(of_list tf_tensor input_tensors |> start)
      (List.length inputs)
      CArray.(of_list string outputs |> start)
      (CArray.start output_tensors)
      output_len
      CArray.(of_list string targets |> start)
      (List.length targets)
      status;
    let output_tensors =
      CArray.to_list output_tensors
      |> List.map Tensor.of_c_tensor
    in
    result_or_error status output_tensors
end

let () =
  ignore
    ( data_type_to_int
    , tf_settarget
    , tf_setconfig
    , Status.set)
