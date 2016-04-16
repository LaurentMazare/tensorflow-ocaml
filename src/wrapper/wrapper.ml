open Ctypes
open Foreign

let () =
  (* Hacky solution for now, try loading python but do not fail if not available,
     this way we'll only fail later if the symbols are needed. *)
  try
    ignore
      (Dl.dlopen ~filename:"libpython2.7.so" ~flags:[ RTLD_GLOBAL; RTLD_LAZY ] : Dl.library)
  with _ -> ()

let from = Dl.dlopen ~filename:"libtensorflow.so" ~flags:[ RTLD_LAZY ]

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

let int_to_data_type = function
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
  foreign "TF_NewTensor" ~from
    (int
    @-> ptr int64_t (* dims *)
    @-> int         (* num dims *)
    @-> ptr void    (* data *)
    @-> size_t      (* len *)
    @-> funptr (ptr void @-> int @-> ptr void @-> returning void) (* deallocator *)
    @-> ptr void    (* deallocator arg *)
    @-> returning tf_tensor)

let tf_deletetensor =
  foreign "TF_DeleteTensor" ~from (tf_tensor @-> returning void)

let tf_numdims =
  foreign "TF_NumDims" ~from (tf_tensor @-> returning int)

let tf_dim =
  foreign "TF_Dim" ~from (tf_tensor @-> int @-> returning int)

let tf_tensorbytesize =
  foreign "TF_TensorByteSize" ~from (tf_tensor @-> returning size_t)

let tf_tensordata =
  foreign "TF_TensorData" ~from (tf_tensor @-> returning (ptr void))

let tf_tensortype =
  foreign "TF_TensorType" ~from (tf_tensor @-> returning int)

(* TF_STATUS *)
type tf_status = unit ptr
let tf_status : tf_status typ = ptr void

let tf_newstatus =
  foreign "TF_NewStatus" ~from (void @-> returning tf_status)

let tf_deletestatus =
  foreign "TF_DeleteStatus" ~from (tf_status @-> returning void)

let tf_setstatus =
  foreign "TF_SetStatus" ~from (tf_status @-> int @-> string @-> returning void)

let tf_getcode =
  foreign "TF_GetCode" ~from (tf_status @-> returning int)

let tf_message =
  foreign "TF_Message" ~from (tf_status @-> returning string)

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
  foreign "TF_NewSessionOptions" ~from (void @-> returning tf_sessionoptions)

let tf_settarget =
  foreign "TF_SetTarget" ~from (tf_sessionoptions @-> string @-> returning void)

let tf_setconfig =
  foreign "TF_SetConfig" ~from
    (tf_sessionoptions
    @-> ptr void
    @-> size_t
    @-> tf_status
    @-> returning tf_sessionoptions)

let tf_deletesessionoptions =
  foreign "TF_DeleteSessionOptions" ~from (tf_sessionoptions @-> returning void)

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
  foreign "TF_NewSession" ~from
    (tf_sessionoptions @-> tf_status @-> returning tf_session)

let tf_closesession =
  foreign "TF_CloseSession" ~from (tf_session @-> tf_status @-> returning void)

let tf_deletesession =
  foreign "TF_DeleteSession" ~from (tf_session @-> tf_status @-> returning void)

let tf_extendgraph =
  foreign "TF_ExtendGraph" ~from
    (tf_session @-> string @-> size_t @-> tf_status @-> returning void)

(* We use [ptr (ptr char)] rather than [ptr string] as when using [CArray.of_list]
   to build the [ptr string], the reference to the C (copied) version of the string can
   get lost which will be problematic if the GC triggers just after. *)
let tf_run =
  foreign "TF_Run" ~from
    (tf_session
    @-> ptr void (* run_options *)
    (* Input tensors *)
    @-> ptr (ptr char)
    @-> ptr tf_tensor
    @-> int
    (* Output tensors *)
    @-> ptr (ptr char)
    @-> ptr tf_tensor
    @-> int
    (* Target nodes *)
    @-> ptr (ptr char)
    @-> int
    @-> ptr void (* run_metadata *)
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

  let sizeof = function
    | TF_FLOAT -> 4
    | TF_DOUBLE -> 8
    | TF_INT32 -> 4
    | TF_UINT16
    | TF_INT16 -> 2
    | TF_UINT8
    | TF_INT8 -> 1
    | TF_INT64 -> 8
    | TF_STRING
    | TF_COMPLEX
    | TF_BOOL
    | TF_QINT8
    | TF_QUINT8
    | TF_QINT32
    | TF_BFLOAT16
    | TF_QINT16
    | TF_QUINT16
    | Unknown _ -> failwith "Unsupported tensor type"

  let data_type_of_kind (type a) (type b) (kind : (a, b) Bigarray.kind) =
    match kind with
    | Bigarray.Float32 -> TF_FLOAT
    | Bigarray.Float64 -> TF_DOUBLE
    | Bigarray.Int64 -> TF_INT64
    | Bigarray.Int32 -> TF_INT32
    | _ -> failwith "Unsupported yet"

  let fresh_id =
    let cnt = ref 0 in
    fun () -> incr cnt; !cnt

  let live_tensors = Hashtbl.create 1024
  let deallocate _ _ id =
    let id = raw_address_of_ptr id |> Nativeint.to_int in
    if verbose
    then Printf.printf "Deallocating tensor %d\n%!" id;
    Hashtbl.remove live_tensors id

  let c_tensor_of_tensor packed_tensor =
    let Tensor.P tensor = packed_tensor in
    let id = fresh_id () in
    let dim_array = Bigarray.Genarray.dims tensor in
    let dims =
      Array.to_list dim_array
      |> List.map Int64.of_int
      |> CArray.of_list int64_t
      |> CArray.start
    in
    let data_type = Bigarray.Genarray.kind tensor |> data_type_of_kind in
    let size = Array.fold_left ( * ) 1 dim_array * sizeof data_type in
    Hashtbl.add live_tensors id packed_tensor;
    tf_newtensor (data_type_to_int data_type)
      dims
      (Bigarray.Genarray.num_dims tensor)
      (bigarray_start genarray tensor |> to_voidp)
      (Unsigned.Size_t.of_int size)
      deallocate
      (Nativeint.of_int id |> ptr_of_raw_address)

  let tensor_of_c_tensor c_tensor =
    let num_dims = tf_numdims c_tensor in
    let dims = Array.init num_dims (fun i -> tf_dim c_tensor i) in
    let apply_kind kind =
      let data = tf_tensordata c_tensor |> from_voidp (typ_of_bigarray_kind kind) in
      let data = bigarray_of_ptr genarray dims kind data in
      Gc.finalise (fun _data -> tf_deletetensor c_tensor) data;
      Tensor.P data
    in
    match tf_tensortype c_tensor |> int_to_data_type with
    | TF_FLOAT -> apply_kind Bigarray.float32
    | TF_DOUBLE -> apply_kind Bigarray.float64
    | TF_INT32 -> apply_kind Bigarray.int32
    | TF_INT64 -> apply_kind Bigarray.int64
    | _ -> failwith "Unsupported tensor type"

  let char_list_of_string s =
    let rec char_list_of_string idx acc =
      if idx < 0
      then acc
      else char_list_of_string (idx - 1) (s.[idx] :: acc)
    in
    char_list_of_string (String.length s - 1) [ Char.chr 0 ]

  let ptr_ptr_char l =
    let l =
      List.map (fun s -> char_list_of_string s |> CArray.of_list char) l
    in
    let ptr =
      CArray.of_list (ptr char) (List.map CArray.start l)
      |> CArray.start
    in
    (* Keep a reference to l as it could be GCed otherwise. *)
    Gc.finalise (fun _ -> ignore l; ()) ptr;
    ptr

  let run ?(inputs = []) ?(outputs = []) ?(targets = []) t =
    let status = Status.create () in
    let input_names, input_tensors = List.split inputs in
    let input_tensors = List.map c_tensor_of_tensor input_tensors in
    let output_len = List.length outputs in
    let output_tensors = CArray.make tf_tensor output_len in
    tf_run
      t
      null
      (ptr_ptr_char input_names)
      CArray.(of_list tf_tensor input_tensors |> start)
      (List.length inputs)
      (ptr_ptr_char outputs)
      (CArray.start output_tensors)
      output_len
      (ptr_ptr_char targets)
      (List.length targets)
      null
      status;
    match result_or_error status () with
    | Ok () ->
      let output_tensors =
        CArray.to_list output_tensors
        |> List.map tensor_of_c_tensor
      in
      Ok output_tensors
    | Error _ as err -> err

  let ok_exn = function
    | Ok ok -> ok
    | Error status ->
      failwith
        (Printf.sprintf "%d %s" (tf_getcode status) (Status.message status))
end

let () =
  ignore
    ( data_type_to_int
    , tf_settarget
    , tf_setconfig
    , tf_tensorbytesize
    , Status.set)
