open Ctypes
open Foreign

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
    @-> int
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
  foreign "TF_TensorByteSize" (tf_tensor @-> returning int)

let tf_tensordata =
  foreign "TF_TensorData" (tf_tensor @-> returning (ptr void))


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

let deallocate _ _ _ = ()

let create1d elts =
  let size = elts * 8 in
  let data =
    Ctypes.CArray.make char size
  in
  tf_newtensor 2
    (Ctypes.CArray.of_list int64_t [ Int64.of_int elts ] |> Ctypes.CArray.start)
    1
    (Ctypes.CArray.start data)
    (elts * 8)
    deallocate
    null

let () =
  let vector = create1d 100 in
  Printf.printf ">> %d %d %d\n%!" (tf_numdims vector) (tf_dim vector 0) (tf_tensorbytesize vector)
