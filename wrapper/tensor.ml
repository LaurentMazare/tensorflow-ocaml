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
    @-> ptr char
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
