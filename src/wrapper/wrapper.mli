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

module Session_options : sig
  type t

  val create : unit -> t
end

module Status : sig
  type t
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

  val code : t -> code

  val message : t -> string
end

module Session : sig
  type t
  type 'a result =
    | Ok of 'a
    | Error of Status.t

  val create
    :  ?session_options:Session_options.t
    -> unit
    -> t result

  val extend_graph
    :  t
    -> Protobuf.t
    -> unit result

  val run
    :  ?inputs:(string * Tensor.p) list
    -> ?outputs:string list
    -> ?targets:string list
    -> t
    -> Tensor.p list result
end
