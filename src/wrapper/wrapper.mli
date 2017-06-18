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

  type 'a result =
    | Ok of 'a
    | Error of t

  val ok_exn : 'a result -> 'a
end

module Graph : sig
  type t
  type operation
  type operation_description
  type output

  val create : unit -> t

  val new_operation
    :  t
    -> op_name:string
    -> name:string
    -> operation_description

  val finish_operation
    :  operation_description
    -> operation Status.result

  val add_control_input
    :  operation_description
    -> operation
    -> unit

  val add_input
    :  operation_description
    -> operation
    -> index:int
    -> unit

  val add_inputs
    :  operation_description
    -> (operation * int) list
    -> unit

  val create_output : operation -> index:int -> output

  val set_attr_int
    :  operation_description
    -> attr_name:string
    -> int
    -> unit

  val set_attr_int_list
    :  operation_description
    -> attr_name:string
    -> int list
    -> unit

  val set_attr_float
    :  operation_description
    -> attr_name:string
    -> float
    -> unit

  val set_attr_float_list
    :  operation_description
    -> attr_name:string
    -> float list
    -> unit

  val set_attr_bool
    :  operation_description
    -> attr_name:string
    -> bool
    -> unit

  val set_attr_bool_list
    :  operation_description
    -> attr_name:string
    -> bool list
    -> unit

  val set_attr_string
    :  operation_description
    -> attr_name:string
    -> string
    -> unit

  val set_attr_type
    :  operation_description
    -> attr_name:string
    -> data_type
    -> unit

  val set_attr_type_list
    :  operation_description
    -> attr_name:string
    -> data_type list
    -> unit

  val set_attr_tensor
    :  operation_description
    -> attr_name:string
    -> Tensor.p
    -> unit Status.result

  val set_attr_tensor_string
    :  operation_description
    -> attr_name:string
    -> shape:int list
    -> string list
    -> unit Status.result

  val set_attr_tensors
    :  operation_description
    -> attr_name:string
    -> Tensor.p list
    -> unit Status.result

  val set_attr_shape
    :  operation_description
    -> attr_name:string
    -> int list
    -> unit

  val import
    :  t
    -> string
    -> unit Status.result

  val find_operation
    :  t
    -> string
    -> operation option

  val shape
    :  t
    -> output
    -> int list Status.result

  val add_gradients
    :  t
    -> output list
    -> xs:output list
    -> output list Status.result
end

module Session : sig
  type t

  val create
    :  ?session_options:Session_options.t
    -> Graph.t
    -> t Status.result

  val run
    :  ?inputs:(Graph.output * Tensor.p) list
    -> ?outputs:Graph.output list
    -> ?targets:Graph.operation list
    -> t
    -> Tensor.p list Status.result
end
