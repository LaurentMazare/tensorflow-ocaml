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

module Session : sig
  type t

  val create
    :  ?session_options:Session_options.t
    -> unit
    -> t Status.result

  val extend_graph
    :  t
    -> Protobuf.t
    -> unit Status.result

  val run
    :  ?inputs:(string * Tensor.p) list
    -> ?outputs:string list
    -> ?targets:string list
    -> t
    -> Tensor.p list Status.result
end

module Graph : sig
  type t
  type operation
  type operation_description

  val create : unit -> t

  val new_operation
    :  t
    -> op_type:string
    -> op_name:string
    -> operation_description

  val finish_operation
    :  operation_description
    -> operation Status.result

  val add_input
    :  operation_description
    -> operation
    -> index:int
    -> unit
end
