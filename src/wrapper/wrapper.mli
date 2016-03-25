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
