(* TODO:
    - Improve the API.
      - Add a non-mutable status.
      - Return [Result.t].
    - Handle memory properly.
*)
module Tensor : sig
  type t

  val create1d : 'a Ctypes.typ -> int -> t

  val num_dims : t -> int

  val dim : t -> int -> int

  val byte_size : t -> int

  val data : t -> 'a Ctypes.typ -> int -> 'a Ctypes.CArray.t
end

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

  val create : unit -> t

  val set : t -> int -> string -> unit

  val code : t -> code

  val message : t -> string
end

module Session : sig
  type t

  val create : Session_options.t -> Status.t -> t

  val extend_graph : t -> Protobuf.t -> Status.t -> unit

  val run
    :  t
    -> inputs:(string * Tensor.t) list
    -> outputs:string list
    -> targets:string list
    -> [ `Ok of Tensor.t list | `Error of (Status.code * string) ]
end
