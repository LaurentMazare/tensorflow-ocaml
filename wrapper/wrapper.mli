(* TODO:
    - add a non-mutable status;
    - return [Result.t]
*)
module Tensor : sig
  type t

  val create1d : int -> t

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

  val create : unit -> t

  val set : t -> int -> string -> unit

  val code : t -> int

  val message : t -> string
end

module Session : sig
  type t

  val create : Session_options.t -> Status.t -> t

  val extend_graph : t -> 'a Ctypes.CArray.t -> int -> Status.t -> unit

  val run
    :  t
    -> inputs:(string * Tensor.t) list
    -> outputs:string list
    -> targets:string list
    -> Status.t
    -> Tensor.t list
end
