type t
val const : float list -> t
val add : t -> t -> t
val sub : t -> t -> t
val mul : t -> t -> t
val div : t -> t -> t
val exp : t -> t

val name : t -> string

module Protobuf : sig
  val to_protobuf : t -> Protobuf.t
end
