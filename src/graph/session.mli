open Tensorflow_core

module Input : sig
  type t

  val float : [ `float ] Ops.Placeholder.t -> (float, Bigarray.float32_elt) Tensor.t -> t

  val double
    :  [ `double ] Ops.Placeholder.t
    -> (float, Bigarray.float64_elt) Tensor.t
    -> t

  val bool
    :  [ `bool ] Ops.Placeholder.t
    -> (int, Bigarray.int8_unsigned_elt) Tensor.t
    -> t
end

module Output : sig
  type 'a t

  val return : 'a -> 'a t
  val map : 'a t -> f:('a -> 'b) -> 'b t
  val both : 'a t -> 'b t -> ('a * 'b) t
  val three : 'a t -> 'b t -> 'c t -> ('a * 'b * 'c) t
  val four : 'a t -> 'b t -> 'c t -> 'd t -> ('a * 'b * 'c * 'd) t
  val five : 'a t -> 'b t -> 'c t -> 'd t -> 'e t -> ('a * 'b * 'c * 'd * 'e) t

  val six
    :  'a t
    -> 'b t
    -> 'c t
    -> 'd t
    -> 'e t
    -> 'f t
    -> ('a * 'b * 'c * 'd * 'e * 'f) t

  val empty : unit t
  val float : [ `float ] Node.t -> (float, Bigarray.float32_elt) Tensor.t t
  val double : [ `double ] Node.t -> (float, Bigarray.float64_elt) Tensor.t t
  val int32 : [ `int32 ] Node.t -> (int32, Bigarray.int32_elt) Tensor.t t
  val int64 : [ `int64 ] Node.t -> (int64, Bigarray.int64_elt) Tensor.t t

  (* Useful for loss *)
  val scalar_float : [ `float ] Node.t -> float t
  val scalar_double : [ `double ] Node.t -> float t
  val scalar_int32 : [ `int32 ] Node.t -> int t
  val scalar_int64 : [ `int64 ] Node.t -> int64 t
end

val run : ?inputs:Input.t list -> ?targets:Node.p list -> 'a Output.t -> 'a

module Vars : sig
  val set_float
    :  ([ `float ] Node.t * (float, Bigarray.float32_elt) Tensor.t) list
    -> unit

  val set_double
    :  ([ `double ] Node.t * (float, Bigarray.float64_elt) Tensor.t) list
    -> unit
end
