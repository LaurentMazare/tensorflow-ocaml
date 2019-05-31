open Tensorflow_core

type t =
  { tensor : (float, Bigarray.float32_elt) Tensor.t
  ; width : int
  ; height : int
  }

val load : string -> t
val save : (float, Bigarray.float32_elt) Tensor.t -> string -> unit
