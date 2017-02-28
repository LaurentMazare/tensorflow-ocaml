open Core_kernel.Std
open Tensorflow_core

type 'a t

val create
  :  string
  -> (float, 'a) Bigarray.kind
  -> 'a t

val batch_sequence
  :  ?pos:int
  -> ?len:int
  -> 'a t
  -> seq_len:int
  -> batch_size:int
  -> ((float, 'a) Tensor.t * (float, 'a) Tensor.t) Sequence.t
  
val map : _ t -> int Int.Map.t

val length : _ t -> int

val dim : _ t -> int
