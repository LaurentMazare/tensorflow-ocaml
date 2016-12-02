open Core_kernel.Std

type 'a t

val create
  :  string
  -> (float, 'a) Bigarray.kind
  -> 'a t

val batch_sequence
  :  'a t
  -> pos:int
  -> len:int
  -> seq_len:int
  -> batch_size:int
  -> ((float, 'a, Bigarray.c_layout) Bigarray.Array3.t
   * (float, 'a, Bigarray.c_layout) Bigarray.Array3.t) Sequence.t
  
val map : _ t -> int Int.Map.t

val length : _ t -> int

val dim : _ t -> int
