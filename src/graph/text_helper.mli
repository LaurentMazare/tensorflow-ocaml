open Core_kernel.Std

type t

val create : string -> t

val onehot
  :  t
  -> (float, 'a) Bigarray.kind
  -> pos:int
  -> len:int
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array2.t
  
val map : t -> int Int.Map.t
