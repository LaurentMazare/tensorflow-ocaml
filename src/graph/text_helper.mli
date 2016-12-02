open Core_kernel.Std

val read_file
  :  string
  -> (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t * int Int.Map.t

val read_file_onehot
  :  string
  -> (float, 'a) Bigarray.kind
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array2.t * int Int.Map.t
