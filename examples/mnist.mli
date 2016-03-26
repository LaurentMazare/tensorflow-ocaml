val read_images
  :  ?nsamples:int
  -> string
  -> (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t

val read_labels
  :  ?nsamples:int
  -> string
  -> (Int32.t, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t
