val read_images
  :  string
  -> (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array2.t

val read_labels
  :  string
  -> (Int32.t, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t
