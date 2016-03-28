type float32_genarray =
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Genarray.t

type t =
  { train_images : float32_genarray
  ; train_labels : float32_genarray
  ; test_images : float32_genarray
  ; test_labels : float32_genarray
  }

val read_files
  :  ?train_image_file:string
  -> ?train_label_file:string
  -> ?test_image_file:string
  -> ?test_label_file:string
  -> unit
  -> t

val image_dim : int
val label_count : int

val slice2
  :  ('a, 'b, 'c) Bigarray.Array2.t
  -> int -> int -> ('a, 'b, Bigarray.c_layout) Bigarray.Array2.t
