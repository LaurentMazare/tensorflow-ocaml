type float32_genarray =
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Genarray.t

type t =
  { train_images : float32_genarray
  ; train_labels : float32_genarray
  ; validation_images : float32_genarray
  ; validation_labels : float32_genarray
  }

val read_files
  :  ?image_file:string
  -> ?label_file:string
  -> train_size:int
  -> validation_size:int
  -> unit
  -> t

val image_dim : int
val label_count : int

val slice1
  :  ('a, 'b, 'c) Bigarray.Array1.t
  -> int -> int -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t

val slice2
  :  ('a, 'b, 'c) Bigarray.Array2.t
  -> int -> int -> ('a, 'b, Bigarray.c_layout) Bigarray.Array2.t
