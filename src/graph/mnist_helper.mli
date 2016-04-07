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

val train_batch
  :  t
  -> batch_size:int
  -> batch_idx:int
  -> float32_genarray * float32_genarray

val image_dim : int
val label_count : int

val accuracy
  :  float32_genarray
  -> float32_genarray
  -> float
