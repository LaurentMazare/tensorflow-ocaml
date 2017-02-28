open Tensorflow_core
type float32_tensor = (float, Bigarray.float32_elt) Tensor.t

type t =
  { train_images : float32_tensor
  ; train_labels : float32_tensor
  ; test_images : float32_tensor
  ; test_labels : float32_tensor
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
  -> float32_tensor * float32_tensor

val image_dim : int
val label_count : int

val accuracy
  :  float32_tensor
  -> float32_tensor
  -> float
