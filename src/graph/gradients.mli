(* [gradient_caml] uses a caml implemented backpropagation. *)
val gradient_caml
  :  [< `double | `float ] Node.t
  -> with_respect_to_float:[ `float ] Node.t list
  -> with_respect_to_double:[ `double ] Node.t list
  -> [ `float ] Node.t list * [ `double ] Node.t list

(* [gradient_tf] uses the TensorFlow C++ backpropagation. *)
val gradient_tf
  :  [< `double | `float ] Node.t
  -> with_respect_to_float:[ `float ] Node.t list
  -> with_respect_to_double:[ `double ] Node.t list
  -> [ `float ] Node.t list * [ `double ] Node.t list
