type t

val input
  :  shape:int list
  -> [ `float ] Node.t * t

val dense
  (* TODO: add init *)
  :  t
  -> shape:int list
  -> t

val sigmoid : t -> t

val tanh : t -> t

val relu : t -> t
