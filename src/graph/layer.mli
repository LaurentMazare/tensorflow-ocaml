(** [batch_normalization node ~testing ~shape] takes as input a node which first dimension is
    assumed to be the batch dimension. The following dimension have to match [shape].
    If [testing] is false the batch mean and variance are computed and used to update the
    normalization variables.
*)
val batch_normalization
  : ([< `double | `float ] as 'a) Node.t
  -> testing:[ `bool ] Node.t
  -> shape:int list
  -> 'a Node.t

