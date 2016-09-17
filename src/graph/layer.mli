(** [batch_normalization ?decay node ~testing ~dims ~feature_count] takes as
    input a node which last dimension is assumed to be the feature dimension.
    [dims] has to be the number of dimensions for [node] excluding the feature
    dimension but including the batch dimension.
    [feature_count] is the number of features in the last dimension of [node].
    If [testing] is false the batch mean and variance are computed and used to update the
    normalization variables.
*)
val batch_normalization
  :  ?decay:float
  -> ([< `double | `float ] as 'a) Node.t
  -> testing:[ `bool ] Node.t
  -> dims:int
  -> feature_count:int
  -> 'a Node.t

