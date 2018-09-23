(** [batch_normalization ?decay node ~update_moments ~dims ~feature_count] takes as
    input a node which last dimension is assumed to be the feature dimension.
    [dims] has to be the number of dimensions for [node] excluding the feature
    dimension but including the batch dimension.
    [feature_count] is the number of features in the last dimension of [node].
    If [update_moments] is [ `always ] or [ `not_in_testing false ] the batch
    mean and variance are computed and used to update the normalization
    variables.
*)
val batch_normalization
  :  ?decay:float
  -> ([< `double | `float ] as 'a) Node.t
  -> update_moments:[ `always | `not_in_testing of [ `bool ] Node.t ]
  -> dims:int
  -> feature_count:int
  -> 'a Node.t

type 'a linear

type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu of float (* max xs (alpha * xs) *)
  | Sigmoid

val linear_with_vars
  :  ?activation:activation
  -> ([< `double | `float ] as 'a) Node.t
  -> output_dim:int
  -> 'a linear

val linear
  :  ?activation:activation
  -> ([< `double | `float ] as 'a) Node.t
  -> output_dim:int
  -> 'a Node.t

val linear_vars : 'a linear -> 'a Node.t list
val linear_output : 'a linear -> 'a Node.t
val linear_apply
  :  ([< `double | `float ] as 'a) linear
  -> 'a Node.t
  -> 'a Node.t

type padding =
  | Same
  | Valid

val max_pool
  :  ?padding:padding (* default: Same *)
  -> ([< `double | `float ] as 'a) Node.t
  -> ksize:(int * int)
  -> strides:(int * int)
  -> 'a Node.t

val conv2d
  :  ?padding:padding (* default: Same *)
  -> ([< `double | `float ] as 'a) Node.t
  -> ksize:(int * int)
  -> strides:(int * int)
  -> output_dim:int
  -> 'a Node.t

(** [flatten] preserves the first (batch) dimension. *)
val flatten
  :  'a Node.t
  -> 'a Node.t

val reshape
  : 'a Node.t
  -> shape:int list
  -> 'a Node.t
