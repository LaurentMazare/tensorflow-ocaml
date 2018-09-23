(* The batch normalization is applied to the last dimension. *)
module Batch_norm : sig
  type 'a t

  val create
    :  ?epsilon:float
    -> ?decay:float
    -> ([< `double | `float ] as 'a) Node.t
    -> 'a t

  val apply_infer
    :  ([< `double | `float ] as 'a) t
    -> 'a Node.t
    -> 'a Node.t

  val apply_train
    :  ([< `double | `float ] as 'a) t
    -> 'a Node.t
    -> 'a Node.t * [ `update_ops of 'a Node.t list ]
end

(** [batch_norm ?decay node ~is_training] takes as input a node which last
    dimension is assumed to be the feature dimension on which batch norm is
    computed.
    When [is_training] is true the batch statistics from [node] are used.
    When false variables storing the running mean and variance are used instead.
*)
val batch_norm
  :  ?decay:float
  -> ([< `double | `float ] as 'a) Node.t
  -> is_training:[ `bool ] Node.t
  -> 'a Node.t * [ `update_ops of 'a Node.t list ]

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
