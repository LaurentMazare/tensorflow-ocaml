module Placeholder : sig
  type 'a t

  val to_node : 'a t -> 'a Node.t
end

(* ==== Binary Operations ==== *)

(* The common type for binary operators. *)
type 't b = ?name:string -> 't Node.t -> 't Node.t -> 't Node.t

(* Pointwise arithmetic operations. *)
val ( + ) : [< `float | `double | `int32 | `int64 | `complex64 | `string ] b
val ( - ) : [< `float | `double | `int32 | `int64 | `complex64 ] b
val ( / ) : [< `float | `double | `int32 | `int64 | `complex64 ] b
val ( * ) : [< `float | `double | `int32 | `int64 | `complex64 ] b

(* Matrix multiplication. *)
val ( *^ ) : [< `float | `double | `int32 | `complex64 ] b

(* ==== Constant Tensors ==== *)

(* Constant tensors using a single value. *)
val f_or_d
  :  ?shape:int list
  -> type_:([< `float | `double ] as 'a) Node.Type.t
  -> float
  -> 'a Node.t

val f : ?shape:int list -> float -> [ `float ] Node.t
val d : ?shape:int list -> float -> [ `double ] Node.t

(* Constant tensors using different values. *)
val cf : ?shape:int list -> float list -> [ `float ] Node.t
val cd : ?shape:int list -> float list -> [ `double ] Node.t
val ci32 : ?shape:int list -> int list -> [ `int32 ] Node.t
val ci64 : ?shape:int list -> int list -> [ `int64 ] Node.t

(* Some more refined constant creation functions. *)
val const_float
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ?shape:int list
  -> type_:([< `float | `double ] as 'dtype) Node.Type.t
  -> float list
  -> 'dtype Node.t

val const_int
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ?shape:int list
  -> type_:([< `int32 | `int64 ] as 'dtype) Node.Type.t
  -> int list
  -> 'dtype Node.t

val const_string : ?name:string -> ?shape:int list -> string list -> [ `string ] Node.t
val const_string0 : ?name:string -> string -> [ `string ] Node.t

val scalar
  :  ?empty_shape:unit
  -> type_:([< `float | `double ] as 'dtype) Node.Type.t
  -> float
  -> 'dtype Node.t

(* Useful int scalar values. *)
val four32 : [ `int32 ] Node.t
val three32 : [ `int32 ] Node.t
val two32 : [ `int32 ] Node.t
val one32 : [ `int32 ] Node.t
val zero32 : [ `int32 ] Node.t

(* ==== Reduction Functions ==== *)

(* Reduction functions, [dims] is the list of dimensions across which the reduction
   is performed. *)
type 'a reduce_fn =
  ?dims:int list
  -> ([< `complex64 | `double | `float | `int32 | `int64 ] as 'a) Node.t
  -> 'a Node.t

val reduce_sum : 'a reduce_fn
val reduce_min : 'a reduce_fn
val reduce_max : 'a reduce_fn
val reduce_mean : 'a reduce_fn
val reduce_prod : 'a reduce_fn
val reduce_all : ?dims:int list -> [ `bool ] Node.t -> [ `bool ] Node.t
val reduce_any : ?dims:int list -> [ `bool ] Node.t -> [ `bool ] Node.t

(* ==== Saving Tensors to disk ==== *)
val save_
  :  ?name:string
  -> [ `string ] Node.t (* Filename *)
  -> [ `string ] Node.t (* Tensor names *)
  -> Node.p list (* Tensor to save. *)
  -> [ `unit ] Node.t

val save : filename:string -> (string * Node.p) list -> [ `unit ] Node.t

(* ==== Split ==== *)
val split2 : ?name:string -> [ `int32 ] Node.t -> 't Node.t -> 't Node.t * 't Node.t

val split3
  :  ?name:string
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t * 't Node.t * 't Node.t

val split4
  :  ?name:string
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t * 't Node.t * 't Node.t * 't Node.t

(* ==== Misc ==== *)

(* [range n] where [n] is a scalar tensor returns tensor [ 0; 1; ...; n-1 ]. *)
val range : [ `int32 ] Node.t -> [ `int32 ] Node.t

(* A placeholder that can be bound to a tensor via [inputs] in [Session.run]. *)
val placeholder : ?name:string -> type_:'a Node.Type.t -> int list -> 'a Placeholder.t

(* [dropout n ~keep_prob] returns a tensor with the same shape as [n] that either
   have the same value as [n] with prob [keep_prob] or else is [0].
   The resulting values are then scaled by [1/keep_prob]. *)
val dropout : ([< `float | `double ] as 'a) Node.t -> keep_prob:'a Node.t -> 'a Node.t
val cast : ?name:string -> 'srcT Node.t -> type_:'dstT Node.Type.t -> 'dstT Node.t
val count : 'a Node.t -> dims:int list -> [ `int32 ] Node.t

type 'a moments =
  { mean : 'a Node.t
  ; variance : 'a Node.t
  }

val moments : ([< `double | `float ] as 'a) Node.t -> dims:int list -> 'a moments

val normalize
  :  ?epsilon:float
  -> ([< `double | `float ] as 'a) Node.t
  -> 'a moments
  -> 'a Node.t

(* If [if_true] and [if_false] use their [control_input] argument to build a node
   this node will only be evaluated if necessary. *)
val cond_with_control_inputs
  :  [ `bool ] Node.t
  -> if_true:(control_inputs:Node.p list -> 'a Node.t)
  -> if_false:(control_inputs:Node.p list -> 'a Node.t)
  -> 'a Node.t

(* [if_true] and [if_false] will always be evaluated because of the 'not-so lazy'
   behavior of TensorFlow switch. *)
val cond : [ `bool ] Node.t -> if_true:'a Node.t -> if_false:'a Node.t -> 'a Node.t

val shape32
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't Node.t
  -> [ `int32 ] Node.t

(* TODO: add a logit version similar to tf.nn.sigmoid_cross_entropy_with_logits. *)
(* TODO: use 'labels' rather than ys. *)
val cross_entropy
  :  ?epsilon:float
  -> ys:([< `double | `float ] as 'a) Node.t (** Actual y values. *)
  -> y_hats:'a Node.t (** Predicted y values. *)
  -> [ `sum | `mean ]
  -> 'a Node.t

val binary_cross_entropy
  :  ?epsilon:float
  -> labels:([< `double | `float ] as 'a) Node.t (** Actual y values. *)
  -> model_values:'a Node.t (** Predicted y values. *)
  -> [ `sum | `mean ]
  -> 'a Node.t

val leaky_relu : ([< `double | `float ] as 'a) Node.t -> alpha:float -> 'a Node.t
