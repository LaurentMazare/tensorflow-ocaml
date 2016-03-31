(* ==== Binary Operations ==== *)

(* The common type for binary operators. *)
type 't b =  ?name:string -> 't Node.t -> 't Node.t -> 't Node.t

(* Pointwise arithmetic operations. *)
val (+) : [< `float | `double | `int32 | `int64 | `complex64 | `string ] b
val (-) : [< `float | `double | `int32 | `int64 | `complex64 ] b
val (/) : [< `float | `double | `int32 | `int64 | `complex64 ] b
val ( * ) : [< `float | `double | `int32 | `int64 | `complex64 ] b

(* Matrix multiplication. *)
val ( *^) : [< `float | `double | `int32 | `complex64 ] b

(* ==== Constant Tensors ==== *)

(* Constant tensors using a single value. *)
val f : ?shape:int list -> float -> [ `float ] Node.t
val d : ?shape:int list -> float -> [ `double ] Node.t

(* Constant tensors using different values. *)
val cf : ?shape:int list -> float list -> [ `float ] Node.t
val cd : ?shape:int list -> float list -> [ `double ] Node.t

(* Some more refined constant creation functions. *)
val const_float
  :  ?name:string
  -> ?shape:int list
  -> type_:([< `float | `double ] as 'dtype) Node.Type.t
  -> float list
  -> 'dtype Node.t

val const_int
  :  ?name:string
  -> ?shape:int list
  -> type_:([< `int32 | `int64 ] as 'dtype) Node.Type.t
  -> int list
  -> 'dtype Node.t

val scalar
  :  ?empty_shape:unit
  -> type_:([< `float | `double ] as 'dtype) Node.Type.t
  -> float
  -> 'dtype Node.t

(* Useful int scalar values. *)
val one32 : [ `int32 ] Node.t
val zero32 : [ `int32 ] Node.t

(* ==== Reduction Functions ==== *)

(* Reduction functions, [dims] is the list of dimensions across which the reduction
   is performed. *)
type 'a reduce_fn
   =  ?dims:int list
  -> ([< `complex64 | `double | `float | `int32 | `int64 ] as 'a) Node.t
  -> 'a Node.t

val reduce_sum : 'a reduce_fn
val reduce_min : 'a reduce_fn
val reduce_max : 'a reduce_fn
val reduce_mean : 'a reduce_fn
val reduce_prod : 'a reduce_fn
val reduce_all : ?dims:int list -> [ `bool ] Node.t -> [ `bool ] Node.t
val reduce_any : ?dims:int list -> [ `bool ] Node.t -> [ `bool ] Node.t

(* ==== Misc ==== *)

(* [range n] where [n] is a scalar tensor returns tensor [ 0; 1; ...; n-1 ]. *)
val range : [ `int32 ] Node.t -> [ `int32 ] Node.t

(* A placeholder that can be bound to a tensor via [inputs] in [Session.run]. *)
val placeholder : ?name:string -> type_:'a Node.Type.t -> int list -> 'a Node.t

(* [dropout n ~keep_prob] returns a tensor with the same shape as [n] that either
   have the same value as [n] with prob [keep_prob] or else is [0].
   The resulting values are then scaled by [1/keep_prob]. *)
val dropout
  :  ([< `float | `double ] as 'a) Node.t
  -> keep_prob:'a Node.t
  -> 'a Node.t
