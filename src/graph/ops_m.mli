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
  :  type_:([< `float | `double ] as 'dtype) Node.Type.t
  -> float
  -> 'dtype Node.t

type 't b =  ?name:string -> 't Node.t -> 't Node.t -> 't Node.t

val (+) : [< `float | `double | `int32 | `int64 | `complex64 | `string ] b
val (-) : [< `float | `double | `int32 | `int64 | `complex64 ] b
val (/) : [< `float | `double | `int32 | `int64 | `complex64 ] b
val ( * ) : [< `float | `double | `int32 | `int64 | `complex64 ] b
val ( *^) : [< `float | `double | `int32 | `complex64 ] b
val ( *.) : [< `float | `double | `int32 | `complex64 ] b

(* Scalar *)
val f : ?shape:int list -> float -> [ `float ] Node.t
val d : ?shape:int list -> float -> [ `double ] Node.t

(* Constant vector/matrixes *)
val cf : ?shape:int list -> float list -> [ `float ] Node.t
val cd : ?shape:int list -> float list -> [ `double ] Node.t

(* Variables *)
val varf : int list -> [ `float ] Node.t
val vard : int list -> [ `double ] Node.t

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

val one32 : [ `int32 ] Node.t
val zero32 : [ `int32 ] Node.t

val range : [ `int32 ] Node.t -> [ `int32 ] Node.t

val broadcast_gradient_args
  : [ `int32 ] Node.t -> [ `int32 ] Node.t -> [ `int32 ] Node.t * [ `int32 ] Node.t
