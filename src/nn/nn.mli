open Core_kernel.Std

module Input_name : sig
  type 'a t
  val to_placeholder : 'a t -> 'a Ops.Placeholder.t
end

type _1d
type _2d
type _3d

module Shape : sig
  type 'a t =
    | D1 : int -> _1d t
    | D2 : int * int -> _2d t
    | D3 : int * int * int -> _3d t

  val dim_list : 'a t -> int list
end

type ('a, 'b) t
type init = [ `const of float | `normal of float | `truncated_normal of float ]

val shape : ('a, 'b) t -> 'a Shape.t
val default_input : ('a, 'b) t -> 'b Input_name.t option
val node : ('a, 'b) t -> 'b Node.t
val type_ : ('a, 'b) t -> 'b Node.Type.t

val input
  :  shape:'a Shape.t
  -> type_:'b Node.Type.t
  -> ('a, 'b) t

val named_input
  :  shape:'a Shape.t
  -> type_:'b Node.Type.t
  -> 'b Input_name.t * ('a, 'b) t

val dense
  :  ?w_init:init
  -> ?b_init:init
  -> (_1d, ([< `double | `float ] as 'a)) t
  -> shape:int
  -> (_1d, 'a) t

val conv2d
  :  ?w_init:init
  -> ?b_init:init
  -> (_3d, ([< `double | `float ] as 'a)) t
  -> filter:int*int
  -> out_channels:int
  -> strides:int*int
  -> padding:[ `same | `valid ]
  -> (_3d, 'a) t

val sigmoid : ('a, ([< `double | `float ] as 'b)) t -> ('a, 'b) t

val tanh : ('a, ([< `double | `float ] as 'b)) t -> ('a, 'b) t

val relu : ('a, ([< `double | `float ] as 'b)) t -> ('a, 'b) t

val softmax : ('a, ([< `double | `float ] as 'b)) t -> ('a, 'b) t

val max_pool
  :  (_3d, ([ `float ] as 'a)) t
  -> filter:int*int
  -> strides:int*int
  -> padding:[ `same | `valid ]
  -> (_3d, 'a) t

val concat : (_1d, 'a) t -> (_1d, 'a) t -> (_1d, 'a) t

val ( + ) : ('a, ([< `double | `float ] as 'b)) t -> ('a, 'b) t -> ('a, 'b) t

val ( - ) : ('a, ([< `double | `float ] as 'b)) t -> ('a, 'b) t -> ('a, 'b) t

val ( * ) : ('a, ([< `double | `float ] as 'b)) t -> ('a, 'b) t -> ('a, 'b) t

val f : float -> shape:'a Shape.t -> ('a, [ `float ]) t

val reshape
  : ('a, 'c) t
  -> shape:'b Shape.t
  -> ('b, 'c) t

val flatten : ('a, 'b) t -> (_1d, 'b) t

val split : (_2d, 'a) t -> (_1d, 'a) t list

val concatN : (_1d, 'a) t list -> (_2d, 'a) t

module Shared_var : sig
  (* Allows to build variables of type 'a with the shape without knowing
     where it is going to be applied yet.
     It needs to be applied only to input of the same Shape *)
  val with_shape
    :  f : (shape:'dim Shape.t -> type_:'c Node.Type.t -> 'a)
    -> ((('dim, 'c) t -> 'a) -> 'b)
    -> 'b Staged.t

  val dense
    :  ?w_init:init
    -> ?b_init:init
    -> shape:int
    -> unit
    -> ((_1d, ([< `double | `float ] as 'a)) t -> (_1d, 'a) t) Staged.t

  val conv2d
    :  ?w_init:init
    -> ?b_init:init
    -> filter:int*int
    -> out_channels:int
    -> strides:int*int
    -> padding:[ `same | `valid ]
    -> unit
    -> ((_3d, ([< `double | `float ] as 'a)) t -> (_3d, 'a) t) Staged.t
end
