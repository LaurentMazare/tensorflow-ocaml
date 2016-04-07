open Core_kernel.Std

module Input_name : sig
  type t
  val to_node : t -> [ `float ] Node.t
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

type 'a t
type init = [ `const of float | `normal of float ]

val shape : 'a t -> 'a Shape.t
val default_input : 'a t -> Input_name.t option
val node : 'a t -> [ `float ] Node.t

val input
  :  shape:'a Shape.t
  -> 'a t

val named_input
  :  shape:'a Shape.t
  -> Input_name.t * 'a t

val dense
  :  ?w_init:init
  -> ?b_init:init
  -> _1d t
  -> shape:int
  -> _1d t

val conv2d
  :  ?w_init:init
  -> ?b_init:init
  -> _3d t
  -> filter:int*int
  -> out_channels:int
  -> strides:int*int
  -> padding:[ `same | `valid ]
  -> _3d t

val sigmoid : 'a t -> 'a t

val tanh : 'a t -> 'a t

val relu : 'a t -> 'a t

val softmax : 'a t -> 'a t

val max_pool
  :  _3d t
  -> filter:int*int
  -> strides:int*int
  -> padding:[ `same | `valid ]
  -> _3d t

val concat : _1d t -> _1d t -> _1d t

val ( + ) : 'a t -> 'a t -> 'a t

val ( - ) : 'a t -> 'a t -> 'a t

val ( * ) : 'a t -> 'a t -> 'a t

val f : float -> shape:'a Shape.t -> 'a t

val reshape
  : 'a t
  -> shape:'b Shape.t
  -> 'b t

val flatten : 'a t -> _1d t

val split : _2d t -> _1d t list

val concatN : _1d t list -> _2d t

module Shared_var : sig
  (* Allows to build variables of type 'a with the shape without knowing
     where it is going to be applied yet.
     It needs to be applied only to input of the same Shape *)
  val with_shape
    :  f : (shape:'dim Shape.t -> 'a)
    -> (('dim t -> 'a) -> 'b)
    -> 'b Staged.t

  val dense
    :  ?w_init:init
    -> ?b_init:init
    -> shape:int
    -> unit
    -> (_1d t -> _1d t) Staged.t

  val conv2d
    :  ?w_init:init
    -> ?b_init:init
    -> filter:int*int
    -> out_channels:int
    -> strides:int*int
    -> padding:[ `same | `valid ]
    -> unit
    -> (_3d t -> _3d t) Staged.t
end
