open Base
open Tensorflow_core
open! Tensorflow

type _1d
type _2d
type _3d

module Shape : sig
  type 'a t =
    | D1 : int -> _1d t
    | D2 : int * int -> _2d t
    | D3 : int * int * int -> _3d t
end

module Id : sig
  type t
end

module Input_id : sig
  type t
end

type 'a t
type init = [ `const of float | `normal of float | `truncated_normal of float ]

val shape : 'a t -> 'a Shape.t
val id : _ t -> Id.t

val input
  :  shape:'a Shape.t
  -> 'a t * Input_id.t

val const
  :  float
  -> shape:'a Shape.t
  -> 'a t

val sigmoid    : 'a t -> 'a t
val tanh       : 'a t -> 'a t
val relu       : 'a t -> 'a t
val softmax    : 'a t -> 'a t
val reduce_sum : 'a t -> 'a t
val square     : 'a t -> 'a t
val neg        : 'a t -> 'a t

val (+) : 'a t -> 'a t -> 'a t
val (-) : 'a t -> 'a t -> 'a t
val ( * ) : 'a t -> 'a t -> 'a t

val dense
  :  ?w_init:init
  -> ?b_init:init
  -> ?name:string
  -> int
  -> _1d t
  -> _1d t

val conv2d
  :  ?w_init:init
  -> ?b_init:init
  -> ?name:string
  -> filter:int*int
  -> out_channels:int
  -> strides:int*int
  -> padding:[ `same | `valid ]
  -> unit
  -> _3d t
  -> _3d t

val dense'
  :  ?w_init:init
  -> ?b_init:init
  -> ?name:string
  -> int
  -> (_1d t -> _1d t) Staged.t

val conv2d'
  :  ?w_init:init
  -> ?b_init:init
  -> ?name:string
  -> filter:int*int
  -> out_channels:int
  -> strides:int*int
  -> padding:[ `same | `valid ]
  -> unit
  -> (_3d t -> _3d t) Staged.t

val avg_pool
  :  _3d t
  -> filter:int*int
  -> strides:int*int
  -> padding:[ `same | `valid ]
  -> _3d t

val max_pool
  :  _3d t
  -> filter:int*int
  -> strides:int*int
  -> padding:[ `same | `valid ]
  -> _3d t

val reshape
  :  _ t
  -> shape:'a Shape.t
  -> 'a t

val flatten
  :  _ t
  -> _1d t

val split : _2d t -> _1d t list

val concat : _1d t list -> _2d t

val var : 'a t -> 'a t

module Optimizer : sig
  type t
  val gradient_descent : learning_rate:float -> t
  val momentum : learning_rate:float -> momentum:float -> t
  val adam : learning_rate:float -> ?beta1:float -> ?beta2:float -> ?epsilon:float -> unit -> t
end

module Loss : sig
  type t
  val cross_entropy : [ `sum | `mean ] -> t
  val l2 : [ `sum | `mean ] -> t
end

module Model : sig
  type 'a fnn = 'a t
  type ('a, 'b, 'c) t

  val create
    :  ('c * 'b) Tensor.eq
    -> 'a fnn
    -> ('a, 'b, 'c) t

  val predict
    :  ('a, 'b, 'c) t
    -> ?output_id:Id.t
    -> (Input_id.t * (float, 'c) Tensor.t) list
    -> (float, 'c) Tensor.t

  (** [fit ~xs ~ys ?batch_size ~epoch ...] trains a model using [xs] and [ys] as training data.
      Training will be done in batches of size [batch_size]. The total amount of training steps
      is [epochs * |xs| / batch_size]. If [batch_size] is not given or is larger than [|xs|], it defaults to [|xs|].
      The last [|xs| mod batch_size] training samples are not used for training.
      After each epoch the average loss for the epoch is printed to {!stdout}.
  *)
  val fit
    :  ?addn_inputs:(Input_id.t * (float, 'c) Tensor.t) list
    -> ?batch_size:int
    -> ?explicit_vars:('a fnn list)
    -> ('a, 'b, 'c) t
    -> loss:Loss.t
    -> optimizer:Optimizer.t
    -> epochs:int
    -> input_id:Input_id.t
    -> xs:(float, 'c) Tensor.t
    -> ys:(float, 'c) Tensor.t
    -> unit

  val save
    :  ?inputs:(Input_id.t * (float, 'c) Tensor.t) list
    -> ('a, 'b, 'c) t
    -> filename:string
    -> unit

  val load
    :  ?inputs:(Input_id.t * (float, 'c) Tensor.t) list
    -> ('a, 'b, 'c) t
    -> filename:string
    -> unit
end
