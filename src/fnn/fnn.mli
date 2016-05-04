open Core_kernel.Std

type _1d
type _2d
type _3d

module Shape : sig
  type 'a t =
    | D1 : int -> _1d t
    | D2 : int * int -> _2d t
    | D3 : int * int * int -> _3d t
end

module Input_id : sig
  type t
end

type 'a t
type init = [ `const of float | `normal of float | `truncated_normal of float ]

val shape : 'a t -> 'a Shape.t

val input
  :  shape:'a Shape.t
  -> 'a t * Input_id.t

val const
  :  float
  -> shape:'a Shape.t
  -> 'a t

val sigmoid : 'a t -> 'a t
val tanh : 'a t -> 'a t
val relu : 'a t -> 'a t
val softmax : 'a t -> 'a t

val (+) : 'a t -> 'a t -> 'a t
val (-) : 'a t -> 'a t -> 'a t
val ( * ) : 'a t -> 'a t -> 'a t

val dense
  :  ?w_init:init
  -> ?b_init:init
  -> int
  -> (_1d t -> _1d t) Staged.t

module Model : sig
  type 'a fnn = 'a t
  type ('a, 'b) t

  val create
    :  'a fnn
    -> ([ `float | `double ] as 'b) Node.Type.t
    -> ('a, 'b) t

  val save
    :  ('a, 'b) t
    -> filename:string
    -> unit

  val load
    :  ('a, 'b) t
    -> filename:string
    -> unit
end
