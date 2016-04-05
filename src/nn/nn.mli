open Core_kernel.Std

module Input_name : sig
  type t
end

type _1d
type _2d
type _3d

type 'a shape =
  | D1 : int -> _1d shape
  | D2 : int * int -> _2d shape
  | D3 : int * int * int -> _3d shape

type 'a t

val shape : 'a t -> 'a shape

val input
  :  shape:'a shape
  -> 'a t

val named_input
  :  shape:'a shape
  -> Input_name.t * 'a t

val dense
  (* TODO: add init *)
  :  _1d t
  -> shape:int
  -> _1d t

val sigmoid : 'a t -> 'a t

val tanh : 'a t -> 'a t

val relu : 'a t -> 'a t

val softmax : 'a t -> 'a t

val concat : _1d t -> _1d t -> _1d t

val ( + ) : 'a t -> 'a t -> 'a t

val ( - ) : 'a t -> 'a t -> 'a t

val ( * ) : 'a t -> 'a t -> 'a t

val f : float -> shape:'a shape -> 'a t

module Shared_var : sig
  (* Allows to build variables of type 'a with the shape without knowing
     where it is going to be applied yet.
     It needs to be applied only to input of the same Shape *)
  val with_shape
    :  f : (shape:'dim shape -> 'a)
    -> (('dim t -> 'a) -> 'b)
    -> 'b Staged.t

  val dense
    :  shape:int
    -> (_1d t -> _1d t) Staged.t

end

module Model : sig
  type 'a net = 'a t
  type 'a t

  val create : 'a net -> 'a t

  val evaluate
    :  ?named_inputs:(Input_name.t * (float, Bigarray.float32_elt) Tensor.t) list
    -> 'a t
    -> (float, Bigarray.float32_elt) Tensor.t
    -> (float, Bigarray.float32_elt) Tensor.t

  type optimizer =
    | Gradient_descent of float

  type loss =
    | Cross_entropy

  val fit
    :  ?named_inputs: (Input_name.t * (float, Bigarray.float32_elt) Tensor.t) list
    -> loss:loss
    -> optimizer:optimizer
    -> epochs:int
    -> xs:(float, Bigarray.float32_elt) Tensor.t
    -> ys:(float, Bigarray.float32_elt) Tensor.t
    -> 'a t
    -> unit
end
