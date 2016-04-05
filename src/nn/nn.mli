open Core_kernel.Std
type t

val input
  :  shape:int list
  -> [ `float ] Node.t * t

val dense
  (* TODO: add init *)
  :  t
  -> shape:int list
  -> t

val sigmoid : t -> t

val tanh : t -> t

val relu : t -> t

val concat : t -> t -> t

val ( + ) : t -> t -> t

val ( - ) : t -> t -> t

val ( * ) : t -> t -> t

val f : float -> t

module Shared_var :
sig

  (* Allows to build variables of type 'a with the shape without knowing
     where it is going to be applied yet.
     It needs to be applied only to input of the same Shape *)
  val with_shape
    :  f : (shape:int list -> 'a)
    -> ((t -> 'a) -> 'b)
    -> 'b Staged.t

  val dense :
    shape:int list
    -> (t -> t) Staged.t

end
