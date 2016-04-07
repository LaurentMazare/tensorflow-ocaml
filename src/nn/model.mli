type t

val create : Nn._1d Nn.t -> t

val evaluate
  :  ?named_inputs:(Nn.Input_name.t * (float, Bigarray.float32_elt) Tensor.t) list
  -> t
  -> (float, Bigarray.float32_elt) Tensor.t
  -> (float, Bigarray.float32_elt) Tensor.t

module Optimizer : sig
  type t
  val gradient_descent : alpha:float -> t
  val momentum : alpha:float -> momentum:float -> t
  val adam : alpha:float -> ?beta1:float -> ?beta2:float -> ?epsilon:float -> unit -> t
end

module Loss : sig
  type t
  val cross_entropy : t
  val l2_mean : t
end

val fit
  :  ?named_inputs: (Nn.Input_name.t * (float, Bigarray.float32_elt) Tensor.t) list
  -> loss:Loss.t
  -> optimizer:Optimizer.t
  -> epochs:int
  -> xs:(float, Bigarray.float32_elt) Tensor.t
  -> ys:(float, Bigarray.float32_elt) Tensor.t
  -> t
  -> unit
