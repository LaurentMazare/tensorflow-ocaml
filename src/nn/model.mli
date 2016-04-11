type 'a t

val create : (Nn._1d, 'a) Nn.t -> 'a t

val evaluate
  :  ?named_inputs:(([ `float ] as 'a) Nn.Input_name.t * (float, Bigarray.float32_elt) Tensor.t) list
  -> ?batch_size:int
  -> ?node:'a Node.t
  -> 'a t
  -> (float, Bigarray.float32_elt) Tensor.t
  -> (float, Bigarray.float32_elt) Tensor.t

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

val fit
  :  ?named_inputs:(([ `float ] as 'a) Nn.Input_name.t * (float, Bigarray.float32_elt) Tensor.t) list
  -> ?batch_size:int
  -> ?on_epoch:(int -> err:float -> loss:'a Node.t -> [ `print_err | `do_nothing ])
  -> loss:Loss.t
  -> optimizer:Optimizer.t
  -> epochs:int
  -> xs:(float, Bigarray.float32_elt) Tensor.t
  -> ys:(float, Bigarray.float32_elt) Tensor.t
  -> 'a t
  -> unit

val save
  :  'a t
  -> filename:string
  -> unit

val load
  :  'a t
  -> filename:string
  -> unit
