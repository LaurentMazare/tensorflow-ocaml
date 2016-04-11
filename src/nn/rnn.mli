open Core_kernel.Std

val gru
  :  shape:int
  -> (h:(Nn._1d, [ `float ]) Nn.t -> x:(Nn._1d, [ `float ]) Nn.t -> (Nn._1d, [ `float ]) Nn.t) Staged.t

val fold
  :  (Nn._2d, [ `float ]) Nn.t
  -> init:'a
  -> f:('a -> (Nn._1d, [ `float ]) Nn.t -> 'a)
  -> 'a

val scan
  :  (Nn._2d, [ `float ]) Nn.t
  -> init:(Nn._1d, [ `float ]) Nn.t
  -> f:((Nn._1d, [ `float ]) Nn.t -> (Nn._1d, [ `float ]) Nn.t -> (Nn._1d, [ `float ]) Nn.t)
  -> (Nn._2d, [ `float ]) Nn.t * (Nn._1d, [ `float ]) Nn.t

val scan'
  :  (Nn._2d, [ `float ]) Nn.t
  -> init:'a
  -> f:('a -> (Nn._1d, [ `float ]) Nn.t -> (Nn._1d, [ `float ]) Nn.t * 'a)
  -> (Nn._2d, [ `float ]) Nn.t * 'a
