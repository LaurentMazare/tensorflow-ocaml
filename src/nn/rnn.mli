open Core_kernel.Std

val gru
  :  shape:int
  -> (h:Nn._1d Nn.t -> x:Nn._1d Nn.t -> Nn._1d Nn.t) Staged.t

val fold
  :  Nn._2d Nn.t
  -> init:'a
  -> f:('a -> Nn._1d Nn.t -> 'a)
  -> 'a

val scan
  :  Nn._2d Nn.t
  -> init:Nn._1d Nn.t
  -> f:(Nn._1d Nn.t -> Nn._1d Nn.t -> Nn._1d Nn.t)
  -> Nn._2d Nn.t * Nn._1d Nn.t

val scan'
  :  Nn._2d Nn.t
  -> init:'a
  -> f:('a -> Nn._1d Nn.t -> Nn._1d Nn.t * 'a)
  -> Nn._2d Nn.t * 'a
