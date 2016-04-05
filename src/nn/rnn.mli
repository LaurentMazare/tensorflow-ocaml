open Core_kernel.Std

val gru
  :  shape:int
  -> (h:Nn._1d Nn.t -> x:Nn._1d Nn.t -> Nn._1d Nn.t) Staged.t
