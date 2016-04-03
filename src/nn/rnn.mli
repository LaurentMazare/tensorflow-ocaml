open Core_kernel.Std

val gru
  :  shape:int list
  -> (h:Nn.t -> x:Nn.t -> Nn.t) Staged.t
