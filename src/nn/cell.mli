open Core_kernel.Std

val lstm
  :  size_c:int
  -> size_x:int
  -> (  h: [ `float ] Node.t
     -> x: [ `float ] Node.t
     -> c: [ `float ] Node.t
     -> [ `h of [ `float ] Node.t ] * [ `c of [ `float ] Node.t ]) Staged.t

val lstm_d
  :  size_c:int
  -> size_x:int
  -> (  h: [ `double ] Node.t
     -> x: [ `double ] Node.t
     -> c: [ `double ] Node.t
     -> [ `h of [ `double ] Node.t ] * [ `c of [ `double ] Node.t ]) Staged.t

val gru
  :  size_h:int
  -> size_x:int
  -> (  h: [ `float ] Node.t
     -> x: [ `float ] Node.t
     -> [ `float ] Node.t) Staged.t

val gru_d
  :  size_h:int
  -> size_x:int
  -> (  h: [ `double ] Node.t
     -> x: [ `double ] Node.t
     -> [ `double ] Node.t) Staged.t
