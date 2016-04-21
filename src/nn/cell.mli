open Core_kernel.Std

val lstm
  :  size_c:int
  -> size_x:int
  -> (  h: [ `float ] Node.t
     -> x: [ `float ] Node.t
     -> c: [ `float ] Node.t
     -> [ `h of [ `float ] Node.t ] * [ `c of [ `float ] Node.t ]) Staged.t
