type 'a optimizer =
  learning_rate:[ `float ] Node.t
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> 'a Node.t
  -> Node.p list

val gradient_descent_minimizer : [< `float | `double ] optimizer
val momentum_minimizer : momentum:[ `float ] Node.t -> [< `float | `double ] optimizer

val adam_minimizer
  :  ?beta1:[ `float ] Node.t
  -> ?beta2:[ `float ] Node.t
  -> ?epsilon:[ `float ] Node.t
  -> [< `float | `double ] optimizer

val adagrad_minimizer : ?init:[ `float ] Node.t -> [< `float | `double ] optimizer

val rmsprop_minimizer
  :  ?decay:[ `float ] Node.t
  -> ?momentum:[ `float ] Node.t
  -> ?epsilon:[ `float ] Node.t
  -> [< `float | `double ] optimizer
