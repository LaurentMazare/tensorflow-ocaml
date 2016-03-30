
val gradient_descent_minimizer
  :  alpha:[ `float ] Node.t
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list

val momentum_minimizer
  :  alpha:[ `float ] Node.t
  -> momentum:[ `float ] Node.t
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list

val adam_minimizer
  :  alpha:[ `float ] Node.t
  -> ?beta1:[ `float ] Node.t
  -> ?beta2:[ `float ] Node.t
  -> ?epsilon:[ `float ] Node.t
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list

val adagrad_minimizer
  :  alpha:[ `float ] Node.t
  -> ?init:[ `float ] Node.t
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list

val rmsprop_minimizer
  :  alpha:[ `float ] Node.t
  -> ?decay:[ `float ] Node.t
  -> ?momentum:[ `float ] Node.t
  -> ?epsilon:[ `float ] Node.t
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list
