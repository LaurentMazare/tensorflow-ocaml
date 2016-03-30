
val gradient_descent_minimizer
  :  alpha:[ `float ] Node.t
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list

val momentum_minimizer
  :  alpha:[ `float ] Node.t
  -> momentum:float
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list

val adam_minimizer
  :  alpha:[ `float ] Node.t
  -> ?beta1:float
  -> ?beta2:float
  -> ?epsilon:float
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list

val adagrad_minimizer
  :  alpha : [ `float ] Node.t
  -> ?init:float
  -> ?varsf:[ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd:[ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list
