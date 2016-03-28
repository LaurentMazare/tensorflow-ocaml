
val gradient_descent_minimizer
  :  alpha : [ `float ] Node.t
  -> ?varsf : [ `float ] Node.t list (* Have to be variables. *)
  -> ?varsd : [ `double ] Node.t list (* Have to be variables. *)
  -> [< `float | `double ] Node.t
  -> Node.p list
