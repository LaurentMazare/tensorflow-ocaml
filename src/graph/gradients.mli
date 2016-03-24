
val gradient
  :  [< `double | `float ] Node.t
  -> with_respect_to_float:[ `float ] Node.t list
  -> with_respect_to_double:[ `double ] Node.t list
  -> ([ `float ] Node.t list) * ([ `double ] Node.t list)
