val float : int list -> init:([ `float ] Node.t) -> [ `float ] Node.t
val double : int list -> init:([ `double ] Node.t) -> [ `double ] Node.t
val f : int list -> float -> [ `float ] Node.t
val d : int list -> float -> [ `double ] Node.t

val get_init : Node.p -> Node.p option
