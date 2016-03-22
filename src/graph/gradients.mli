
type t =
  { f : 'a .
          (  self:([< `float | `double] as 'a) Node.t
          -> gradient:'a Node.t
          -> Node.p option list)
  }

val register_gradient
  :  Node.Op_name.t
  -> t
  -> unit

val gradient
  :  [< `double | `float ] Node.t
  -> with_respect_to_float:[ `float ] Node.t list
  -> with_respect_to_double:[ `double ] Node.t list
  -> ([ `float ] Node.t list) * ([ `double ] Node.t list)
