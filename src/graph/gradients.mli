val register_gradient
  :  string
  -> (self:Node.p -> gradient:Node.p -> Node.p option list)
  -> unit

val gradient
  :  [< `double | `float ] Node.t
  -> with_respect_to:Node.p list
  -> Node.p Node.Name.Table.t
