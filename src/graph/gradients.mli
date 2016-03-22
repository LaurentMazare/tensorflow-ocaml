
type t =
  { f : 'a .
          (  self:([< `float | `double] as 'a) Node.t
          -> gradient:'a Node.t
          -> Node.p option list)
  }

val register_gradient
  :  string
  -> t
  -> unit

val gradient
  :  [< `double | `float ] Node.t
  -> with_respect_to:Node.p list
  -> Node.p Node.Name.Table.t
