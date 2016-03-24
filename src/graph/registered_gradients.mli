type t =
  { f : 'a .
          (  self:([< `float | `double] as 'a) Node.t
          -> gradient:'a Node.t
          -> Node.p option list)
  }

val add
  :  Node.Op_name.t
  -> t
  -> unit

val find
  :  Node.Op_name.t
  -> (self:Node.p -> gradient:Node.p -> Node.p option list) option
