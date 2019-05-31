open Base

type t =
  { f :
      'a. self:([< `float | `double ] as 'a) Node.t -> gradient:'a Node.t
      -> Node.p option list
  }

val add : Node.Op_name.t -> t -> unit

val find
  :  Node.Op_name.t
  -> (self:Node.p -> gradient:Node.p -> Node.p option list) option

type multi =
  { g :
      'a. self:([< `float | `double ] as 'a) Node.t -> gradients:'a Node.t Map.M(Int).t
      -> Node.p option list
  }

val add_multi : Node.Op_name.t -> multi -> unit

val find_multi
  :  Node.Op_name.t
  -> (self:Node.p -> gradients:Node.p Map.M(Int).t -> Node.p option list) option
