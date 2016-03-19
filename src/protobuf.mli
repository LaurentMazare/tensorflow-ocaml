type t
val of_string : string -> t
val to_string : t -> string

val of_node : 'a Node.t -> t
val of_nodes : Node.p list -> t
