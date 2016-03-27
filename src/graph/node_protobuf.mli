val of_node : 'a Node.t -> Protobuf.t
val of_nodes : Node.p list -> Protobuf.t

val of_nodes'
  :  ?verbose:unit
  -> already_exported_nodes:Node.p Node.Id.Table.t
  -> Node.p list
  -> Protobuf.t
