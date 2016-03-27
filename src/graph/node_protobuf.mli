val of_node
  :  ?verbose:unit
  -> 'a Node.t
  -> Protobuf.t

val of_nodes
  :  ?verbose:unit
  -> Node.p list
  -> Protobuf.t

val of_nodes'
  :  ?verbose:unit
  -> already_exported_nodes:Node.p Node.Id.Table.t
  -> Node.p list
  -> Protobuf.t
