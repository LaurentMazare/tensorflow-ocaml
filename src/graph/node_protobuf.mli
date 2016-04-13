val of_node
  :  ?verbose:unit
  -> 'a Node.t
  -> Protobuf.t option

val of_nodes
  :  ?verbose:unit
  -> Node.p list
  -> Protobuf.t option

val of_nodes'
  :  ?verbose:unit
  -> already_exported_nodes:Node.Id.Hash_set.t
  -> Node.p list
  -> Protobuf.t option
