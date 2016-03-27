let init_table = Node.Weak_table.create ()

let float shape ~init =
  let node = Ops_m.varf shape in
  let assign = Node.P (Ops.assign node init) in
  Node.Weak_table.set init_table ~key:(Node.P node) ~data:assign;
  node

let f shape x = float shape ~init:(Ops_m.f ~shape x)

let double shape ~init =
  let node = Ops_m.vard shape in
  let assign = Node.P (Ops.assign node init) in
  Node.Weak_table.set init_table ~key:(Node.P node) ~data:assign;
  node

let d shape x = double shape ~init:(Ops_m.d ~shape x)

let get_init p =
  Node.Weak_table.find init_table p
