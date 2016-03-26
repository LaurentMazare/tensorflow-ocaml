let init_table = Node.Weak_table.create ()

let float shape ~init =
  let res = Ops_m.varf shape in
  Node.Weak_table.set init_table ~key:(Node.P res) ~data:(Node.P init);
  res

let double shape ~init =
  let res = Ops_m.vard shape in
  Node.Weak_table.set init_table ~key:(Node.P res) ~data:(Node.P init);
  res

let get_init p =
  Node.Weak_table.find init_table p
