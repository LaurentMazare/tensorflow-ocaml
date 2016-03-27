let init_table = Node.Weak_table.create ()

let float shape ~init =
  let node = Ops_m.varf shape in
  let assign = Node.P (Ops.assign node init) in
  Node.Weak_table.set init_table ~key:(Node.P node) ~data:assign;
  node

let f shape x = float shape ~init:(Ops_m.f ~shape x)

let normalf shape ~stddev =
  let init =
    Ops.randomStandardNormal (Ops_m.const_int ~type_:Int32 shape) ~type_:Float
    |> Ops.mul (Ops_m.f stddev)
  in
  float shape ~init

let double shape ~init =
  let node = Ops_m.vard shape in
  let assign = Node.P (Ops.assign node init) in
  Node.Weak_table.set init_table ~key:(Node.P node) ~data:assign;
  node

let d shape x = double shape ~init:(Ops_m.d ~shape x)

let normald shape ~stddev =
  let init =
    Ops.randomStandardNormal (Ops_m.const_int ~type_:Int32 shape) ~type_:Double
    |> Ops.mul (Ops_m.d stddev)
  in
  double shape ~init

let get_init p =
  Node.Weak_table.find init_table p
