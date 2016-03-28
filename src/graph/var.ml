open Core_kernel.Std

let init_table = Node.Weak_table.create ()

let float shape ~init =
  let node =
    Ops_generated.variable ()
      ~type_:Float
      ~shape:(List.map shape ~f:(fun size -> { Node.Dim.size; name = None }))
  in
  let assign = Node.P (Ops.assign node init) in
  Node.Weak_table.set init_table ~key:(Node.P node) ~data:assign;
  node

let f shape x = float shape ~init:(Ops.f ~shape x)

let normalf shape ~stddev =
  let init =
    Ops.randomStandardNormal (Ops.const_int ~type_:Int32 shape) ~type_:Float
    |> Ops.mul (Ops.f stddev)
  in
  float shape ~init

let double shape ~init =
  let node =
    Ops_generated.variable ()
      ~type_:Double
      ~shape:(List.map shape ~f:(fun size -> { Node.Dim.size; name = None }))
  in
  let assign = Node.P (Ops.assign node init) in
  Node.Weak_table.set init_table ~key:(Node.P node) ~data:assign;
  node

let d shape x = double shape ~init:(Ops.d ~shape x)

let normald shape ~stddev =
  let init =
    Ops.randomStandardNormal (Ops.const_int ~type_:Int32 shape) ~type_:Double
    |> Ops.mul (Ops.d stddev)
  in
  double shape ~init

let get_init p =
  Node.Weak_table.find init_table p
