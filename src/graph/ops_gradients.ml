open Core.Std

let all = List.map ~f:Option.some

let add_gradient ~self:_ ~gradient =
  let gradient = Node.P gradient in
  all [ gradient; gradient ]

let sub_gradient ~self:_ ~gradient =
  let minus_gradient = Node.P (Ops.neg gradient) in
  let gradient = Node.P gradient in
  all [ gradient; minus_gradient ]

let abs_gradient (type a) ~self ~(gradient : a Node.t) =
  let Node.P input = List.hd_exn self.Node.inputs in
  let output =
    match input.output_type, gradient.Node.output_type with
    | Node.Type.Double, Node.Type.Double -> Node.P (Ops.mul gradient (Ops.sign input))
    | Node.Type.Float, Node.Type.Float -> Node.P (Ops.mul gradient (Ops.sign input))
    | _ -> assert false
  in
  all [ output ]

let register_all () =
  Gradients.register_gradient (Node.Op_name.of_string "Add") { f = add_gradient };
  Gradients.register_gradient (Node.Op_name.of_string "Sub") { f = sub_gradient };
  Gradients.register_gradient (Node.Op_name.of_string "Abs") { f = abs_gradient }

