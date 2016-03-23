open Core.Std

type unary =
  { f1 : 'a .
          (  [< `float | `double] as 'a) Node.t
          -> gradient:'a Node.t
          -> 'a Node.t
  }

let all = List.map ~f:Option.some

let unary_exn (type a) ~self ~(gradient : a Node.t) ~t =
  let Node.P input =
    match self.Node.inputs with
    | [] | _ :: _ :: _ -> failwith "Not a unary function"
    | [ node ] -> node
  in
  let output =
    match input.output_type, gradient.Node.output_type with
    | Node.Type.Double, Node.Type.Double -> Node.P (t.f1 input ~gradient)
    | Node.Type.Float, Node.Type.Float -> Node.P (t.f1 input ~gradient)
    | _ -> failwith "Inconsistent types"
  in
  all [ output ]

let add_gradient ~self:_ ~gradient =
  let gradient = Node.P gradient in
  all [ gradient; gradient ]

let sub_gradient ~self:_ ~gradient =
  let minus_gradient = Node.P (Ops.neg gradient) in
  let gradient = Node.P gradient in
  all [ gradient; minus_gradient ]

let abs_gradient (type a) ~self ~(gradient : a Node.t) =
  let t = { f1 = fun input ~gradient -> Ops.mul gradient (Ops.sign input) } in
  unary_exn ~self ~gradient ~t

let register_all () =
  Gradients.register_gradient (Node.Op_name.of_string "Add") { f = add_gradient };
  Gradients.register_gradient (Node.Op_name.of_string "Sub") { f = sub_gradient };
  Gradients.register_gradient (Node.Op_name.of_string "Abs") { f = abs_gradient }

