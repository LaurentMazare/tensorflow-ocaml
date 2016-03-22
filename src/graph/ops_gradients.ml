open Core.Std

let all = List.map ~f:Option.some

let add_gradient ~self:_ ~gradient =
  let gradient = Node.P gradient in
  all [ gradient; gradient ]

let sub_gradient ~self:_ ~gradient =
  let minus_gradient = Node.P (Ops.neg gradient) in
  let gradient = Node.P gradient in
  all [ gradient; minus_gradient ]

let register_all () =
  Gradients.register_gradient "Add" { f = add_gradient };
  Gradients.register_gradient "Sub" { f = sub_gradient }

