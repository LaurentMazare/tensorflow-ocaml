open Core.Std
module N = Node
module T = Node.Type

type unary =
  { f1 : 'a .
          (  [< `float | `double] as 'a) Node.t
          -> 'a Node.t
  }

let all = List.map ~f:Option.some

let pointwise_unary_exn (type a) ~self ~(gradient : a Node.t) ~t =
  let Node.P input =
    match self.Node.inputs with
    | [] | _ :: _ :: _ -> failwith "Not a unary function"
    | [ node ] -> node
  in
  let output =
    match input.output_type, gradient.Node.output_type with
    | Node.Type.Double, Node.Type.Double ->
      Node.P (Ops.mul gradient (t.f1 input))
    | Node.Type.Float, Node.Type.Float ->
      Node.P (Ops.mul gradient (t.f1 input))
    | _ -> failwith "Inconsistent types"
  in
  all [ output ]

let binary_extract_exn : type a . a Node.t -> (a Node.t * a Node.t) = fun node ->
  let Node.P lhs, Node.P rhs =
    match node.inputs with
    | [ lhs; rhs ] -> lhs, rhs
    | _ -> failwith "Not a binary function"
  in
  match lhs.output_type, rhs.output_type, node.output_type with
  | T.Double, T.Double, T.Double -> lhs, rhs
  | T.Float, T.Float, T.Float -> lhs, rhs
  | _ -> failwith "Inconsistent types"

let add_gradient ~self:_ ~gradient =
  let gradient = Node.P gradient in
  all [ gradient; gradient ]

let sub_gradient ~self:_ ~gradient =
  let minus_gradient = Node.P (Ops.neg gradient) in
  let gradient = Node.P gradient in
  all [ gradient; minus_gradient ]

let abs_gradient (type a) ~self ~(gradient : a Node.t) =
  let t = { f1 = fun input -> Ops.sign input } in
  pointwise_unary_exn ~self ~gradient ~t

let square_gradient (type a) ~self ~(gradient : a Node.t) =
  let t =
    { f1 = fun input -> Ops.mul input (Ops_m.scalar ~type_:input.output_type 2.) }
  in
  pointwise_unary_exn ~self ~gradient ~t

let matmul_gradient ~self ~gradient =
  let get_transpose str =
    List.Assoc.find self.Node.attributes str
    |> Option.value_map
        ~default:false
        ~f:(function
          | Node.Bool b -> b
          | _ -> assert false)
  in
  let transpose_a = get_transpose "transpose_a" in
  let transpose_b = get_transpose "transpose_b" in
  let lhs, rhs = binary_extract_exn self in
  match transpose_a, transpose_b with
  | false, false ->
    [ N.P (Ops.matMul gradient rhs ~transpose_b:true)
    ; N.P (Ops.matMul lhs gradient ~transpose_a:true)
    ] |> all
  | false, true ->
    [ N.P (Ops.matMul gradient rhs)
    ; N.P (Ops.matMul gradient lhs ~transpose_a:true)
    ] |> all
  | true, false ->
    [ N.P (Ops.matMul rhs gradient ~transpose_b:true)
    ; N.P (Ops.matMul lhs gradient)
    ] |> all
  | true, true ->
    [ N.P (Ops.matMul rhs gradient ~transpose_a:true ~transpose_b:true)
    ; N.P (Ops.matMul gradient lhs ~transpose_a:true ~transpose_b:true)
    ] |> all

let register_all () =
  List.iter ~f:(fun (name, f) ->
    Registered_gradients.add (Node.Op_name.of_string name) f)
    [ "Add", { Registered_gradients.f = add_gradient }
    ; "Sub", { f = sub_gradient }
    ; "Abs", { f = abs_gradient }
    ; "Square", { f = square_gradient }
    ; "MatMul", { f = matmul_gradient }
    ]
