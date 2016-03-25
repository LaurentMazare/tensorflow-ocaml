open Core_kernel.Std
module N = Node
module T = Node.Type

type unary =
  { f1 : 'a .input:(  [< `float | `double] as 'a) N.t
          -> gradient:'a N.t
          -> 'a N.t
  }

let all = List.map ~f:Option.some

let pointwise_unary_exn (type a) ~self ~(gradient : a N.t) ~t =
  let N.P input =
    match self.N.inputs with
    | [] | _ :: _ :: _ -> failwith "Not a unary function"
    | [ node ] -> node
  in
  let output =
    match input.output_type, gradient.N.output_type with
    | T.Double, T.Double ->
      N.P (t.f1 ~input ~gradient)
    | T.Float, T.Float ->
      N.P (t.f1 ~input ~gradient)
    | _ -> failwith "Inconsistent types"
  in
  all [ output ]

let binary_extract_exn : type a . a N.t -> (a N.t * a N.t) = fun node ->
  let N.P lhs, N.P rhs =
    match node.inputs with
    | [ lhs; rhs ] -> lhs, rhs
    | _ -> failwith "Not a binary function"
  in
  match lhs.output_type, rhs.output_type, node.output_type with
  | T.Double, T.Double, T.Double -> lhs, rhs
  | T.Float, T.Float, T.Float -> lhs, rhs
  | _ -> failwith "Inconsistent types"

let add_gradient ~self:_ ~gradient =
  let gradient = N.P gradient in
  all [ gradient; gradient ]

let sub_gradient ~self:_ ~gradient =
  let minus_gradient = N.P (Ops.neg gradient) in
  let gradient = N.P gradient in
  all [ gradient; minus_gradient ]

let abs_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = fun ~input ~gradient -> Ops.sign input |> Ops.mul gradient } in
  pointwise_unary_exn ~self ~gradient ~t

let square_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 = fun ~input ~gradient ->
        Ops.mul input (Ops_m.scalar ~type_:input.output_type 2.)
        |> Ops.mul gradient
    }
  in
  pointwise_unary_exn ~self ~gradient ~t

let log_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = fun ~input ~gradient -> Ops.mul gradient (Ops.inv input) } in
  pointwise_unary_exn ~self ~gradient ~t

let relu_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = fun ~input ~gradient -> Ops.reluGrad gradient input } in
  pointwise_unary_exn ~self ~gradient ~t

let matmul_gradient ~self ~gradient =
  let get_transpose str =
    List.Assoc.find self.N.attributes str
    |> Option.value_map
        ~default:false
        ~f:(function
          | N.Bool b -> b
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

(* Direct adaptation of math_grad.py from the tensorflow source code. *)
let reduce_gradient ~self =
  let N.P input, N.P indices =
    match self.N.inputs with
    | [ input; indices ] -> input, indices
    | _ -> failwith "Incorrect number of inputs"
  in
  let input_shape = Ops.shape input in
  let input_rank = Ops.rank input in
  let indices_shape = Ops.shape indices in
  let indices =
    N.extract (N.P indices) Int32
  in
  let new_output_shape =
    Ops.dynamicStitch
      [ Ops_m.range input_rank; Option.value_exn indices ]
      [ input_shape; Ops.fill indices_shape Ops_m.one32 ]
  in
  new_output_shape, input_shape

let sum_gradient_ ~self ~gradient =
  let new_output_shape, input_shape = reduce_gradient ~self in
  let tile_scaling = Ops.div input_shape new_output_shape in
  Ops.tile (Ops.reshape gradient new_output_shape) tile_scaling

let sum_gradient ~self ~gradient =
  [ Some (N.P (sum_gradient_ ~self ~gradient)); None ]

let mean_gradient ~self ~gradient =
  let sum_gradient = sum_gradient_ ~self ~gradient in
  let N.P input = List.hd_exn self.inputs in
  let input_shape = Ops.shape input in
  let output_shape = Ops.shape self in
  let factor = Ops.div (Ops_m.reduce_prod input_shape) (Ops_m.reduce_prod output_shape) in
  let gradient = Ops.div sum_gradient (Ops.cast factor ~type_:sum_gradient.output_type) in
  [ Some (N.P gradient); None ]

let softmax_gradient ~self ~gradient =
  let gradient =
    Ops_m.(
      (gradient
        - Ops.reshape
            (reduce_sum ~dims:[ 1 ] (gradient * self))
            (const_int ~type_:Int32 [ -1; 1 ])
      ) * self)
  in
  all [ N.P gradient ]

let register_all () =
  List.iter ~f:(fun (name, f) ->
    Registered_gradients.add (N.Op_name.of_string name) f)
    [ "Abs",     { Registered_gradients.f = abs_gradient }
    ; "Add",     { f = add_gradient }
    ; "Log",     { f = log_gradient }
    ; "MatMul",  { f = matmul_gradient }
    ; "Mean",    { f = mean_gradient }
    ; "Relu",    { f = relu_gradient }
    ; "Softmax", { f = softmax_gradient }
    ; "Square",  { f = square_gradient }
    ; "Sub",     { f = sub_gradient }
    ; "Sum",     { f = sum_gradient }
    ]
