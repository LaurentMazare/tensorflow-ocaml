open Core_kernel.Std
module N = Node
module T = Node.Type

type unary =
  { f1 : 'a .x:(  [< `float | `double] as 'a) N.t
          -> y:'a N.t
          -> gradient:'a N.t
          -> 'a N.t
  }

let all = List.map ~f:Option.some

let pointwise_unary_exn (type a) ~self ~(gradient : a N.t) ~t =
  let N.P x =
    match self.N.inputs with
    | [] | _ :: _ :: _ -> failwith "Not a unary function"
    | [ node ] -> node
  in
  let output =
    match x.output_type, gradient.N.output_type with
    | T.Double, T.Double ->
      N.P (t.f1 ~x ~y:self ~gradient)
    | T.Float, T.Float ->
      N.P (t.f1 ~x ~y:self ~gradient)
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

let add_gradient_ ~self ~gradient =
  let slhs, srhs =
    match self.N.inputs with
    | [ N.P lhs; N.P rhs ] -> Ops.shape lhs, Ops.shape rhs
    | _ -> failwith "Not a binary function"
  in
  let rlhs, rrhs = Ops_m.broadcast_gradient_args slhs srhs in
  let lhs = Ops.reshape (Ops.sum gradient rlhs) slhs in
  let rhs = Ops.reshape (Ops.sum gradient rrhs) srhs in
  lhs, rhs

let add_gradient ~self ~gradient =
  let lhs, rhs = add_gradient_ ~self ~gradient in
  all [ N.P lhs; N.P rhs ]

let sub_gradient ~self ~gradient =
  let lhs, rhs = add_gradient_ ~self ~gradient in
  all [ N.P lhs; N.P (Ops.neg rhs) ]

let mul_gradient ~self ~gradient =
  let lhs, rhs = binary_extract_exn self in
  let shape_lhs = Ops.shape lhs in
  let shape_rhs = Ops.shape rhs in
  let rlhs, rrhs = Ops_m.broadcast_gradient_args shape_lhs shape_rhs in
  let lhs_gradient = Ops.reshape (Ops.sum Ops_m.(gradient * rhs) rlhs) shape_lhs in
  let rhs_gradient = Ops.reshape (Ops.sum Ops_m.(lhs * gradient) rrhs) shape_rhs in
  all [ N.P lhs_gradient; N.P rhs_gradient ]

let div_gradient ~self ~gradient =
  let lhs, rhs = binary_extract_exn self in
  let shape_lhs = Ops.shape lhs in
  let shape_rhs = Ops.shape rhs in
  let rlhs, rrhs = Ops_m.broadcast_gradient_args shape_lhs shape_rhs in
  let lhs_gradient = Ops.reshape (Ops.sum Ops_m.(gradient / rhs) rlhs) shape_lhs in
  let rhs_gradient =
    Ops.reshape (Ops.sum Ops_m.(gradient * (Ops.neg (lhs / Ops.square rhs))) rrhs) shape_rhs
  in
  all [ N.P lhs_gradient; N.P rhs_gradient ]

let pow_gradient ~self ~gradient =
  let lhs, rhs = binary_extract_exn self in
  let shape_lhs = Ops.shape lhs in
  let shape_rhs = Ops.shape rhs in
  let rlhs, rrhs = Ops_m.broadcast_gradient_args shape_lhs shape_rhs in
  let one = Ops_m.const_float ~type_:self.N.output_type [ 1. ] in
  let lhs_gradient =
    Ops.reshape (Ops.sum Ops_m.(gradient * rhs * Ops.pow lhs (rhs - one)) rlhs) shape_lhs
  in
  let rhs_gradient =
    Ops.reshape (Ops.sum Ops_m.(gradient * self * Ops.log lhs) rrhs) shape_rhs
  in
  all [ N.P lhs_gradient; N.P rhs_gradient ]

let neg_gradient ~self:_ ~gradient =
  all [ N.P (Ops.neg gradient) ]

let abs_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = fun ~x ~y:_ ~gradient -> Ops.sign x |> Ops.mul gradient } in
  pointwise_unary_exn ~self ~gradient ~t

let square_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 = fun ~x ~y:_ ~gradient ->
        Ops.mul x (Ops_m.scalar ~type_:x.output_type 2.)
        |> Ops.mul gradient
    }
  in
  pointwise_unary_exn ~self ~gradient ~t

let log_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = fun ~x ~y:_ ~gradient -> Ops.mul gradient (Ops.inv x) } in
  pointwise_unary_exn ~self ~gradient ~t

let relu_gradient ~self ~gradient =
  all [ N.P (Ops.reluGrad gradient self) ]

let sigmoid_gradient ~self ~gradient =
  let one = Ops_m.const_float ~type_:self.N.output_type [ 1. ] in
  all [ N.P Ops_m.(gradient * self * (one - self)) ]

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

let minmax_gradient ~self ~gradient =
  let input =
    match self.N.inputs with
    | [ input; _ ] -> Option.value_exn (N.extract input self.output_type)
    | _ -> failwith "Not a binary function"
  in
  let new_output_shape, _ = reduce_gradient ~self in
  let self = Ops.reshape self new_output_shape in
  let gradient = Ops.reshape gradient new_output_shape in
  let gradient =
    Ops.cast (Ops.equal self input) ~type_:self.N.output_type
    |> Ops.mul gradient
  in
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

let exp_gradient ~self ~gradient =
  all [ N.P (Ops.mul gradient self) ]

let sqrt_gradient ~self ~gradient =
  let gradient =
    Ops_m.(gradient * const_float ~type_:self.N.output_type [ 0.5 ] * Ops.inv self)
  in
  all [ N.P gradient ]

let tanh_gradient ~self ~gradient =
  let gradient =
    Ops_m.(gradient * (const_float ~type_:self.N.output_type [ 1. ] - Ops.square self))
  in
  all [ N.P gradient ]

let sign_gradient ~self ~gradient:_ =
  all [ N.P (Ops.zerosLike self) ]

let sin_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = fun ~x ~y:_ ~gradient -> Ops.mul gradient (Ops.cos x) } in
  pointwise_unary_exn ~self ~gradient ~t

let cos_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = fun ~x ~y:_ ~gradient -> Ops.mul gradient (Ops.sin x) |> Ops.neg } in
  pointwise_unary_exn ~self ~gradient ~t

let addn_gradient ~self ~gradient =
  List.map self.N.inputs ~f:(fun _ -> Some (N.P gradient))

let inv_gradient ~self ~gradient =
  [ Some (N.P (Ops.mul gradient (Ops.neg (Ops.square self)))) ]

let rsqrt_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 = fun ~x ~y ~gradient ->
        Ops_m.(gradient * const_float ~type_:y.N.output_type [ -0.5 ] * Ops.inv x * y)
    }
  in
  pointwise_unary_exn ~self ~gradient ~t

let two_over_pi = 2. /. 3.1415926535897932384626434

let erf_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 = fun ~x ~y ~gradient ->
        Ops_m.(gradient * const_float ~type_:y.N.output_type [ two_over_pi ]
          * Ops.exp (Ops.neg (Ops.square x)))
    }
  in
  pointwise_unary_exn ~self ~gradient ~t

let erfc_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 = fun ~x ~y ~gradient ->
        Ops_m.(gradient * const_float ~type_:y.N.output_type [ -. two_over_pi ]
          * Ops.exp (Ops.neg (Ops.square x)))
    }
  in
  pointwise_unary_exn ~self ~gradient ~t

let lgamma_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 = fun ~x ~y:_ ~gradient -> Ops.mul gradient (Ops.digamma x) }
  in
  pointwise_unary_exn ~self ~gradient ~t

let register_all () =
  let module O = Ops.Op_names in
  List.iter ~f:(fun (name, f) -> Registered_gradients.add name f)
    [ O.abs,     { Registered_gradients.f = abs_gradient }
    ; O.add,     { f = add_gradient }
    ; O.addN,    { f = addn_gradient }
    ; O.cos,     { f = cos_gradient }
    ; O.div,     { f = div_gradient }
    ; O.erf,     { f = erf_gradient }
    ; O.erfc,    { f = erfc_gradient }
    ; O.exp,     { f = exp_gradient }
    ; O.inv,     { f = inv_gradient }
    ; O.lgamma,  { f = lgamma_gradient }
    ; O.log,     { f = log_gradient }
    ; O.matMul,  { f = matmul_gradient }
    ; O.max,     { f = minmax_gradient }
    ; O.mean,    { f = mean_gradient }
    ; O.min,     { f = minmax_gradient }
    ; O.mul,     { f = mul_gradient }
    ; O.neg,     { f = neg_gradient }
    ; O.pow,     { f = pow_gradient }
    ; O.relu,    { f = relu_gradient }
    ; O.rsqrt,   { f = rsqrt_gradient }
    ; O.sigmoid, { f = sigmoid_gradient }
    ; O.sign,    { f = sign_gradient }
    ; O.sin,     { f = sin_gradient }
    ; O.softmax, { f = softmax_gradient }
    ; O.sqrt,    { f = sqrt_gradient }
    ; O.square,  { f = square_gradient }
    ; O.sub,     { f = sub_gradient }
    ; O.sum,     { f = sum_gradient }
    ; O.tanh,    { f = tanh_gradient }
    ]
