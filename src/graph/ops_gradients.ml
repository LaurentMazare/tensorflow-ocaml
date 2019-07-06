open Base
open Float.O_dot
module O = Tensorflow_core.Operation
module N = Node
module T = Node.Type

type unary =
  { f1 : 'a. x:([< `float | `double ] as 'a) N.t -> y:'a N.t -> gradient:'a N.t -> 'a N.t
  }

let all = List.map ~f:Option.some

let unary_wrapper_exn (type a) ~self ~(gradient : a N.t) ~t =
  let (N.P x) =
    match N.inputs self with
    | [] | [ `multi _ ] | _ :: _ :: _ -> failwith "Not a unary function"
    | [ `single node ] -> node
  in
  let output =
    match N.output_type x, N.output_type gradient with
    | T.Double, T.Double -> N.P (t.f1 ~x ~y:self ~gradient)
    | T.Float, T.Float -> N.P (t.f1 ~x ~y:self ~gradient)
    | _ -> failwith "Inconsistent types"
  in
  all [ output ]

let binary_extract_exn : type a. a N.t -> a N.t * a N.t =
 fun node ->
  let N.P lhs, N.P rhs =
    match N.inputs node with
    | [ `single lhs; `single rhs ] -> lhs, rhs
    | _ -> failwith "Not a binary function"
  in
  match N.output_type lhs, N.output_type rhs, N.output_type node with
  | T.Double, T.Double, T.Double -> lhs, rhs
  | T.Float, T.Float, T.Float -> lhs, rhs
  | _ -> failwith "Inconsistent types"

let add_gradient_ ~self ~gradient =
  let slhs, srhs =
    match N.inputs self with
    | [ `single (N.P lhs); `single (N.P rhs) ] -> Ops.shape32 lhs, Ops.shape32 rhs
    | _ -> failwith "Not a binary function"
  in
  let rlhs, rrhs = Ops.broadcastGradientArgs slhs srhs in
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
  let shape_lhs = Ops.shape32 lhs in
  let shape_rhs = Ops.shape32 rhs in
  let rlhs, rrhs = Ops.broadcastGradientArgs shape_lhs shape_rhs in
  let lhs_gradient = Ops.reshape (Ops.sum Ops.(gradient * rhs) rlhs) shape_lhs in
  let rhs_gradient = Ops.reshape (Ops.sum Ops.(lhs * gradient) rrhs) shape_rhs in
  all [ N.P lhs_gradient; N.P rhs_gradient ]

let div_gradient ~self ~gradient =
  let lhs, rhs = binary_extract_exn self in
  let shape_lhs = Ops.shape32 lhs in
  let shape_rhs = Ops.shape32 rhs in
  let rlhs, rrhs = Ops.broadcastGradientArgs shape_lhs shape_rhs in
  let lhs_gradient = Ops.reshape (Ops.sum Ops.(gradient / rhs) rlhs) shape_lhs in
  let rhs_gradient =
    Ops.reshape (Ops.sum Ops.(gradient * Ops.neg (lhs / Ops.square rhs)) rrhs) shape_rhs
  in
  all [ N.P lhs_gradient; N.P rhs_gradient ]

let pow_gradient ~self ~gradient =
  let lhs, rhs = binary_extract_exn self in
  let shape_lhs = Ops.shape32 lhs in
  let shape_rhs = Ops.shape32 rhs in
  let rlhs, rrhs = Ops.broadcastGradientArgs shape_lhs shape_rhs in
  let one = Ops.const_float ~type_:(N.output_type self) [ 1. ] in
  let lhs_gradient =
    Ops.reshape (Ops.sum Ops.(gradient * rhs * Ops.pow lhs (rhs - one)) rlhs) shape_lhs
  in
  let rhs_gradient =
    Ops.reshape (Ops.sum Ops.(gradient * self * Ops.log lhs) rrhs) shape_rhs
  in
  all [ N.P lhs_gradient; N.P rhs_gradient ]

let neg_gradient ~self:_ ~gradient = all [ N.P (Ops.neg gradient) ]

let abs_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = (fun ~x ~y:_ ~gradient -> Ops.sign x |> Ops.mul gradient) } in
  unary_wrapper_exn ~self ~gradient ~t

let square_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 =
        (fun ~x ~y:_ ~gradient ->
          Ops.mul x (Ops.scalar ~type_:(N.output_type x) 2.) |> Ops.mul gradient)
    }
  in
  unary_wrapper_exn ~self ~gradient ~t

let log_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = (fun ~x ~y:_ ~gradient -> Ops.mul gradient (Ops.reciprocal x)) } in
  unary_wrapper_exn ~self ~gradient ~t

let relu_gradient ~self ~gradient = all [ N.P (Ops.reluGrad gradient self) ]

let sigmoid_gradient ~self ~gradient =
  let one = Ops.const_float ~type_:(N.output_type self) [ 1. ] in
  all [ N.P Ops.(gradient * self * (one - self)) ]

let matmul_gradient ~self ~gradient =
  let get_transpose str =
    List.Assoc.find ~equal:String.equal (N.attributes self) str
    |> Option.value_map ~default:false ~f:(function
           | O.Bool b -> b
           | _ -> assert false)
  in
  let transpose_a = get_transpose "transpose_a" in
  let transpose_b = get_transpose "transpose_b" in
  let lhs, rhs = binary_extract_exn self in
  match transpose_a, transpose_b with
  | false, false ->
    [ N.P (Ops.matMul gradient rhs ~transpose_b:true)
    ; N.P (Ops.matMul lhs gradient ~transpose_a:true)
    ]
    |> all
  | false, true ->
    [ N.P (Ops.matMul gradient rhs); N.P (Ops.matMul gradient lhs ~transpose_a:true) ]
    |> all
  | true, false ->
    [ N.P (Ops.matMul rhs gradient ~transpose_b:true); N.P (Ops.matMul lhs gradient) ]
    |> all
  | true, true ->
    [ N.P (Ops.matMul rhs gradient ~transpose_a:true ~transpose_b:true)
    ; N.P (Ops.matMul gradient lhs ~transpose_a:true ~transpose_b:true)
    ]
    |> all

let batch_matmul_gradient ~self ~gradient =
  let get_adj str =
    List.Assoc.find ~equal:String.equal (N.attributes self) str
    |> Option.value_map ~default:false ~f:(function
           | O.Bool b -> b
           | _ -> assert false)
  in
  let adj_x = get_adj "adj_x" in
  let adj_y = get_adj "adj_y" in
  let lhs, rhs = binary_extract_exn self in
  match adj_x, adj_y with
  | false, false ->
    [ N.P (Ops.batchMatMul gradient rhs ~adj_x:false ~adj_y:true)
    ; N.P (Ops.batchMatMul lhs gradient ~adj_x:true ~adj_y:false)
    ]
    |> all
  | false, true ->
    [ N.P (Ops.batchMatMul gradient rhs ~adj_x:false ~adj_y:false)
    ; N.P (Ops.batchMatMul lhs gradient ~adj_x:true ~adj_y:false)
    ]
    |> all
  | true, false ->
    [ N.P (Ops.batchMatMul gradient rhs ~adj_x:false ~adj_y:true)
    ; N.P (Ops.batchMatMul lhs gradient ~adj_x:false ~adj_y:false)
    ]
    |> all
  | true, true ->
    [ N.P (Ops.batchMatMul gradient rhs ~adj_x:true ~adj_y:true)
    ; N.P (Ops.batchMatMul lhs gradient ~adj_x:true ~adj_y:true)
    ]
    |> all

(* Direct adaptation of math_grad.py from the tensorflow source code. *)
let reduce_gradient ~self =
  let N.P input, N.P indices =
    match N.inputs self with
    | [ `single input; `single indices ] -> input, indices
    | _ -> failwith "Incorrect number of inputs"
  in
  let input_shape = Ops.shape32 input in
  let input_rank = Ops.rank input in
  let indices_shape = Ops.shape32 indices in
  let indices = N.extract (N.P indices) Int32 in
  let new_output_shape =
    Ops.dynamicStitch
      [ Ops.range input_rank; Option.value_exn indices ]
      [ input_shape; Ops.fill indices_shape Ops.one32 ]
  in
  new_output_shape, input_shape

let sum_gradient_ ~self ~gradient =
  let new_output_shape, input_shape = reduce_gradient ~self in
  let tile_scaling = Ops.div input_shape new_output_shape in
  Ops.tile (Ops.reshape gradient new_output_shape) tile_scaling

let sum_gradient ~self ~gradient = [ Some (N.P (sum_gradient_ ~self ~gradient)); None ]

let mean_gradient ~self ~gradient =
  let sum_gradient = sum_gradient_ ~self ~gradient in
  let (N.P input) = List.hd_exn (N.flat_inputs self) in
  let input_shape = Ops.shape32 input in
  let output_shape = Ops.shape32 self in
  let factor = Ops.div (Ops.reduce_prod input_shape) (Ops.reduce_prod output_shape) in
  let gradient =
    Ops.div sum_gradient (Ops.cast factor ~type_:(N.output_type sum_gradient))
  in
  [ Some (N.P gradient); None ]

let minmax_gradient ~self ~gradient =
  let input =
    match N.flat_inputs self with
    | [ input; _ ] -> Option.value_exn (N.extract input (N.output_type self))
    | _ -> failwith "Not a binary function"
  in
  let new_output_shape, _ = reduce_gradient ~self in
  let self = Ops.reshape self new_output_shape in
  let gradient = Ops.reshape gradient new_output_shape in
  let gradient =
    Ops.cast (Ops.equal self input) ~type_:(N.output_type self) |> Ops.mul gradient
  in
  [ Some (N.P gradient); None ]

let minimum_maximum_gradient ~self ~gradient =
  match N.flat_inputs self with
  | [ input1; input2 ] ->
    let gradient_if_equal input =
      Option.value_exn (N.extract input (N.output_type self))
      |> Ops.equal self
      |> Ops.cast ~type_:(N.output_type self)
      |> Ops.mul gradient
      |> fun n -> Some (N.P n)
    in
    [ gradient_if_equal input1; gradient_if_equal input2 ]
  | _ -> failwith "Not a binary function"

let softmax_gradient ~self ~gradient =
  let gradient =
    Ops.(
      (gradient
      - Ops.reshape
          (reduce_sum ~dims:[ 1 ] (gradient * self))
          (const_int ~type_:Int32 [ -1; 1 ]))
      * self)
  in
  all [ N.P gradient ]

let exp_gradient ~self ~gradient = all [ N.P (Ops.mul gradient self) ]

let sqrt_gradient ~self ~gradient =
  let gradient =
    Ops.(
      gradient * const_float ~type_:(N.output_type self) [ 0.5 ] * Ops.reciprocal self)
  in
  all [ N.P gradient ]

let tanh_gradient ~self ~gradient =
  let gradient =
    Ops.(gradient * (const_float ~type_:(N.output_type self) [ 1. ] - Ops.square self))
  in
  all [ N.P gradient ]

let sign_gradient ~self ~gradient:_ = all [ N.P (Ops.zerosLike self) ]

let sin_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = (fun ~x ~y:_ ~gradient -> Ops.mul gradient (Ops.cos x)) } in
  unary_wrapper_exn ~self ~gradient ~t

let cos_gradient (type a) ~self ~(gradient : a N.t) =
  let t = { f1 = (fun ~x ~y:_ ~gradient -> Ops.mul gradient (Ops.sin x) |> Ops.neg) } in
  unary_wrapper_exn ~self ~gradient ~t

let addn_gradient ~self ~gradient =
  List.map (N.inputs self) ~f:(fun _ -> Some (N.P gradient))

let inv_gradient ~self ~gradient =
  [ Some (N.P (Ops.mul gradient (Ops.neg (Ops.square self)))) ]

let rsqrt_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 =
        (fun ~x ~y ~gradient ->
          Ops.(
            gradient
            * const_float ~type_:(N.output_type y) [ -0.5 ]
            * Ops.reciprocal x
            * y))
    }
  in
  unary_wrapper_exn ~self ~gradient ~t

let two_over_pi = 2. /. 3.1415926535897932384626434

let erf_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 =
        (fun ~x ~y ~gradient ->
          Ops.(
            gradient
            * const_float ~type_:(N.output_type y) [ two_over_pi ]
            * Ops.exp (Ops.neg (Ops.square x))))
    }
  in
  unary_wrapper_exn ~self ~gradient ~t

let erfc_gradient (type a) ~self ~(gradient : a N.t) =
  let t =
    { f1 =
        (fun ~x ~y ~gradient ->
          Ops.(
            gradient
            * const_float ~type_:(N.output_type y) [ -.two_over_pi ]
            * Ops.exp (Ops.neg (Ops.square x))))
    }
  in
  unary_wrapper_exn ~self ~gradient ~t

let conv2d_gradient : type a. self:a N.t -> gradient:a N.t -> N.p option list =
 fun ~self ~gradient ->
  match N.output_type self, N.output_type gradient with
  | N.Type.Float, N.Type.Float ->
    let inputs0, inputs1 = binary_extract_exn self in
    let strides = Option.value_exn (N.get_attr_int_list self "strides") in
    let use_cudnn_on_gpu = N.get_attr_bool self "use_cudnn_on_gpu" in
    let padding = Option.value_exn (N.get_attr_string self "padding") in
    let gradient_input =
      Ops.conv2DBackpropInput
        (Ops.shape32 inputs0)
        inputs1
        gradient
        ~strides
        ?use_cudnn_on_gpu
        ~padding
    in
    let gradient_filter =
      Ops.conv2DBackpropFilter
        inputs0
        (Ops.shape32 inputs1)
        gradient
        ~strides
        ?use_cudnn_on_gpu
        ~padding
    in
    all [ N.P gradient_input; N.P gradient_filter ]
  | _, _ -> failwith "Inconsistent types"

let conv2dbackpropinput_gradient
    : type a. self:a N.t -> gradient:a N.t -> N.p option list
  =
 fun ~self ~gradient ->
  match N.output_type self, N.output_type gradient with
  | N.Type.Float, N.Type.Float ->
    let N.P filter, N.P input =
      match N.inputs self with
      | [ _; `single filter; `single input ] -> filter, input
      | _ -> failwith "Not a ternary function"
    in
    (match N.output_type filter, N.output_type input with
    | T.Float, T.Float ->
      let strides = Option.value_exn (N.get_attr_int_list self "strides") in
      let use_cudnn_on_gpu = N.get_attr_bool self "use_cudnn_on_gpu" in
      let padding = Option.value_exn (N.get_attr_string self "padding") in
      let gradient_filter =
        Ops.conv2DBackpropFilter
          gradient
          (Ops.shape32 filter)
          input
          ~strides
          ?use_cudnn_on_gpu
          ~padding
      in
      let gradient_input =
        Ops.conv2D gradient filter ~strides ?use_cudnn_on_gpu ~padding
      in
      [ None; Some (N.P gradient_filter); Some (N.P gradient_input) ]
    | _, _ -> failwith "Expected float inputs")
  | _, _ -> failwith "Inconsistent types"

let avgpool_gradient : type a. self:a N.t -> gradient:a N.t -> N.p option list =
 fun ~self ~gradient ->
  match N.output_type self, N.output_type gradient with
  | N.Type.Float, N.Type.Float ->
    let ksize = Option.value_exn (N.get_attr_int_list self "ksize") in
    let strides = Option.value_exn (N.get_attr_int_list self "strides") in
    let padding = Option.value_exn (N.get_attr_string self "padding") in
    let input_shape =
      match N.inputs self with
      | [] | _ :: _ :: _ | [ `multi _ ] -> failwith "Not a unary function"
      | [ `single (N.P input) ] -> Ops.shape32 input
    in
    let gradient = Ops.avgPoolGrad input_shape gradient ~ksize ~strides ~padding in
    all [ N.P gradient ]
  | _, _ -> failwith "Inconsistent types"

let maxpool_gradient : type a. self:a N.t -> gradient:a N.t -> N.p option list =
 fun ~self ~gradient ->
  match N.output_type self, N.output_type gradient with
  | N.Type.Float, N.Type.Float ->
    let ksize = Option.value_exn (N.get_attr_int_list self "ksize") in
    let strides = Option.value_exn (N.get_attr_int_list self "strides") in
    let padding = Option.value_exn (N.get_attr_string self "padding") in
    let input =
      match N.inputs self with
      | [] | _ :: _ :: _ | [ `multi _ ] -> failwith "Not a unary function"
      | [ `single (N.P input) ] ->
        (match N.output_type input with
        | T.Float -> (input : [ `float ] N.t)
        | _ -> failwith "Inconsistent types")
    in
    let gradient = Ops.maxPoolGrad input self gradient ~ksize ~strides ~padding in
    all [ N.P gradient ]
  | _, _ -> failwith "Inconsistent types"

let reshape_gradient ~self ~gradient =
  let input_shape =
    match N.inputs self with
    | [ `single (N.P input); _ ] -> Ops.shape32 input
    | _ -> failwith "Not a binary function"
  in
  [ Some (N.P (Ops.reshape gradient input_shape)); None ]

let none ~self ~gradient:_ = List.map (N.inputs self) ~f:(fun _ -> None)
let fill_gradient ~self:_ ~gradient = [ None; Some (N.P (Ops.reduce_sum gradient)) ]

let concat_gradient ~self ~gradient =
  match N.inputs self with
  | [ `single _; `multi [ _ ] ] -> [ None; Some (N.P gradient) ]
  | [ `single concat_dim; `multi concat_inputs ] ->
    let concat_dim =
      match N.extract concat_dim Int32 with
      | None -> failwith "The first parameter of concat should have type int32."
      | Some concat_dim -> concat_dim
    in
    let concat_inputs =
      List.map concat_inputs ~f:(fun concat_input ->
          match N.extract concat_input (N.output_type self) with
          | None -> failwith "All the concatenated inputs should have the same type."
          | Some concat_input -> concat_input)
    in
    let sizes = Ops.shapeN concat_inputs ~type_:Int32 in
    let offsets = Ops.concatOffset concat_dim sizes in
    let concat_grads =
      List.map2_exn sizes offsets ~f:(fun size offset ->
          N.P (Ops.slice gradient offset size))
    in
    None :: all concat_grads
  | _ -> failwith "Concat nodes have multiple inputs."

let split_gradient ~self ~gradients =
  match N.inputs self with
  | [ `single split_dim; _ ] ->
    let num_split = Option.value_exn (N.get_attr_int self "num_split") in
    let all_gradients =
      List.init num_split ~f:(fun output_idx ->
          match Map.find gradients output_idx with
          | Some gradient -> gradient
          | None -> Ops.zerosLike (N.set_output_idx self (Some output_idx)))
    in
    let split_dim =
      match N.extract split_dim Int32 with
      | None -> failwith "The first parameter of split should have type int32."
      | Some split_dim -> split_dim
    in
    [ None; Some (N.P (Ops.concat split_dim all_gradients)) ]
  | _ -> failwith "split must have two arguments"

let select_gradient ~self ~gradient =
  let (N.P cond) =
    match N.inputs self with
    | [ `single cond; `single _lhs; `single _rhs ] -> cond
    | _ -> failwith "Unexpected arity for select"
  in
  match N.output_type cond with
  | T.Bool ->
    let zeros = Ops.zerosLike gradient in
    let lhs_gradient = Ops.select cond gradient zeros in
    let rhs_gradient = Ops.select cond zeros gradient in
    [ None; Some (N.P lhs_gradient); Some (N.P rhs_gradient) ]
  | _ -> failwith "Unexpected type for condition of select"

let identity_gradient ~self:_ ~gradient = [ Some (N.P gradient) ]

let merge_gradient ~self ~gradients =
  match N.inputs self with
  | [ `multi inputs ] ->
    let output_index = N.set_output_idx_and_output_type self (Some 1) ~type_:Int32 in
    let gradient = Map.find_exn gradients 0 in
    List.mapi inputs ~f:(fun index _input ->
        let equal =
          Ops.const_int ~shape:[] ~type_:Int32 [ index ] |> Ops.equal output_index
        in
        let _, gradient = Ops.switch gradient equal in
        Some (N.P gradient))
  | _ -> failwith "merge should have a multi inputs"

let pad_gradient ~self ~gradient =
  match N.inputs self with
  | [ `single (N.P x); `single padding ] ->
    let padding = N.extract_exn padding Int32 in
    let pad_before =
      Ops.slice
        padding
        (Ops.const_int ~shape:[ 2 ] ~type_:Int32 [ 0; 0 ])
        (Ops.pack [ Ops.rank x; Ops.one32 ])
    in
    let gradient =
      Ops.slice
        gradient
        (Ops.reshape pad_before (Ops.const_int ~shape:[ 1 ] ~type_:Int32 [ -1 ]))
        (Ops.shape32 x)
    in
    [ Some (N.P gradient); None ]
  | _ -> failwith "pad should have two single inputs"

let slice_gradient ~self ~gradient =
  match N.inputs self with
  | [ `single (N.P x); `single start_idxs; `single sizes ] ->
    let start_idxs = N.extract_exn start_idxs Int32 in
    let sizes = N.extract_exn sizes Int32 in
    let shape_ = Ops.pack [ Ops.rank x; Ops.one32 ] in
    let before_pad = Ops.reshape start_idxs shape_ in
    let after_pad = Ops.(reshape (shape ~type_:Int32 x - sizes - start_idxs) shape_) in
    let paddings = Ops.concat Ops.one32 [ before_pad; after_pad ] in
    let gradient = Ops.pad gradient paddings in
    [ Some (N.P gradient); None; None ]
  | _ -> failwith "slice should have three single inputs"

let transpose_gradient ~self ~gradient =
  let gradients ~indices =
    [ Some (N.P (Ops.transpose gradient (Ops.invertPermutation indices))); None ]
  in
  let indices =
    match N.inputs self with
    | [ `single _; `single indices ] -> indices
    | _ -> failwith "Incorrect number of inputs"
  in
  match N.extract indices Int32 with
  | Some indices -> gradients ~indices
  | None ->
    (match N.extract indices Int64 with
    | Some indices -> gradients ~indices
    | None -> failwith "Improper type for indexes in transpose")

let register_all () =
  let module O = Ops.Op_names in
  List.iter
    ~f:(fun (name, f) -> Registered_gradients.add name f)
    [ O.abs, { Registered_gradients.f = abs_gradient }
    ; O.add, { f = add_gradient }
    ; O.addN, { f = addn_gradient }
    ; O.avgPool, { f = avgpool_gradient }
    ; O.batchMatMul, { f = batch_matmul_gradient }
    ; O.concat, { f = concat_gradient }
    ; O.cos, { f = cos_gradient }
    ; O.conv2D, { f = conv2d_gradient }
    ; O.conv2DBackpropInput, { f = conv2dbackpropinput_gradient }
    ; O.div, { f = div_gradient }
    ; O.erf, { f = erf_gradient }
    ; O.erfc, { f = erfc_gradient }
    ; O.exp, { f = exp_gradient }
    ; O.fill, { f = fill_gradient }
    ; O.floor, { f = none }
    ; O.identity, { f = identity_gradient }
    ; O.inv, { f = inv_gradient }
    ; O.log, { f = log_gradient }
    ; O.matMul, { f = matmul_gradient }
    ; O.max, { f = minmax_gradient }
    ; O.maxPool, { f = maxpool_gradient }
    ; O.maximum, { f = minimum_maximum_gradient }
    ; O.mean, { f = mean_gradient }
    ; O.min, { f = minmax_gradient }
    ; O.minimum, { f = minimum_maximum_gradient }
    ; O.mul, { f = mul_gradient }
    ; O.neg, { f = neg_gradient }
    ; O.pad, { f = pad_gradient }
    ; O.pow, { f = pow_gradient }
    ; O.reciprocal, { f = inv_gradient }
    ; O.relu, { f = relu_gradient }
    ; O.reshape, { f = reshape_gradient }
    ; O.rsqrt, { f = rsqrt_gradient }
    ; O.select, { f = select_gradient }
    ; O.sigmoid, { f = sigmoid_gradient }
    ; O.sign, { f = sign_gradient }
    ; O.sin, { f = sin_gradient }
    ; O.slice, { f = slice_gradient }
    ; O.softmax, { f = softmax_gradient }
    ; O.sqrt, { f = sqrt_gradient }
    ; O.square, { f = square_gradient }
    ; O.sub, { f = sub_gradient }
    ; O.sum, { f = sum_gradient }
    ; O.tanh, { f = tanh_gradient }
    ; O.transpose, { f = transpose_gradient }
    ];
  List.iter
    ~f:(fun (name, g) -> Registered_gradients.add_multi name g)
    [ O.split, { Registered_gradients.g = split_gradient }
    ; O.merge, { Registered_gradients.g = merge_gradient }
    ]
