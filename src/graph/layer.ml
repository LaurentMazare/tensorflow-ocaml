open Base
open Float.O_dot

let batch_normalization ?(decay = 0.9) t ~update_moments ~dims ~feature_count =
  let type_ = Node.output_type t in
  let zero = Ops.const_float ~type_ (List.init feature_count ~f:(fun _ -> 0.)) in
  let one = Ops.const_float ~type_ (List.init feature_count ~f:(fun _ -> 1.)) in
  let one_minus_decay = Ops.scalar ~type_ (1. -. decay) in
  let beta = Var.create [ feature_count ] ~type_ ~init:zero in
  let gamma = Var.create [ feature_count ] ~type_ ~init:one in
  let batch_moments = Ops.moments t ~dims:(List.init dims ~f:Fn.id) in
  let beta_with_update ~control_inputs =
    (* EWMA update. *)
    Ops.assignSub
      beta
      Ops.(one_minus_decay * (beta - batch_moments.mean))
      ~control_inputs
  in
  let gamma_with_update ~control_inputs =
    (* EWMA update. *)
    Ops.assignSub
      gamma
      Ops.(one_minus_decay * (gamma - batch_moments.variance))
      ~control_inputs
  in
  let beta, gamma =
    match update_moments with
    | `always ->
      Ops.identity beta ~control_inputs:[ Node.P (beta_with_update ~control_inputs:[]) ],
      Ops.identity gamma ~control_inputs:[ Node.P (gamma_with_update ~control_inputs:[]) ]
    | `not_in_testing testing ->
      let beta ~control_inputs:_ = beta in
      let gamma ~control_inputs:_ = gamma in
      Ops.cond_with_control_inputs testing ~if_true:beta ~if_false:beta_with_update,
      Ops.cond_with_control_inputs testing ~if_true:gamma ~if_false:gamma_with_update
  in
  Ops.normalize t { mean = beta; variance = gamma }

type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu of float (* max xs (alpha * xs) *)
  | Sigmoid

type 'a linear =
  { output : 'a Node.t
  ; w : 'a Node.t
  ; b : 'a Node.t
  ; activation : activation option
  }

let linear_vars linear = [ linear.w; linear.b ]
let linear_output linear = linear.output

let linear_apply xs ~w ~b ~activation =
  let ys = Ops.(xs *^ w + b) in
  match activation with
  | Some Relu -> Ops.relu ys
  | Some Softmax -> Ops.softmax ys
  | Some Tanh -> Ops.tanh ys
  | Some Sigmoid -> Ops.sigmoid ys
  | Some (Leaky_relu alpha) ->
    let type_ = Node.output_type xs in
    Ops.(maximum ys (f_or_d ~type_ alpha * ys))
  | None -> ys

let linear_with_vars ?activation xs ~output_dim =
  let last_xs_dim = Node.shape xs |> List.last_exn in
  let type_ = Node.output_type xs in
  let w = Var.normal ~type_ [ last_xs_dim; output_dim ] ~stddev:0.1 in
  let b = Var.f_or_d ~type_ [ output_dim ] 0. in
  { output = linear_apply xs ~w ~b ~activation; w; b; activation }

let linear ?activation xs ~output_dim =
  (linear_with_vars ?activation xs ~output_dim).output

let linear_apply linear xs =
  linear_apply xs ~w:linear.w ~b:linear.b ~activation:linear.activation

type padding =
  | Same
  | Valid

let padding_string = function
  | Same -> "SAME"
  | Valid -> "VALID"

let max_pool ?(padding = Same) xs ~ksize ~strides =
  let k1, k2 = ksize in
  let s1, s2 = strides in
  Ops.maxPool xs
    ~ksize:[ 1; k1; k2; 1 ] ~strides:[ 1; s1; s2; 1 ] ~padding:(padding_string padding)

let conv2d ?(padding = Same) xs ~ksize ~strides ~output_dim =
  let last_xs_dim = Node.shape xs |> List.last_exn in
  let k1, k2 = ksize in
  let s1, s2 = strides in
  let type_ = Node.output_type xs in
  let w = Var.normal ~type_ [ k1; k2; last_xs_dim; output_dim ] ~stddev:0.1 in
  let b = Var.f_or_d ~type_ [ output_dim ] 0. in
  let conv2d = Ops.conv2D xs w ~strides:[ 1; s1; s2; 1 ] ~padding:(padding_string padding) in
  Ops.add conv2d b

let shape_to_string shape =
  List.map shape ~f:Int.to_string
  |> String.concat ~sep:", "
  |> Printf.sprintf "(%s)"

let reshape xs ~shape =
  Ops.reshape xs (Ops.const_int ~type_:Int32 shape)

let flatten xs =
  let shape = Node.shape xs in
  let total_dim =
    List.fold (List.tl_exn shape) ~init:1 ~f:(fun acc d ->
      if d <= 0
      then
        let msg =
          Printf.sprintf "cannot flatten %s shape %s"
            (Node.name xs |> Node.Name.to_string)
            (shape_to_string shape)
        in
        invalid_arg msg
      else d * acc)
  in
  reshape xs ~shape:[ -1; total_dim ]
