open Base
open Float.O_dot

(* TODO: add some trainable gamma/beta variables and return beta + gamma * current_output. *)
module Batch_norm = struct
  type 'a t =
    { running_mean : 'a Node.t
    ; running_var : 'a Node.t
    ; epsilon : float
    ; decay : float
    }

  let create ?(epsilon = 1e-8) ?(decay = 0.99) xs =
    let type_ = Node.output_type xs in
    let feature_count = Node.shape xs |> List.last_exn in
    let zeros = Ops.f_or_d ~shape:[ feature_count ] ~type_ 0. in
    let ones = Ops.f_or_d ~shape:[ feature_count ] ~type_ 1. in
    { running_mean = Var.create [ feature_count ] ~type_ ~init:zeros
    ; running_var = Var.create [ feature_count ] ~type_ ~init:ones
    ; epsilon
    ; decay
    }

  let apply_infer t xs =
    let type_ = Node.output_type xs in
    Ops.((xs - t.running_mean) * rsqrt (t.running_var + f_or_d ~type_ t.epsilon))

  let apply_train t xs =
    let type_ = Node.output_type xs in
    let nb_dims = Node.shape xs |> List.length in
    let batch_moments = Ops.moments xs ~dims:(List.init (nb_dims - 1) ~f:Fn.id) in
    let ys = Ops.normalize xs batch_moments ~epsilon:t.epsilon in
    let one_minus_decay = Ops.f_or_d ~type_ (1. -. t.decay) in
    let mean_update =
      Ops.assignSub
        t.running_mean
        Ops.(one_minus_decay * (t.running_mean - batch_moments.mean))
    in
    let var_update =
      Ops.assignSub
        t.running_var
        Ops.(one_minus_decay * (t.running_var - batch_moments.variance))
    in
    ys, `update_ops [ mean_update; var_update ]
end

module Update_ops_store = struct
  type t = Node.p list ref

  let create () = ref []
  let ops t = !t
end

let batch_norm ?(decay = 0.99) xs ~is_training ~update_ops_store =
  let type_ = Node.output_type xs in
  let bn = Batch_norm.create ~decay xs in
  let infer = Batch_norm.apply_infer bn xs in
  let train, `update_ops update_ops = Batch_norm.apply_train bn xs in
  update_ops_store := List.map update_ops ~f:(fun n -> Node.P n) @ !update_ops_store;
  let not_is_training = Ops.(cast (logicalNot is_training) ~type_) in
  let is_training = Ops.cast is_training ~type_ in
  Ops.((is_training * train) + (not_is_training * infer))

type activation =
  | Relu
  | Softmax
  | Tanh
  | Leaky_relu of float (* max xs (alpha * xs) *)
  | Sigmoid

module Linear = struct
  type 'a t =
    { output_dim : int
    ; mutable w : 'a Node.t option
    ; mutable b : 'a Node.t option
    }

  let vars { w; b; output_dim = _ } = [ Option.value_exn w; Option.value_exn b ]
  let create output_dim = { output_dim; w = None; b = None }

  let apply ?activation ?(use_bias = true) t xs =
    let last_xs_dim = Node.shape xs |> List.last_exn in
    let type_ = Node.output_type xs in
    let w =
      match t.w with
      | Some w -> w
      | None ->
        let w = Var.normal ~type_ [ last_xs_dim; t.output_dim ] ~stddev:0.1 in
        t.w <- Some w;
        w
    in
    let b =
      match t.b with
      | Some b -> b
      | None ->
        let b = Var.f_or_d ~type_ [ t.output_dim ] 0. in
        t.b <- Some b;
        b
    in
    let ys = if use_bias then Ops.((xs *^ w) + b) else Ops.(xs *^ w) in
    match activation with
    | Some Relu -> Ops.relu ys
    | Some Softmax -> Ops.softmax ys
    | Some Tanh -> Ops.tanh ys
    | Some Sigmoid -> Ops.sigmoid ys
    | Some (Leaky_relu alpha) -> Ops.leaky_relu ys ~alpha
    | None -> ys
end

let linear ?activation ?use_bias xs ~output_dim =
  let linear = Linear.create output_dim in
  Linear.apply ?activation ?use_bias linear xs

type padding =
  | Same
  | Valid

let padding_string = function
  | Same -> "SAME"
  | Valid -> "VALID"

let max_pool ?(padding = Same) xs ~ksize ~strides =
  let k1, k2 = ksize in
  let s1, s2 = strides in
  Ops.maxPool
    xs
    ~ksize:[ 1; k1; k2; 1 ]
    ~strides:[ 1; s1; s2; 1 ]
    ~padding:(padding_string padding)

module Conv2D = struct
  type 'a t =
    { output_dim : int
    ; mutable w : 'a Node.t option
    ; mutable b : 'a Node.t option
    ; ksize : int * int
    ; strides : int * int
    ; padding : padding
    }

  let vars { w; b; _ } = [ Option.value_exn w; Option.value_exn b ]

  let create ~ksize ~strides ~padding output_dim =
    { output_dim; w = None; b = None; ksize; strides; padding }

  let apply ?(use_bias = true) t xs =
    let last_xs_dim = Node.shape xs |> List.last_exn in
    let k1, k2 = t.ksize in
    let s1, s2 = t.strides in
    let type_ = Node.output_type xs in
    let w =
      match t.w with
      | Some w -> w
      | None ->
        let w = Var.normal ~type_ [ k1; k2; last_xs_dim; t.output_dim ] ~stddev:0.1 in
        t.w <- Some w;
        w
    in
    let b =
      match t.b with
      | Some b -> b
      | None ->
        let b = Var.f_or_d ~type_ [ t.output_dim ] 0. in
        t.b <- Some b;
        b
    in
    let conv2d =
      Ops.conv2D xs w ~strides:[ 1; s1; s2; 1 ] ~padding:(padding_string t.padding)
    in
    if use_bias then Ops.add conv2d b else conv2d
end

let conv2d ?(padding = Same) ?use_bias xs ~ksize ~strides ~output_dim =
  let conv2d = Conv2D.create ~padding ~ksize ~strides output_dim in
  Conv2D.apply ?use_bias conv2d xs

module Conv2DTranspose = struct
  type 'a t =
    { output_dim : int
    ; mutable w : 'a Node.t option
    ; mutable b : 'a Node.t option
    ; ksize : int * int
    ; strides : int * int
    ; padding : padding
    }

  let vars { w; b; _ } = [ Option.value_exn w; Option.value_exn b ]

  let create ~ksize ~strides ~padding output_dim =
    { output_dim; w = None; b = None; ksize; strides; padding }

  let apply ?(use_bias = true) t xs =
    let batch_dim, input_w, input_h, last_xs_dim =
      match Node.shape xs with
      | [ a; b; c; d ] -> a, b, c, d
      | _ -> failwith "unexpected shape for conv2d_transpose input"
    in
    let output_length input_l ~ksize ~stride =
      match t.padding with
      | Valid -> (input_l * stride) + max 0 (ksize - stride)
      | Same -> input_l * stride
    in
    let k1, k2 = t.ksize in
    let s1, s2 = t.strides in
    let output_w = output_length input_w ~ksize:k1 ~stride:s1 in
    let output_h = output_length input_h ~ksize:k2 ~stride:s2 in
    let type_ = Node.output_type xs in
    let w =
      match t.w with
      | Some w -> w
      | None ->
        let w = Var.normal [ k1; k2; t.output_dim; last_xs_dim ] ~type_ ~stddev:0.1 in
        t.w <- Some w;
        w
    in
    let b =
      match t.b with
      | Some b -> b
      | None ->
        let b = Var.f_or_d ~type_ [ t.output_dim ] 0. in
        t.b <- Some b;
        b
    in
    let conv2d_t =
      Ops.conv2DBackpropInput
        ~strides:[ 1; s1; s2; 1 ]
        ~padding:(padding_string t.padding)
        (Ops.ci32 ~shape:[ 4 ] [ batch_dim; output_w; output_h; t.output_dim ])
        w
        xs
    in
    if use_bias then Ops.add conv2d_t b else conv2d_t
end

let conv2d_transpose ?(padding = Same) ?use_bias xs ~ksize ~strides ~output_filters =
  let conv2d_t = Conv2DTranspose.create ~padding ~ksize ~strides output_filters in
  Conv2DTranspose.apply ?use_bias conv2d_t xs

let shape_to_string shape =
  List.map shape ~f:Int.to_string |> String.concat ~sep:", " |> Printf.sprintf "(%s)"

let reshape xs ~shape = Ops.reshape xs (Ops.const_int ~type_:Int32 shape)

let flatten xs =
  let shape = Node.shape xs in
  let total_dim =
    List.fold (List.tl_exn shape) ~init:1 ~f:(fun acc d ->
        if d <= 0
        then (
          let msg =
            Printf.sprintf
              "cannot flatten %s shape %s"
              (Node.name xs |> Node.Name.to_string)
              (shape_to_string shape)
          in
          invalid_arg msg)
        else d * acc)
  in
  reshape xs ~shape:[ -1; total_dim ]
