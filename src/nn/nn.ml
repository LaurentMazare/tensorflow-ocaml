open Core_kernel.Std
exception Shape_mismatch of int list * int list * string
let () =
  Caml.Printexc.register_printer (function
    | Shape_mismatch (dims, dims', str) ->
      let dims = List.map dims ~f:Int.to_string |> String.concat ~sep:", " in
      let dims' = List.map dims' ~f:Int.to_string |> String.concat ~sep:", " in
      Some (sprintf "Shape mismatch %s: %s <> %s" str dims dims')
    | _ -> None)

module Input_name = struct
  type t = [ `float ] Node.t

  let merge t_option t_option' =
    match t_option, t_option' with
    | None, None -> None
    | (Some _ as s), None | None, (Some _ as s) -> s
    | Some t as s, Some t' when Node.(Id.(=) (id t) (id t')) -> s
    | Some _, Some _ -> failwith "Different inputs"
end

(* TODO: handle double ? *)
type _1d
type _2d
type _3d

type 'a shape =
  | D1 : int -> _1d shape
  | D2 : int * int -> _2d shape
  | D3 : int * int * int -> _3d shape

let dim_list (type a) (shape : a shape) =
  match shape with
  | D1 d -> [ d ]
  | D2 (d, d') -> [ d; d' ]
  | D3 (d, d', d'') -> [ d; d'; d'' ]

type 'a t =
  { shape : 'a shape
  ; node : [ `float ] Node.t
  ; variables : [ `float ] Node.t list
  ; default_input : Input_name.t option
  }

let shape t = t.shape

let named_input ~shape =
  let placeholder = Ops.placeholder ~type_:Float (dim_list shape) in
  let t =
    { shape
    ; node = placeholder
    ; variables = []
    ; default_input = None
    }
  in
  placeholder, t

let input ~shape =
  let placeholder = Ops.placeholder ~type_:Float (dim_list shape) in
  { shape
  ; node = placeholder
  ; variables = []
  ; default_input = Some placeholder
  }

let shape_mismatch shape1 shape2 ~op_name =
  let shape1 = dim_list shape1 in
  let shape2 = dim_list shape2 in
  raise (Shape_mismatch (shape1, shape2, op_name))

let padding_str = function
  | `same -> "SAME"
  | `valid -> "VALID"

module Shared_var = struct
  let with_shape ~f g =
    let shape_a = ref (`F f) in
    let f t =
      let s = t.shape in
      match !shape_a with
      | `F f ->
        let a = f ~shape:s in
        shape_a := `Computed (s, a);
        a
      | `Computed (shape, a) ->
        if s <> shape
        then failwith "Dimensions do not match"
        else a
    in
    Staged.stage (g f)

  let dense ~shape =
    with_shape ~f:(fun ~shape:input_shape ->
      let input_shape =
        match input_shape with
        | D1 input_shape -> input_shape
      in
      let w = Var.f [ input_shape; shape ] 0. in
      let b = Var.f [ shape ] 0. in
      w, b)
      (fun f t ->
        let w, b = f t in
        let node = Ops.(t.node *^ w + b) in
        { shape = D1 shape
        ; node
        ; variables = [ w; b ]
        ; default_input = t.default_input
        })

  let conv2d ~filter ~out_channels ~strides ~padding =
    let filter_height, filter_width = filter in
    let stride_height, stride_width = strides in
    let strides = [ 1; stride_height; stride_width; 1 ] in
    with_shape ~f:(fun ~shape:input_shape ->
      let image_height, image_width, in_channels =
        match input_shape with
        | D3 (d1, d2, d3) -> d1, d2, d3
      in
      let w = Var.f [ filter_height; filter_width; in_channels; out_channels ] 0. in
      let b = Var.f [ out_channels ] 0. in
      image_height, image_width, w, b)
      (fun f t ->
        let input_height, input_width, w, b = f t in
        let input_height, input_width =
          match padding with
          | `same -> input_height, input_width
          | `valid -> input_height - filter_height + 1, input_width - filter_width + 1
        in
        let output_height, output_width =
          (input_height - 1) / stride_height + 1,
          (input_width - 1) / stride_width + 1
        in
        let padding = padding_str padding in
        let node = Ops.(conv2D ~strides ~padding t.node w + b) in
        { shape = D3 (output_height, output_width, out_channels)
        ; node
        ; variables = [ w; b ]
        ; default_input = t.default_input
        })
end

let f v ~shape =
  { node = Ops.f v ~shape:(dim_list shape)
  ; shape
  ; variables = []
  ; default_input = None
  }

let unary op t = { t with node = op t.node }

let sigmoid t = unary Ops.sigmoid t
let relu t = unary Ops.relu t
let tanh t = unary Ops.tanh t
let softmax t = unary Ops.softmax t

let tuple4_to_list (a, b, c, d) = [ a; b; c; d ]

let max_pool t ~ksize ~strides ~padding =
  unary
    (Ops.maxPool
      ~ksize:(tuple4_to_list ksize)
      ~strides:(tuple4_to_list strides)
      ~padding:(padding_str padding))
    t

let dense t ~shape =
  Staged.unstage (Shared_var.dense ~shape) t

let conv2d t ~filter ~out_channels ~strides ~padding =
  Staged.unstage
    (Shared_var.conv2d ~filter ~out_channels ~strides ~padding)
    t

let concat t1 t2 =
  let shape =
    match t1.shape, t2.shape with
    | D1 shape, D1 shape' -> D1 (shape + shape')
  in
  { variables = t1.variables @ t2.variables
  ; shape
  (* We use one32 as the concat dim as the batch-size dimension is 0. *)
  ; node = Ops.(concat one32 [ t1.node; t2.node ])
  ; default_input = Input_name.merge t1.default_input t2.default_input
  }

let binary ~op_name op t1 t2 =
  if t1.shape <> t2.shape
  then shape_mismatch t1.shape t2.shape ~op_name;
  { node = op t1.node t2.node
  ; shape = t1.shape
  ; variables = t1.variables @ t2.variables
  ; default_input = Input_name.merge t1.default_input t2.default_input
  }

let ( * ) t t' = binary ~op_name:"Mul" Ops.( * ) t t'

let (+) t t' = binary ~op_name:"Add" Ops.(+) t t'
let (-) t t' = binary ~op_name:"Add" Ops.(-) t t'

module Model = struct
  type 'a net = 'a t

  type 'a t =
    { session : Session.t
    ; net : 'a net
    ; placeholder : [ `float ] Node.t
    }

  (* TODO: stochastic gradient descent. *)
  module Optimizer = struct
    (* We should use some inline records here when they will be available. *)
    type t =
      | Gradient_descent of float
      | Adam of float * float option * float option * float option
      | Momentum of float * float

    let gradient_descent ~alpha = Gradient_descent alpha

    let adam ~alpha ?beta1 ?beta2 ?epsilon () =
      Adam (alpha, beta1, beta2, epsilon)

    let momentum ~alpha ~momentum =
      Momentum (alpha, momentum)

    let get t ~loss =
      match t with
      | Gradient_descent alpha ->
        Optimizers.gradient_descent_minimizer ~alpha:(Ops.f alpha) loss
      | Adam (alpha, beta1, beta2, epsilon) ->
        Optimizers.adam_minimizer loss
          ~alpha:(Ops.f alpha)
          ?beta1:(Option.map beta1 ~f:Ops.f)
          ?beta2:(Option.map beta2 ~f:Ops.f)
          ?epsilon:(Option.map epsilon ~f:Ops.f)
      | Momentum (alpha, momentum) ->
        Optimizers.momentum_minimizer loss
          ~alpha:(Ops.f alpha)
          ~momentum:(Ops.f momentum)
  end

  module Loss = struct
    type t =
      | Cross_entropy
      | L2_mean

    let cross_entropy = Cross_entropy
    let l2_mean = L2_mean

    let get t ~sample_ys ~model_ys =
      match t with
      | Cross_entropy ->
        Ops.(neg (reduce_mean (sample_ys * log model_ys)))
      | L2_mean ->
        Ops.(reduce_mean (square (sample_ys - model_ys)))
  end

  let create net =
    let session = Session.create () in
    let placeholder = Ops.placeholder ~type_:Float (dim_list net.shape) in
    { session
    ; net
    ; placeholder
    }

  let all_inputs ?(named_inputs=[]) t xs =
    let inputs =
      List.map named_inputs ~f:(fun (node, value) ->
        Session.Input.float node value)
    in
    match t.net.default_input with
    | None -> inputs
    | Some node -> Session.Input.float node xs :: inputs

  let fit ?named_inputs ~loss ~optimizer ~epochs ~xs ~ys t =
    let loss = Loss.get loss ~sample_ys:t.placeholder ~model_ys:t.net.node in
    let optimizer = Optimizer.get optimizer ~loss in
    let inputs =
      (Session.Input.float t.placeholder ys)
      :: all_inputs ?named_inputs t xs
    in
    for epoch = 1 to epochs do
      let err =
        Session.run
          ~inputs
          ~targets:optimizer
          ~session:t.session
          (Session.Output.scalar_float loss)
      in
      printf "Epoch: %6d/%-6d   Loss: %.2f\n%!" epoch epochs err
    done

  let evaluate ?named_inputs t xs =
    let inputs = all_inputs ?named_inputs t xs in
    Session.run
      ~inputs
      ~session:t.session
      (Session.Output.float t.net.node)
end
