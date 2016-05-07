open Core_kernel.Std

type _1d
type _2d
type _3d

module Shape = struct
  type 'a t =
    | D1 : int -> _1d t
    | D2 : int * int -> _2d t
    | D3 : int * int * int -> _3d t

  let dim_list (type a) (t : a t) =
    match t with
    | D1 d -> [ d ]
    | D2 (d, d') -> [ d; d' ]
    | D3 (d, d', d'') -> [ d; d'; d'' ]
end

exception Shape_mismatch of int list * int list * string
let () =
  Caml.Printexc.register_printer (function
    | Shape_mismatch (dims, dims', str) ->
      let dims = List.map dims ~f:Int.to_string |> String.concat ~sep:", " in
      let dims' = List.map dims' ~f:Int.to_string |> String.concat ~sep:", " in
      Some (sprintf "Shape mismatch %s: %s <> %s" str dims dims')
    | _ -> None)

let shape_mismatch shape1 shape2 ~op_name =
  let shape1 = Shape.dim_list shape1 in
  let shape2 = Shape.dim_list shape2 in
  raise (Shape_mismatch (shape1, shape2, op_name))

module Id = struct
  include Int

  let create =
    let cnt = ref 0 in
    fun () ->
      incr cnt;
      !cnt
end

module Input_id = struct
  type t = Id.t
end

module Unary = struct
  type t =
    | Sigmoid
    | Tanh
    | Relu
    | Softmax

  let apply t node =
    match t with
    | Sigmoid -> Ops.sigmoid node
    | Tanh    -> Ops.tanh node
    | Relu    -> Ops.relu node
    | Softmax -> Ops.softmax node
end

module Binary = struct
  type t =
    | Plus
    | Minus
    | Times

  let op_name = function
    | Plus  -> "plus"
    | Minus -> "minus"
    | Times -> "times"

  let apply t node1 node2 =
    match t with
    | Plus  -> Ops.(node1 + node2)
    | Minus -> Ops.(node1 - node2)
    | Times -> Ops.(node1 * node2)
end

type init = [ `const of float | `normal of float | `truncated_normal of float ]

type 'a op =
  | Input : 'a op
  | Const : float -> 'a op
  | Unary : Unary.t * 'a t -> 'a op
  | Binary : Binary.t * 'a t * 'a t -> 'a op
  | Dense : init * init * _1d t -> _1d op
and 'a t =
  { shape : 'a Shape.t
  ; op : 'a op
  ; id : Id.t
  }

type p = P : _ t -> p

let shape t = t.shape

let input ~shape =
  let id = Id.create () in
  { shape
  ; op = Input
  ; id
  }, id

let const f ~shape =
  { shape
  ; op = Const f
  ; id = Id.create ()
  }

let unary unary t =
  { shape = shape t
  ; op = Unary (unary, t)
  ; id = Id.create ()
  }

let sigmoid t = unary Sigmoid t
let tanh t = unary Tanh t
let relu t = unary Relu t
let softmax t = unary Softmax t

let binary binary t1 t2 =
  if t1.shape <> t2.shape
  then shape_mismatch t1.shape t2.shape ~op_name:(Binary.op_name binary);
  { shape = shape t1
  ; op = Binary (binary, t1, t2)
  ; id = Id.create ()
  }

let dense ?(w_init = `const 0.) ?(b_init = `const 0.) dim =
  let id = Id.create () in
  Staged.stage (fun t ->
    { shape = D1 dim
    ; op = Dense (w_init, b_init, t)
    ; id
    })

let var dims ~init ~type_ =
  match init with
  | `const f -> Var.f_or_d dims f ~type_
  | `normal stddev -> Var.normal dims ~stddev ~type_
  | `truncated_normal stddev -> Var.truncated_normal dims ~stddev ~type_

let build_node t ~type_ =
  let inputs = Id.Table.create () in
  let dense_vars = Id.Table.create () in
  let rec walk (P t) =
    match t.op with
    | Unary (unary, t) -> Unary.apply unary (walk (P t))
    | Binary (binary, t1, t2) -> Binary.apply binary (walk (P t1)) (walk (P t2))
    | Const f -> Ops.f_or_d ~shape:(Shape.dim_list t.shape) ~type_ f
    | Dense (w_init, b_init, input)->
      let Shape.D1 input_shape = input.shape in
      let Shape.D1 shape = t.shape in
      printf "%d %d\n%!" input_shape shape;
      let w, b =
        Hashtbl.find_or_add dense_vars t.id ~default:(fun () ->
          let w = var ~type_ ~init:w_init [ input_shape; shape ] in
          let b = var ~type_ ~init:b_init [ shape ] in
          w, b)
      in
      Ops.(walk (P input) *^ w + b)
    | Input ->
      Hashtbl.find_or_add inputs t.id ~default:(fun () ->
        Ops.placeholder ~type_ (Shape.dim_list t.shape))
      |> Ops.Placeholder.to_node
  in
  walk t, inputs

module Optimizer = struct
  (* We should use some inline records here when they will be available. *)
  type t =
    | Gradient_descent of float
    | Adam of float * float option * float option * float option
    | Momentum of float * float

  let gradient_descent ~learning_rate = Gradient_descent learning_rate

  let adam ~learning_rate ?beta1 ?beta2 ?epsilon () =
    Adam (learning_rate, beta1, beta2, epsilon)

  let momentum ~learning_rate ~momentum =
    Momentum (learning_rate, momentum)

  let get t ~loss =
    match t with
    | Gradient_descent learning_rate ->
      Optimizers.gradient_descent_minimizer ~learning_rate:(Ops.f learning_rate) loss
    | Adam (learning_rate, beta1, beta2, epsilon) ->
      Optimizers.adam_minimizer loss
        ~learning_rate:(Ops.f learning_rate)
        ?beta1:(Option.map beta1 ~f:Ops.f)
        ?beta2:(Option.map beta2 ~f:Ops.f)
        ?epsilon:(Option.map epsilon ~f:Ops.f)
    | Momentum (learning_rate, momentum) ->
      Optimizers.momentum_minimizer loss
        ~learning_rate:(Ops.f learning_rate)
        ~momentum:(Ops.f momentum)
end

module Loss = struct
  type t =
    | Cross_entropy of [ `sum | `mean ]
    | L2 of [ `sum | `mean ]

  let cross_entropy sum_mean = Cross_entropy sum_mean
  let l2 sum_mean = L2 sum_mean

  let get t ~sample_ys ~model_ys =
    let reduce = function
      | `sum -> Ops.reduce_sum
      | `mean -> Ops.reduce_mean
    in
    match t with
    | Cross_entropy sum_mean ->
      Ops.(neg (reduce sum_mean (sample_ys * log model_ys)))
    | L2 sum_mean ->
      Ops.(reduce sum_mean (square (sample_ys - model_ys)))
end

module Model = struct
  type 'a fnn = 'a t

  type ('a, 'b, 'c) t =
    { session : Session.t
    ; node : 'b Node.t
    ; placeholder : 'b Ops.Placeholder.t
    ; inputs : 'b Ops.Placeholder.t Id.Table.t
    ; save_nodes : [ `unit ] Node.t String.Table.t
    ; load_and_assign_nodes : Node.p list String.Table.t
    ; eq : ('c * 'b) Tensor.eq
    }

  let create (type a) (type b) (eq : (b * a) Tensor.eq) fnn =
    let create eq ~type_ =
      let node, inputs = build_node (P fnn) ~type_ in
      let session = Session.create () in
      let placeholder = Ops.placeholder ~type_ (Shape.dim_list fnn.shape) in
      { session
      ; node
      ; placeholder
      ; inputs
      ; save_nodes = String.Table.create ()
      ; load_and_assign_nodes = String.Table.create ()
      ; eq
      }
    in
    match eq with
    | Tensor.Float -> (create Float ~type_:Float : (_, a, b) t)
    | Tensor.Double -> create Double ~type_:Double

  let predict (type a) (type b)
        (t : (_, a, b) t)
        (inputs : (Input_id.t * (float, b) Tensor.t) list)
    =
    let predict f_or_d_input f_or_d_output =
      let inputs =
        List.map inputs ~f:(fun (id, tensor) ->
          match Hashtbl.find t.inputs id with
          | None -> failwith "missing input"
          | Some placeholder -> f_or_d_input placeholder tensor)
      in
      Session.run ~inputs ~session:t.session (f_or_d_output t.node)
    in
    match Node.output_type t.node, t.eq with
    | Node.Type.Float, Tensor.Float ->
      (predict Session.Input.float Session.Output.float : (float, b) Tensor.t)
    | Node.Type.Double, Tensor.Double ->
      (predict Session.Input.double Session.Output.double : (float, b) Tensor.t)
    | _ -> assert false

  let fit (type a) (type b)
        ?(addn_inputs : (Input_id.t * (float, b) Tensor.t) list option)
        ?batch_size
        (t : (_, a, b) t)
        ~loss
        ~optimizer
        ~epochs
        ~input_id
        ~xs
        ~ys
  =
  let fit placeholder node f_or_d scalar_f_or_d =
    let loss =
      Loss.get loss
        ~sample_ys:(Ops.Placeholder.to_node placeholder)
        ~model_ys:node
    in
    let optimizer = Optimizer.get optimizer ~loss in
    let samples = (Tensor.dims xs).(0) in
    let batch_size =
      match batch_size with
      | None -> None
      | Some batch_size when batch_size > samples -> None
      | Some _ as s -> s
    in
    let find_input id =
      match Hashtbl.find t.inputs id with
      | None -> failwith "missing input"
      | Some placeholder -> placeholder
    in
    let addn_inputs =
      Option.value_map addn_inputs
        ~default:[]
        ~f:(List.map ~f:(fun (id, tensor) -> f_or_d (find_input id) tensor))
    in
    let xs_placeholder = find_input input_id in
    let inputs ~xs ~ys =
      f_or_d xs_placeholder xs :: f_or_d t.placeholder ys :: addn_inputs
    in
    for epoch = 1 to epochs do
      let inputs =
        match batch_size with
        | None -> inputs ~xs ~ys
        | Some batch_size ->
          let offset = ((epoch-1) * batch_size) mod (samples - batch_size) in
          let xs = Tensor.sub_left xs offset batch_size in
          let ys = Tensor.sub_left ys offset batch_size in
          inputs ~xs ~ys
      in
      let err =
        Session.run
          ~inputs
          ~targets:optimizer
          ~session:t.session
          (scalar_f_or_d loss)
      in
      printf "Epoch: %6d/%-6d   Loss: %.2f\n%!" epoch epochs err
    done
  in
  match Node.output_type t.node, t.eq with
  | Node.Type.Float, Tensor.Float ->
    fit t.placeholder t.node Session.Input.float Session.Output.scalar_float
  | Node.Type.Double, Tensor.Double ->
    fit t.placeholder t.node Session.Input.double Session.Output.scalar_double
  | _ -> assert false

  (* Collect all variables in a net. The order of the created list is important as it
     will serve to name the variable.
     This does not seem very robust but will do for now. *)
  let all_vars_with_names t =
    Var.get_all_vars t.node
    |> List.mapi ~f:(fun i var -> sprintf "V%d" i, var)

  let save t ~filename =
    let save_node =
      Hashtbl.find_or_add t.save_nodes filename ~default:(fun () ->
        Ops.save ~filename (all_vars_with_names t))
    in
    Session.run
      ~session:t.session
      ~targets:[ Node.P save_node ]
      Session.Output.empty

  let load t ~filename =
    let load_and_assign_nodes =
      Hashtbl.find_or_add t.load_and_assign_nodes filename ~default:(fun () ->
        let filename = Ops.const_string [ filename ] in
        List.map (all_vars_with_names t) ~f:(fun (var_name, (Node.P var)) ->
          Ops.restore
            ~type_:(Node.output_type var)
            filename
            (Ops.const_string [ var_name ])
          |> Ops.assign var
          |> fun node -> Node.P node))
    in
    Session.run
      ~session:t.session
      ~targets:load_and_assign_nodes
      Session.Output.empty
end

let (+) t1 t2 = binary Plus t1 t2
let (-) t1 t2 = binary Minus t1 t2
let ( * ) t1 t2 = binary Times t1 t2
