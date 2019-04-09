open Base
open Tensorflow_core
open! Tensorflow

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

  let total_dim (type a) (t : a t) =
    match t with
    | D1 d -> d
    | D2 (d, d') -> d * d'
    | D3 (d, d', d'') -> d * d' * d''
end

exception Shape_mismatch of int list * int list * string
let () =
  Caml.Printexc.register_printer (function
    | Shape_mismatch (dims, dims', str) ->
      let dims = List.map dims ~f:Int.to_string |> String.concat ~sep:", " in
      let dims' = List.map dims' ~f:Int.to_string |> String.concat ~sep:", " in
      Some (Printf.sprintf "Shape mismatch %s: %s <> %s" str dims dims')
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
    | Reduce_sum
    | Square
    | Neg

  let apply t node =
    match t with
    | Sigmoid    -> Ops.sigmoid    node
    | Tanh       -> Ops.tanh       node
    | Relu       -> Ops.relu       node
    | Softmax    -> Ops.softmax    node
    | Reduce_sum -> Ops.reduce_sum node
    | Square     -> Ops.square     node
    | Neg        -> Ops.neg        node
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

type pool =
  { filter : int * int
  ; strides : int * int
  ; padding : [ `same | `valid ]
  ; avg_or_max : [ `avg | `max ]
  }

type conv2d =
  { filter : int * int
  ; strides : int * int
  ; padding : [ `same | `valid ]
  ; in_channels : int
  ; out_channels : int
  ; w_init : init
  ; b_init : init
  ; name : string option
  }

type 'a op =
  | Input : 'a op
  | Const : float -> 'a op
  | Unary : Unary.t * 'a t -> 'a op
  | Binary : Binary.t * 'a t * 'a t -> 'a op
  | Dense : init * init * _1d t * string option -> _1d op
  | Pool : pool * _3d t -> _3d op
  | Conv2d : conv2d * _3d t -> _3d op
  | Reshape : 'a Shape.t * 'b t -> 'a op
  | Concat : _1d t list -> _2d op
  | Split : _2d t * int * int -> _1d op
  | Var : 'a t -> 'a op
and 'a t =
  { shape : 'a Shape.t
  ; op : 'a op
  ; id : Id.t
  }

type p = P : _ t -> p

let shape t = t.shape
let id t = t.id

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

let sigmoid    t = unary Sigmoid    t
let tanh       t = unary Tanh       t
let relu       t = unary Relu       t
let softmax    t = unary Softmax    t
let reduce_sum t = unary Reduce_sum t
let square     t = unary Square     t
let neg        t = unary Neg        t

let binary binary t1 t2 =
  if Caml.(<>) t1.shape t2.shape
  then shape_mismatch t1.shape t2.shape ~op_name:(Binary.op_name binary);
  { shape = shape t1
  ; op = Binary (binary, t1, t2)
  ; id = Id.create ()
  }

let reshape t ~shape =
  { shape
  ; op = Reshape (shape, t)
  ; id = Id.create ()
  }

let flatten t =
  reshape t ~shape:(D1 (Shape.total_dim t.shape))

let split t =
  let id = Id.create () in
  let Shape.D2 (num_split, d) = t.shape in
  List.init num_split ~f:(fun idx ->
    { shape = D1 d
    ; op = Split (t, idx, num_split)
    ; id
    })

let concat = function
  | [] -> failwith "concat called on an empty list"
  | hd :: _ as l ->
    let shape { shape = Shape.D1 shape; _ } = shape in
    let hd_shape = shape hd in
    List.iter l ~f:(fun t ->
      if hd_shape <> shape t
      then raise (Shape_mismatch ([ hd_shape ], [ shape t ], "concat")));
    { shape = D2 (List.length l, hd_shape)
    ; op = Concat l
    ; id = Id.create ()
    }

let var t =
  { shape = t.shape
  ; op = Var t
  ; id = Id.create ()
  }

let dense' ?(w_init = `const 0.) ?(b_init = `const 0.) ?name dim =
  let id = Id.create () in
  Staged.stage (fun t ->
    { shape = D1 dim
    ; op = Dense (w_init, b_init, t, name)
    ; id
    })

let dense ?w_init ?b_init ?name dim =
  Staged.unstage (dense' ?w_init ?b_init ?name dim)

let conv_sizes
      ~input_height
      ~input_width
      ~filter_height
      ~filter_width
      ~stride_height
      ~stride_width
      ~padding
  =
  let input_height, input_width =
    match padding with
    | `same -> input_height, input_width
    | `valid -> input_height - filter_height + 1, input_width - filter_width + 1
  in
  (input_height - 1) / stride_height + 1, (input_width - 1) / stride_width + 1

let padding_str = function
  | `same -> "SAME"
  | `valid -> "VALID"

let pool ~avg_or_max t ~filter ~strides ~padding =
  let input_height, input_width, input_channels =
    match t.shape with
    | Shape.D3 (d, d', d'') -> d, d', d''
  in
  let filter_height, filter_width = filter in
  let stride_height, stride_width = strides in
  let output_height, output_width =
    conv_sizes
      ~input_height
      ~input_width
      ~filter_height
      ~filter_width
      ~stride_height
      ~stride_width
      ~padding
  in
  let pool =
    { filter
    ; strides
    ; padding
    ; avg_or_max
    }
  in
  { shape = D3 (output_height, output_width, input_channels)
  ; op = Pool (pool, t)
  ; id = Id.create ()
  }

let max_pool = pool ~avg_or_max:`max
let avg_pool = pool ~avg_or_max:`avg

let conv2d' ?(w_init = `const 0.) ?(b_init = `const 0.) ?name ~filter ~out_channels ~strides ~padding () =
  let id = Id.create () in
  Staged.stage (fun t ->
    let input_height, input_width, input_channels =
      match t.shape with
      | Shape.D3 (d, d', d'') -> d, d', d''
    in
    let conv2d =
      { filter
      ; strides
      ; padding
      ; in_channels = input_channels
      ; out_channels
      ; w_init
      ; b_init
      ; name
      }
    in
    let filter_height, filter_width = filter in
    let stride_height, stride_width = strides in
    let output_height, output_width =
      conv_sizes
        ~input_height
        ~input_width
        ~filter_height
        ~filter_width
        ~stride_height
        ~stride_width
        ~padding
    in
    { shape = D3 (output_height, output_width, out_channels)
    ; op = Conv2d (conv2d, t)
    ; id
    })

let conv2d ?w_init ?b_init ?name ~filter ~out_channels ~strides ~padding () =
  Staged.unstage (conv2d' ?w_init ?b_init ?name ~filter ~out_channels ~strides ~padding ())

let create_var dims ~init ~type_ =
  match init with
  | `const f -> Var.f_or_d dims f ~type_
  | `normal stddev -> Var.normal dims ~stddev ~type_
  | `truncated_normal stddev -> Var.truncated_normal dims ~stddev ~type_

let build_node t ~type_ =
  let inputs = Hashtbl.create (module Id) in
  let explicit_vars = Hashtbl.create (module Id) in
  let dense_vars = Hashtbl.create (module Id) in
  let conv_vars = Hashtbl.create (module Id) in
  let splits = Hashtbl.create (module Id) in
  let var_names = Hashtbl.create (module Node.Id) in
  let all_nodes = Hashtbl.create (module Id) in
  let rec walk (P t) =
    let node =
      match t.op with
      | Unary (unary, t) -> Unary.apply unary (walk (P t))
      | Binary (binary, t1, t2) -> Binary.apply binary (walk (P t1)) (walk (P t2))
      | Const f -> Ops.f_or_d ~shape:(Shape.dim_list t.shape) ~type_ f
      | Dense (w_init, b_init, input, name_opt)->
        let Shape.D1 input_shape = input.shape in
        let Shape.D1 shape = t.shape in
        let w, b =
          Hashtbl.find_or_add dense_vars t.id ~default:(fun () ->
            let w = create_var ~type_ ~init:w_init [ input_shape; shape ] in
            let b = create_var ~type_ ~init:b_init [ shape ] in
            Option.iter name_opt ~f:(fun name ->
              Hashtbl.set var_names ~key:(Node.id w) ~data:(name ^ "/" ^ name ^ "_weights");
              Hashtbl.set var_names ~key:(Node.id b) ~data:(name ^ "/" ^ name ^ "_biases"));
            w, b)
        in
        Ops.(walk (P input) *^ w + b)
      | Input ->
        Hashtbl.find_or_add inputs t.id ~default:(fun () ->
          Ops.placeholder ~type_ (-1 :: Shape.dim_list t.shape))
        |> Ops.Placeholder.to_node
      | Pool (pool, t) ->
        let filter_height, filter_width = pool.filter in
        let stride_height, stride_width = pool.strides in
        let pool_ops =
          match pool.avg_or_max with
          | `max -> Ops.maxPool
          | `avg -> Ops.avgPool
        in
        (* [...Pool] only exists for float and not for double so cast to float. *)
        pool_ops (walk (P t) |> Ops.cast ~type_:Float)
          ~ksize:[ 1; filter_height; filter_width; 1 ]
          ~strides:[ 1; stride_height; stride_width; 1 ]
          ~padding:(padding_str pool.padding)
        |> Ops.cast ~type_
      | Conv2d (conv2d, u) ->
        let filter_height, filter_width = conv2d.filter in
        let out_channels = conv2d.out_channels in
        let w, b, in_channels =
          Hashtbl.find_or_add conv_vars t.id ~default:(fun () ->
            let in_channels = conv2d.in_channels in
            let w =
              create_var ~type_ ~init:conv2d.w_init
                [ filter_height; filter_width; in_channels; out_channels ]
            in
            let b = create_var ~type_ ~init:conv2d.b_init [ out_channels ] in
            Option.iter conv2d.name ~f:(fun name ->
              Hashtbl.set var_names ~key:(Node.id w) ~data:(name ^ "/" ^ name ^ "_filters");
              Hashtbl.set var_names ~key:(Node.id b) ~data:(name ^ "/" ^ name ^ "_biases"));
            w, b, in_channels)
        in
        if in_channels <> conv2d.in_channels
        then shape_mismatch (D1 in_channels) (D1 conv2d.in_channels) ~op_name:"conv2d in-channels";
        let stride_height, stride_width = conv2d.strides in
        let strides = [ 1; stride_height; stride_width; 1 ] in
        Ops.(conv2D ~strides ~padding:(padding_str conv2d.padding) (walk (P u)) w + b)
      | Reshape (shape, u) ->
        let dim_list = Shape.dim_list shape in
        let total_dim_output = Shape.total_dim shape in
        let total_dim_input = Shape.total_dim u.shape in
        if total_dim_output <> total_dim_input
        then shape_mismatch shape u.shape ~op_name:"reshape";
        Ops.reshape (walk (P u)) (Ops.const_int ~type_:Int32 (-1 :: dim_list))
      | Concat list ->
        List.map list ~f:(fun t -> walk (P t))
        |> Ops.(concat one32)
      | Split (u, idx, num_split) ->
        let list =
          Hashtbl.find_or_add splits t.id ~default:(fun () ->
            Ops.(split ~num_split one32 (walk (P u))))
        in
        List.nth_exn list idx
      | Var u ->
        Hashtbl.find_or_add explicit_vars t.id ~default:(fun () ->
          let dim_list = Shape.dim_list t.shape in
          Var.create dim_list
            ~type_
            ~init:(walk (P u)))
    in
    Hashtbl.set all_nodes ~key:t.id ~data:node;
    node
  in
  walk t, inputs, var_names, explicit_vars, all_nodes

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

  let get ?varsf ?varsd t ~loss =
    match t with
    | Gradient_descent learning_rate ->
      Optimizers.gradient_descent_minimizer ?varsf ?varsd ~learning_rate:(Ops.f learning_rate) loss
    | Adam (learning_rate, beta1, beta2, epsilon) ->
      Optimizers.adam_minimizer loss
        ?varsf
        ?varsd
        ~learning_rate:(Ops.f learning_rate)
        ?beta1:(Option.map beta1 ~f:Ops.f)
        ?beta2:(Option.map beta2 ~f:Ops.f)
        ?epsilon:(Option.map epsilon ~f:Ops.f)
    | Momentum (learning_rate, momentum) ->
      Optimizers.momentum_minimizer loss
        ?varsf
        ?varsd
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
    { node : 'b Node.t
    ; placeholder : 'b Ops.Placeholder.t
    ; inputs : (Id.t, 'b Ops.Placeholder.t) Hashtbl.t
    ; save_nodes : (string, [ `unit ] Node.t) Hashtbl.t
    ; load_and_assign_nodes : (string, Node.p list) Hashtbl.t
    ; var_names : (Node.Id.t, string) Hashtbl.t
    ; explicit_vars : (Id.t, 'b Node.t) Hashtbl.t
    ; all_nodes : (Id.t, 'b Node.t) Hashtbl.t
    ; eq : ('c * 'b) Tensor.eq
    }

  let create (type a) (type b) (eq : (b * a) Tensor.eq) fnn =
    let create eq ~type_ =
      let node, inputs, var_names, explicit_vars, all_nodes = build_node (P fnn) ~type_ in
      let placeholder = Ops.placeholder ~type_ (Shape.dim_list fnn.shape) in
      { node
      ; placeholder
      ; inputs
      ; save_nodes = Hashtbl.create (module String)
      ; load_and_assign_nodes = Hashtbl.create (module String)
      ; var_names
      ; explicit_vars
      ; all_nodes
      ; eq
      }
    in
    match eq with
    | Tensor.Float -> (create Float ~type_:Float : (_, a, b) t)
    | Tensor.Double -> failwith "The Double type is not supported."

  let predict (type a) (type b)
        (t : (_, a, b) t)
        ?output_id
        (inputs : (Input_id.t * (float, b) Tensor.t) list)
    =
    let predict f_or_d_input f_or_d_output =
      let inputs =
        List.map inputs ~f:(fun (id, tensor) ->
          match Hashtbl.find t.inputs id with
          | None -> failwith "missing input"
          | Some placeholder -> f_or_d_input placeholder tensor)
      in
      Session.run ~inputs (f_or_d_output t.node)
    in
    let output_node =
      match output_id with
      | None -> t.node
      | Some id ->
        match Hashtbl.find t.all_nodes id with
        | None -> failwith "Cannot find any node with the proper id"
        | Some node -> node
    in
    match Node.output_type output_node, t.eq with
    | Node.Type.Float, Tensor.Float ->
      (predict Session.Input.float Session.Output.float : (float, b) Tensor.t)
    | Node.Type.Double, Tensor.Double ->
      (predict Session.Input.double Session.Output.double : (float, b) Tensor.t)
    | _ -> .

  let fit (type a) (type b)
        ?(addn_inputs : (Input_id.t * (float, b) Tensor.t) list option)
        ?batch_size
        ?explicit_vars
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
    let optimizer =
      let varsf, varsd =
        match explicit_vars with
        | None -> None, None
        | Some explicit_vars ->
          match t.eq with
          | Tensor.Float ->
            let nodes =
              List.map explicit_vars ~f:(fun ev ->
                Hashtbl.find_exn t.explicit_vars ev.id)
            in
            Some (nodes : [ `float ] Node.t list), None
          | Tensor.Double ->
            let nodes =
              List.map explicit_vars ~f:(fun ev ->
                Hashtbl.find_exn t.explicit_vars ev.id)
            in
            None, Some (nodes : [ `double ] Node.t list)
        in
      Optimizer.get optimizer ~loss ?varsf ?varsd
    in
    let samples = (Tensor.dims xs).(0) in
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
    let batches_per_epoch, batch_inputs =
      match batch_size with
      | None -> (1, fun _n -> inputs)
      | Some batch_size when batch_size >= samples -> (1, fun _step -> inputs)
      | Some batch_size ->
        (samples / batch_size,
        fun step ~xs ~ys ->
          let offset = (step * batch_size) in
          let xs = Tensor.sub_left xs offset batch_size in
          let ys = Tensor.sub_left ys offset batch_size in
          inputs ~xs ~ys
        )
    in
    for epoch = 1 to epochs do
      let err_total = ref 0.0 in
      for step = 0 to batches_per_epoch-1 do
        let inputs = batch_inputs step ~xs ~ys in
        let err =
          Session.run
            ~inputs
            ~targets:optimizer
            (scalar_f_or_d loss)
        in
        (* Calculate the mean error over all batches *)
        err_total := !err_total +. err;
      done;
      Stdio.printf "Epoch: %6d/%-6d   Loss: %.2f\n%!" epoch epochs (!err_total /. Float.of_int batches_per_epoch)
    done
  in
  match Node.output_type t.node, t.eq with
  | Node.Type.Float, Tensor.Float ->
    fit t.placeholder t.node Session.Input.float Session.Output.scalar_float
  | Node.Type.Double, Tensor.Double ->
    fit t.placeholder t.node Session.Input.double Session.Output.scalar_double
  | _ -> .

  let all_vars_with_names t =
    Var.get_all_vars [Node.P t.node]
    |> List.filter_map ~f:(fun var ->
      let name =
        let node_id =
          match var with
          | Node.P v -> Node.id v
        in
        Hashtbl.find t.var_names node_id
      in
      Option.map name ~f:(fun name -> name, var))

  let input_list (type a) (type b)
      (t : (_, a, b) t)
      (inputs : (Input_id.t * (float, b) Tensor.t) list)
    =
    List.map inputs ~f:(fun (id, tensor) ->
      match Hashtbl.find t.inputs id with
      | None -> failwith "missing input"
      | Some placeholder ->
        match Node.output_type t.node, t.eq with
        | Node.Type.Float, Tensor.Float ->
          Session.Input.float placeholder tensor
        | Node.Type.Double, Tensor.Double ->
          Session.Input.double placeholder tensor
        | _ -> .)

  let save ?(inputs = []) t ~filename =
    let save_node =
      Hashtbl.find_or_add t.save_nodes filename ~default:(fun () ->
        let all_vars_with_names = all_vars_with_names t in
        if List.is_empty all_vars_with_names then
          failwith "No variable to save can be found (only named variables are saved)";
        Ops.save ~filename all_vars_with_names)
    in
    Session.run
      ~inputs:(input_list t inputs)
      ~targets:[ Node.P save_node ]
      Session.Output.empty

  let load ?(inputs = []) t ~filename =
    let load_and_assign_nodes =
      Hashtbl.find_or_add t.load_and_assign_nodes filename ~default:(fun () ->
        let filename = Ops.const_string0 filename in
        List.map (all_vars_with_names t) ~f:(fun (var_name, (Node.P var)) ->
          Ops.restore
            ~type_:(Node.output_type var)
            filename
            (Ops.const_string0 var_name)
          |> Ops.assign var
          |> fun node -> Node.P node))
    in
    Session.run
      ~inputs:(input_list t inputs)
      ~targets:load_and_assign_nodes
      Session.Output.empty
end

let (+) t1 t2 = binary Plus t1 t2
let (-) t1 t2 = binary Minus t1 t2
let ( * ) t1 t2 = binary Times t1 t2
