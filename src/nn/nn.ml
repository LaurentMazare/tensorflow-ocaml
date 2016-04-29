open Core_kernel.Std

exception Shape_mismatch of int list * int list * string
let () =
  Caml.Printexc.register_printer (function
    | Shape_mismatch (dims, dims', str) ->
      let dims = List.map dims ~f:Int.to_string |> String.concat ~sep:", " in
      let dims' = List.map dims' ~f:Int.to_string |> String.concat ~sep:", " in
      Some (sprintf "Shape mismatch %s: %s <> %s" str dims dims')
    | _ -> None)

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

module Input_name = struct
  type 'a t = 'a Ops.Placeholder.t

  let id t = Ops.Placeholder.to_node t |> Node.id

  let merge t_option t_option' =
    match t_option, t_option' with
    | None, None -> None
    | (Some _ as s), None | None, (Some _ as s) -> s
    | Some t as s, Some t' when Node.Id.(=) (id t) (id t') -> s
    | Some _, Some _ -> failwith "Different inputs"

  let to_placeholder = Fn.id
end

type ('a, 'b) t =
  { shape : 'a Shape.t
  ; node : 'b Node.t
  ; variables : 'b Node.t list
  ; default_input : 'b Input_name.t option
  ; type_ : 'b Node.Type.t
  }

type init = [ `const of float | `normal of float | `truncated_normal of float ]

let shape t = t.shape
let default_input t = t.default_input
let node t = t.node
let type_ t = t.type_

let named_input ~shape ~type_ =
  let placeholder = Ops.placeholder ~type_ (-1 :: Shape.dim_list shape) in
  let t =
    { shape
    ; node = Ops.Placeholder.to_node placeholder
    ; variables = []
    ; default_input = None
    ; type_
    }
  in
  placeholder, t

let input ~shape ~type_ =
  let placeholder = Ops.placeholder ~type_ (-1 :: Shape.dim_list shape) in
  { shape
  ; node = Ops.Placeholder.to_node placeholder
  ; variables = []
  ; default_input = Some placeholder
  ; type_
  }

let shape_mismatch shape1 shape2 ~op_name =
  let shape1 = Shape.dim_list shape1 in
  let shape2 = Shape.dim_list shape2 in
  raise (Shape_mismatch (shape1, shape2, op_name))

let padding_str = function
  | `same -> "SAME"
  | `valid -> "VALID"

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

module Shared_var = struct
  let with_shape ~f g =
    let shape_a = ref (`F f) in
    let f t =
      let s = t.shape in
      match !shape_a with
      | `F f ->
        let a = f ~shape:s ~type_:t.type_ in
        shape_a := `Computed (s, a);
        a
      | `Computed (shape, a) ->
        if s <> shape
        then failwith "Dimensions do not match"
        else a
    in
    Staged.stage (g f)

  let var dims ~init ~type_ =
    match init with
    | `const f -> Var.f_or_d dims f ~type_
    | `normal stddev -> Var.normal dims ~stddev ~type_
    | `truncated_normal stddev -> Var.truncated_normal dims ~stddev ~type_

  let dense ?(w_init = `const 0.) ?(b_init = `const 0.) ~shape () =
    with_shape ~f:(fun ~shape:input_shape ~type_ ->
      let input_shape =
        match input_shape with
        | Shape.D1 input_shape -> input_shape
      in
      let w = var ~type_ ~init:w_init [ input_shape; shape ] in
      let b = var ~type_ ~init:b_init [ shape ] in
      w, b)
      (fun f t ->
        let w, b = f t in
        let node = Ops.(t.node *^ w + b) in
        { shape = D1 shape
        ; node
        ; variables = [ w; b ]
        ; default_input = t.default_input
        ; type_ = t.type_
        })

  let conv2d
        ?(w_init = `const 0.)
        ?(b_init = `const 0.)
        ~filter
        ~out_channels
        ~strides
        ~padding
        ()
    =
    let filter_height, filter_width = filter in
    let stride_height, stride_width = strides in
    let strides = [ 1; stride_height; stride_width; 1 ] in
    with_shape ~f:(fun ~shape:input_shape ~type_ ->
      let image_height, image_width, in_channels =
        match input_shape with
        | Shape.D3 (d1, d2, d3) -> d1, d2, d3
      in
      let w =
        var ~type_ ~init:w_init [ filter_height; filter_width; in_channels; out_channels ]
      in
      let b = var ~type_ ~init:b_init [ out_channels ] in
      image_height, image_width, w, b)
      (fun f t ->
        let input_height, input_width, w, b = f t in
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
        let padding = padding_str padding in
        let node = Ops.(conv2D ~strides ~padding t.node w + b) in
        { shape = D3 (output_height, output_width, out_channels)
        ; node
        ; variables = [ w; b ]
        ; default_input = t.default_input
        ; type_ = t.type_
        })
end

let f v ~shape =
  { node = Ops.f v ~shape:(Shape.dim_list shape)
  ; shape
  ; variables = []
  ; default_input = None
  ; type_ = Float
  }

let unary op t = { t with node = op t.node }

let sigmoid t = unary Ops.sigmoid t
let relu t = unary Ops.relu t
let tanh t = unary Ops.tanh t
let softmax t = unary Ops.softmax t

let max_pool t ~filter ~strides ~padding =
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
  let node =
    Ops.maxPool t.node
      ~ksize:[ 1; filter_height; filter_width; 1 ]
      ~strides:[ 1; stride_height; stride_width; 1 ]
      ~padding:(padding_str padding)
  in
  { node
  ; shape = D3 (output_height, output_width, input_channels)
  ; variables = t.variables
  ; default_input = t.default_input
  ; type_ = t.type_
  }

let dense ?w_init ?b_init t ~shape =
  Staged.unstage (Shared_var.dense ?w_init ?b_init ~shape ()) t

let conv2d ?w_init ?b_init t ~filter ~out_channels ~strides ~padding =
  Staged.unstage
    (Shared_var.conv2d ?w_init ?b_init ~filter ~out_channels ~strides ~padding ())
    t

let concat t1 t2 =
  let shape =
    match t1.shape, t2.shape with
    | Shape.D1 shape, Shape.D1 shape' -> Shape.D1 (shape + shape')
  in
  { variables = t1.variables @ t2.variables
  ; shape
  (* We use one32 as the concat dim as the batch-size dimension is 0. *)
  ; node = Ops.(concat one32 [ t1.node; t2.node ])
  ; default_input = Input_name.merge t1.default_input t2.default_input
  ; type_ = t1.type_
  }

let binary ~op_name op t1 t2 =
  if t1.shape <> t2.shape
  then shape_mismatch t1.shape t2.shape ~op_name;
  { node = op t1.node t2.node
  ; shape = t1.shape
  ; variables = t1.variables @ t2.variables
  ; default_input = Input_name.merge t1.default_input t2.default_input
  ; type_ = t1.type_
  }

let ( * ) t t' = binary ~op_name:"Mul" Ops.( * ) t t'

let (+) t t' = binary ~op_name:"Add" Ops.(+) t t'
let (-) t t' = binary ~op_name:"Sub" Ops.(-) t t'

let reshape t ~shape =
  let dim_list = Shape.dim_list shape in
  let total_dim_output = Shape.total_dim shape in
  let total_dim_input = Shape.total_dim t.shape in
  if total_dim_output <> total_dim_input
  then shape_mismatch shape t.shape ~op_name:"reshape";
  let node = Ops.reshape t.node (Ops.const_int ~type_:Int32 (-1 :: dim_list)) in
  { node
  ; shape
  ; variables = t.variables
  ; default_input = t.default_input
  ; type_ = t.type_
  }

let flatten t =
  reshape t ~shape:(D1 (Shape.total_dim t.shape))

let split (t : (_2d, _) t) =
  let Shape.D2 (num_split, d) = t.shape in
  Ops.(split ~num_split one32 t.node)
  |> List.map ~f:(fun node ->
      { variables = t.variables
      ; node
      ; shape = D1 d
      ; default_input = t.default_input
      ; type_ = t.type_
      }
      |> flatten
    )

let concatN (l : (_1d, _) t list) =
  match l with
  | [] -> failwith "concat called on an empty list"
  | hd :: _ as full_list ->
    let shape { shape = Shape.D1 shape; _ } = shape in
    let hd_shape = shape hd in
    let default_input =
      List.map full_list ~f:(fun t -> t.default_input)
      |> List.reduce_exn ~f:Input_name.merge
    in
    List.iter full_list ~f:(fun t ->
      if hd_shape <> shape t
      then raise (Shape_mismatch ([ hd_shape ], [ shape t ], "concatN")));
    let node =
      List.map full_list ~f:(fun t -> (reshape t ~shape:(D2 (1, hd_shape))).node)
      |> Ops.(concat one32)
    in
    { variables = List.concat_map full_list ~f:(fun t -> t.variables)
    ; shape = D2 (List.length full_list, hd_shape)
    ; node
    ; default_input
    ; type_ = hd.type_
    }
