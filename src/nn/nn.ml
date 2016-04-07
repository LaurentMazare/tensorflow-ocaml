(* TODO: handle double ? *)

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

  let to_node = Fn.id
end

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

type 'a t =
  { shape : 'a Shape.t
  ; node : [ `float ] Node.t
  ; variables : [ `float ] Node.t list
  ; default_input : Input_name.t option
  }

type init = [ `const of float | `normal of float ]

let shape t = t.shape
let default_input t = t.default_input
let node t = t.node

let named_input ~shape =
  let placeholder = Ops.placeholder ~type_:Float (-1 :: Shape.dim_list shape) in
  let t =
    { shape
    ; node = placeholder
    ; variables = []
    ; default_input = None
    }
  in
  placeholder, t

let input ~shape =
  let placeholder = Ops.placeholder ~type_:Float (-1 :: Shape.dim_list shape) in
  { shape
  ; node = placeholder
  ; variables = []
  ; default_input = Some placeholder
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
        let a = f ~shape:s in
        shape_a := `Computed (s, a);
        a
      | `Computed (shape, a) ->
        if s <> shape
        then failwith "Dimensions do not match"
        else a
    in
    Staged.stage (g f)

  let var ~init dims =
    match init with
    | `const f -> Var.f dims f
    | `normal stddev -> Var.normalf dims ~stddev

  let dense ?(w_init = `const 0.) ?(b_init = `const 0.) ~shape () =
    with_shape ~f:(fun ~shape:input_shape ->
      let input_shape =
        match input_shape with
        | Shape.D1 input_shape -> input_shape
      in
      let w = var ~init:w_init [ input_shape; shape ] in
      let b = var ~init:b_init [ shape ] in
      w, b)
      (fun f t ->
        let w, b = f t in
        let node = Ops.(t.node *^ w + b) in
        { shape = D1 shape
        ; node
        ; variables = [ w; b ]
        ; default_input = t.default_input
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
    with_shape ~f:(fun ~shape:input_shape ->
      let image_height, image_width, in_channels =
        match input_shape with
        | Shape.D3 (d1, d2, d3) -> d1, d2, d3
      in
      let w =
        var ~init:w_init [ filter_height; filter_width; in_channels; out_channels ]
      in
      let b = var ~init:b_init [ out_channels ] in
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
        })
end

let f v ~shape =
  { node = Ops.f v ~shape:(Shape.dim_list shape)
  ; shape
  ; variables = []
  ; default_input = None
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
  }

let flatten t =
  reshape t ~shape:(D1 (Shape.total_dim t.shape))

let split (t : _2d t) =
  let Shape.D2 (n, d) = t.shape in
  Ops.(split ~num_split:n one32 t.node)
  |> List.map
      ~f:(fun node ->
      { variables = t.variables
      ; node
      ; shape = D1 d
      ; default_input = t.default_input })
  |> List.map ~f:flatten

let concatN (l : _1d t list) =
  match l with
  | [] -> failwith "concat called on an empty list"
  | t::l ->
    let s t =
      let Shape.D1 shape = t.shape in
      shape
    in
    let shape = s t in
    let default_input =
       List.fold ~init:t.default_input l
         ~f:(fun default_input t ->
             Input_name.merge default_input t.default_input)
    in
    List.iter l
      ~f:(fun t ->
          if shape <> s t
          then failwithf "concat: shape mismatch between %i and %i" shape (s t) ());
    let l =
      List.map (t::l)
        ~f:(reshape ~shape:(D2 (1,shape)))
    in
    let node = Ops.(concat one32 (List.map l ~f:(fun t -> t.node))) in
    {variables = List.map l ~f:(fun t -> t.variables) |> List.concat;
     shape = D2 (List.length l, shape);
     node;
     default_input }



