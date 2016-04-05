open Core_kernel.Std
exception Shape_mismatch of int list * int list * string

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
      if shape <> input_shape
      then shape_mismatch (D1 shape) (D1 input_shape) ~op_name:"dense";
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

let dense t ~shape =
  Staged.unstage (Shared_var.dense ~shape) t

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
  type t
  type optimizer =
    | Gradient_descent of float

  type loss =
    | Cross_entropy

  let create _net = failwith "TODO"

  let fit t ~loss ~optimizer ~epochs ~xs ~ys =
    ignore (t, loss, optimizer, epochs, xs, ys);
    failwith "TODO"

  let evaluate t xs =
    ignore (t, xs);
    failwith "TODO"
end
