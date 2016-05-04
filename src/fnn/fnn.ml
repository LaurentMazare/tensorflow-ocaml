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

let (+) t1 t2 = binary Plus t1 t2
let (-) t1 t2 = binary Minus t1 t2
let ( * ) t1 t2 = binary Times t1 t2

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

module Model = struct
  type 'a fnn = 'a t

  type ('a, 'b) t =
    { session : Session.t
    ; node : 'b Node.t
    ; shape : 'a Shape.t
    ; inputs : 'b Ops.Placeholder.t Id.Table.t
    ; save_nodes : [ `unit ] Node.t String.Table.t
    ; load_and_assign_nodes : Node.p list String.Table.t
    }

  let create fnn type_ =
    let node, inputs = build_node (P fnn) ~type_ in
    let session = Session.create () in
    { session
    ; node
    ; shape = fnn.shape
    ; inputs
    ; save_nodes = String.Table.create ()
    ; load_and_assign_nodes = String.Table.create ()
    }

  let predict (type a) (type b)
        (t : (_, a) t)
        (inputs : (Input_id.t * (float, b) Tensor.t) list)
        (eq : (b * a) Tensor.eq)
    =
    match Node.output_type t.node, eq with
    | Node.Type.Float, Tensor.Float ->
      let inputs =
        List.map inputs ~f:(fun (id, tensor) ->
          match Hashtbl.find t.inputs id with
          | None -> failwith "missing input"
          | Some placeholder -> Session.Input.float placeholder tensor)
      in
      let output =
        Session.run ~inputs ~session:t.session Session.Output.(float t.node)
      in
      (output : (float, b) Tensor.t)
    | Node.Type.Double, Tensor.Double ->
      let inputs =
        List.map inputs ~f:(fun (id, tensor) ->
          match Hashtbl.find t.inputs id with
          | None -> failwith "missing input"
          | Some placeholder -> Session.Input.double placeholder tensor)
      in
      let output =
        Session.run ~inputs ~session:t.session Session.Output.(double t.node)
      in
      (output : (float, b) Tensor.t)
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
