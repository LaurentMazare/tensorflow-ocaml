open Core.Std
open Node

let get_shape ?shape values =
  (* TODO: check shape. *)
  match shape with
  | Some shape -> shape
  | None -> [ List.length values ]

let const_float
    ?(name = "Const")
    ?shape
    ~type_
    values
  =
  let shape = get_shape ?shape values in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Const"
  ; output_type = type_
  ; inputs = []
  ; attributes = [
      "dtype", Type (P type_);
      "value", Tensor_float { type_ = P type_; shape; values };
    ]
  ; output_name = None
  }

let const_int
    ?(name = "Const")
    ?shape
    ~type_
    values
  =
  let shape = get_shape ?shape values in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Const"
  ; output_type = type_
  ; inputs = []
  ; attributes = [
      "dtype", Type (P type_);
      "value", Tensor_int { type_ = P type_; shape; values };
    ]
  ; output_name = None
  }

let scalar ~type_ f =
  const_float
    ~type_
    ~shape:[ 1 ]
    [ f ]

type 't b =  ?name:string -> 't Node.t -> 't Node.t -> 't Node.t

let (+) = Ops.add
let (-) = Ops.sub
let ( * ) = Ops.mul
let ( *^ ) = Ops.matMul ~transpose_a:false ~transpose_b:false
let (/) = Ops.div
let f x = const_float ~type_:Float ~shape:[] [ x ]
let d x = const_float ~type_:Double ~shape:[] [ x ]

let fl x = const_float ~type_:Float ~shape:[ List.length x ] x
let dl x = const_float ~type_:Double ~shape:[ List.length x ] x
