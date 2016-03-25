open Core_kernel.Std
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
let ( *. ) = Ops.matMul ~transpose_a:false ~transpose_b:true
let (/) = Ops.div

let f_or_d ?shape ~type_ x =
  let scalar = const_float ~type_ ~shape:[] [ x ] in
  match shape with
  | None -> scalar
  | Some dims -> Ops.fill (const_int ~type_:Int32 dims) scalar

let f ?shape x = f_or_d ?shape ~type_:Float x
let d ?shape x = f_or_d ?shape ~type_:Double x

let cf ?shape x = const_float ?shape ~type_:Float x
let cd ?shape x = const_float ?shape ~type_:Double x

let varf shape =
  Ops.variable ()
    ~type_:Float
    ~shape:(List.map shape ~f:(fun size -> { Node.Dim.size; name = None }))

let vard shape =
  Ops.variable ()
    ~type_:Double
    ~shape:(List.map shape ~f:(fun size -> { Node.Dim.size; name = None }))
