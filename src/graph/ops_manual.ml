open Base
open Node

module Placeholder = struct
  type nonrec 'a t = 'a t

  let to_node = Fn.id
end

let string_of_shape shape =
  List.map shape ~f:Int.to_string |> String.concat ~sep:", " |> Printf.sprintf "[%s]"

let get_shape ?shape values =
  match shape with
  | Some shape ->
    let vs = List.fold shape ~init:1 ~f:( * ) in
    let len = List.length values in
    if vs <> len
    then
      raise (Invalid_argument (Printf.sprintf "Input length mismatch %d <> %d" vs len));
    shape
  | None -> [ List.length values ]

let const_float ?(name = "Const") ?(control_inputs = []) ?shape ~type_ values =
  let shape = get_shape ?shape values in
  Node.create
    ~name:(Name.of_string name)
    ~op_name:(Op_name.of_string "Const")
    ~output_type:type_
    ~inputs:[]
    ~control_inputs
    ~attributes:
      [ "dtype", Type (P type_)
      ; "value", Tensor_float { type_ = P type_; shape; values }
      ]
    ~output_idx:None

let const_int ?(name = "Const") ?(control_inputs = []) ?shape ~type_ values =
  let shape = get_shape ?shape values in
  Node.create
    ~name:(Name.of_string name)
    ~op_name:(Op_name.of_string "Const")
    ~output_type:type_
    ~inputs:[]
    ~control_inputs
    ~attributes:
      [ "dtype", Type (P type_); "value", Tensor_int { type_ = P type_; shape; values } ]
    ~output_idx:None

let const_string ?(name = "Const") ?shape values =
  let shape = get_shape ?shape values in
  Node.create
    ~name:(Name.of_string name)
    ~op_name:(Op_name.of_string "Const")
    ~output_type:String
    ~inputs:[]
    ~control_inputs:[]
    ~attributes:
      [ "dtype", Type (P String)
      ; "value", Tensor_string { type_ = P String; shape; values }
      ]
    ~output_idx:None

let const_string0 ?name value = const_string ?name ~shape:[] [ value ]

let scalar ?empty_shape ~type_ f =
  const_float ~type_ ~shape:(if Option.is_some empty_shape then [] else [ 1 ]) [ f ]

type 't b = ?name:string -> 't Node.t -> 't Node.t -> 't Node.t

let ( + ) = Ops_generated.add ~control_inputs:[]
let ( - ) = Ops_generated.sub ~control_inputs:[]
let ( * ) = Ops_generated.mul ~control_inputs:[]

let ( *^ ) =
  Ops_generated.matMul ~control_inputs:[] ~transpose_a:false ~transpose_b:false

let ( / ) = Ops_generated.div ~control_inputs:[]

let f_or_d ?shape ~type_ x =
  let scalar = const_float ~type_ ~shape:[] [ x ] in
  match shape with
  | None -> scalar
  | Some dims -> Ops_generated.fill (const_int ~type_:Int32 dims) scalar

let f ?shape x = f_or_d ?shape ~type_:Float x
let d ?shape x = f_or_d ?shape ~type_:Double x
let cf ?shape x = const_float ?shape ~type_:Float x
let cd ?shape x = const_float ?shape ~type_:Double x
let ci32 ?shape x = const_int ?shape ~type_:Int32 x
let ci64 ?shape x = const_int ?shape ~type_:Int64 x
let zero32 = const_int ~shape:[] ~type_:Int32 [ 0 ]
let one32 = const_int ~shape:[] ~type_:Int32 [ 1 ]
let two32 = const_int ~shape:[] ~type_:Int32 [ 2 ]
let three32 = const_int ~shape:[] ~type_:Int32 [ 3 ]
let four32 = const_int ~shape:[] ~type_:Int32 [ 4 ]
let range node = Ops_generated.range zero32 node one32

let reduce_op op ?dims node =
  let dims =
    match dims with
    | Some dims -> const_int ~type_:Int32 dims
    | None -> Ops_generated.range zero32 (Ops_generated.rank node) one32
  in
  op node dims

type 'a reduce_fn =
  ?dims:int list
  -> ([< `complex64 | `double | `float | `int32 | `int64 ] as 'a) Node.t
  -> 'a Node.t

let reduce_sum ?dims node = reduce_op Ops_generated.sum ?dims node
let reduce_mean ?dims node = reduce_op Ops_generated.mean ?dims node
let reduce_min ?dims node = reduce_op Ops_generated.min ?dims node
let reduce_max ?dims node = reduce_op Ops_generated.max ?dims node
let reduce_prod ?dims node = reduce_op Ops_generated.prod ?dims node
let reduce_all ?dims node = reduce_op Ops_generated.all ?dims node
let reduce_any ?dims node = reduce_op Ops_generated.any ?dims node

let placeholder ?name ~type_ shape =
  Ops_generated.placeholder
    ?name
    ~type_
    ~shape:(List.map shape ~f:(fun size -> { Node.Dim.name = None; size }))
    ()

let dropout node ~keep_prob =
  let type_ = Node.output_type node in
  let random =
    Ops_generated.randomUniform ~type_ (Ops_generated.shape node ~type_:Int32)
  in
  Ops_generated.floor (keep_prob + random)
  |> fun binary_tensor -> node * Ops_generated.reciprocal keep_prob * binary_tensor

let save_ ?(name = "Save") filename tensor_names tensors =
  let inputs =
    [ `single (Node.P filename); `single (Node.P tensor_names); `multi tensors ]
  in
  let type_list =
    List.map tensors ~f:(fun (Node.P tensor) -> Node.Type.P (Node.output_type tensor))
  in
  Node.create
    ~name:(Name.of_string name)
    ~op_name:(Op_name.of_string "Save")
    ~output_type:Unit
    ~inputs
    ~control_inputs:[]
    ~attributes:[ "T", List (Type type_list) ]
    ~output_idx:None

let save ~filename named_tensors =
  let tensor_names, tensors = List.unzip named_tensors in
  save_ (const_string0 filename) (const_string tensor_names) tensors

let split2 ?name dim node =
  match Ops_generated.split dim node ?name ~num_split:2 with
  | [ node1; node2 ] -> node1, node2
  | _ -> assert false

let split3 ?name dim node =
  match Ops_generated.split dim node ?name ~num_split:3 with
  | [ node1; node2; node3 ] -> node1, node2, node3
  | _ -> assert false

let split4 ?name dim node =
  match Ops_generated.split dim node ?name ~num_split:4 with
  | [ node1; node2; node3; node4 ] -> node1, node2, node3, node4
  | _ -> assert false

let cast ?name (type a b) (t : a Node.t) ~(type_ : b Node.Type.t) =
  match Node.output_type t, type_ with
  | Node.Type.Float, Node.Type.Float -> (t : b Node.t)
  | _ -> Ops_generated.cast ?name t ~type_

let count t ~dims =
  Ops_generated.gather (Ops_generated.shape t ~type_:Int32) (const_int ~type_:Int32 dims)
  |> reduce_prod

type 'a moments =
  { mean : 'a Node.t
  ; variance : 'a Node.t
  }

let moments t ~dims =
  let divisor =
    count t ~dims
    |> Ops_generated.cast ~type_:(Node.output_type t)
    |> Ops_generated.reciprocal
  in
  let mean = reduce_sum ~dims t * divisor in
  let square_sum = Ops_generated.square t |> reduce_sum ~dims in
  { mean; variance = (square_sum * divisor) - Ops_generated.square mean }

let normalize ?(epsilon = 1e-12) t { mean; variance } =
  let epsilon = scalar ~type_:(Node.output_type t) epsilon in
  Ops_generated.rsqrt (variance + epsilon) * (t - mean)

let cond_with_control_inputs t ~if_true ~if_false =
  let t_false, t_true = Ops_generated.switch t t in
  let if_true =
    if_true
    (* It is important to keep the [identity] below as control inputs do not handle
         ports. *)
      ~control_inputs:[ Node.P (Ops_generated.identity t_true) ]
  in
  let if_false =
    if_false
    (* It is important to keep the [identity] below as control inputs do not handle
         ports. *)
      ~control_inputs:[ Node.P (Ops_generated.identity t_false) ]
  in
  Ops_generated.merge [ if_true; if_false ] |> fst

let cond t ~if_true ~if_false =
  cond_with_control_inputs
    t
    ~if_true:(fun ~control_inputs -> Ops_generated.identity ~control_inputs if_true)
    ~if_false:(fun ~control_inputs -> Ops_generated.identity ~control_inputs if_false)

let shape32 = Ops_generated.shape ~type_:Int32

let cross_entropy ?(epsilon = 1e-7) ~ys ~y_hats sum_or_mean =
  let model_shape = Node.shape y_hats in
  if List.last_exn model_shape <= 1
  then
    raise
      (Invalid_argument
         (Printf.sprintf
            "the last dimension should greater than 1 %s"
            (string_of_shape model_shape)));
  let type_ = Node.output_type ys in
  let reduce =
    match sum_or_mean with
    | `sum -> reduce_sum
    | `mean -> reduce_mean
  in
  Ops_generated.(neg (ys * log (y_hats + f_or_d ~type_ epsilon)) |> reduce)

let binary_cross_entropy ?(epsilon = 1e-7) ~labels ~model_values sum_or_mean =
  let type_ = Node.output_type model_values in
  let model_shape = Node.shape model_values in
  if List.last_exn model_shape <> 1
  then
    raise
      (Invalid_argument
         (Printf.sprintf
            "the last dimension should be 1 %s"
            (string_of_shape model_shape)));
  let reduce =
    match sum_or_mean with
    | `sum -> reduce_sum
    | `mean -> reduce_mean
  in
  Ops_generated.(
    neg
      ((labels * log (model_values + f_or_d ~type_ epsilon))
      + ((f_or_d ~type_ 1. - labels) * log (f_or_d ~type_ (1. +. epsilon) - model_values))
      ))
  |> reduce

let leaky_relu xs ~alpha =
  let type_ = Node.output_type xs in
  Ops_generated.maximum xs (f_or_d ~type_ alpha * xs)
