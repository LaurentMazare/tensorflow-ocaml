open Base
open Tensorflow_core

module Op_name : Identifiable.S = String

module Name : Identifiable.S = String

module Id : sig
  include Identifiable.S
  val create : unit -> t
end = struct
  include Int
  let create =
    let counter = ref 0 in
    fun () ->
      incr counter;
      !counter
end
module Id_table = Hashtbl.M(Id)
module Type = Operation.Type
module Tensor = Operation.Tensor_attr
module Dim = Operation.Dim
module Attr_list = Operation.Attr_list
type attr = Operation.attr

type 'a t =
  { id : Id.t
  ; name : Name.t
  ; op_name : Op_name.t
  ; output_type : 'a Type.t
  ; inputs : input list
  ; control_inputs : p list
  ; attributes : (string * attr) list
  ; output_idx : int option (* Only used for multiple outputs. *)
  }
and p = P : _ t -> p
and input = [ `single of p | `multi of p list ]

let create
      ~name
      ~op_name
      ~output_type
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx
  =
  { id = Id.create ()
  ; name
  ; op_name
  ; output_type
  ; inputs
  ; control_inputs
  ; attributes
  ; output_idx
  }

let id t = t.id
let name t = t.name
let op_name t = t.op_name
let output_type t = t.output_type
let inputs t = t.inputs
let flat_inputs t =
  List.concat_map t.inputs ~f:(function
    | `single p -> [ p ]
    | `multi ps -> ps)

let control_inputs t = t.control_inputs
let attributes t = t.attributes
let output_idx t = t.output_idx
let unique_name t =
  Printf.sprintf "%s-%s" (Name.to_string t.name) (Id.to_string t.id)

let packed_name (P t) = t.name
let packed_inputs (P t) = t.inputs
let packed_flat_inputs (P t) = flat_inputs t
let packed_op_name (P t) = t.op_name
let packed_is_real (P t) =
  match t.output_type with
  | Type.Unit -> false
  | Type.Int32 -> false
  | Type.Int64 -> false
  | Type.Bool -> false
  | Type.String -> false
  | Type.Complex64 -> false
  | Type.Float -> true
  | Type.Double -> true

let packed_id (P t) = t.id
let packed_output_idx (P t) = t.output_idx

let get_attr t str =
  List.Assoc.find ~equal:String.equal t.attributes str

let get_attr_bool t str =
  Option.bind (get_attr t str) ~f:(function
    | Bool b -> Some b
    | _ -> None)

let get_attr_string t str =
  Option.bind (get_attr t str) ~f:(function
    | String s -> Some s
    | _ -> None)

let get_attr_int t str =
  Option.bind (get_attr t str) ~f:(function
    | Int l -> Some l
    | _ -> None)

let get_attr_int_list t str =
  Option.bind (get_attr t str) ~f:(function
    | List (Int l) -> Some l
    | _ -> None)

let get_shape t =
  Option.bind (get_attr t "shape") ~f:(function
    | Shape shape -> Some shape
    | _ -> None)

let set_output_idx t output_idx = { t with output_idx }

let set_output_idx_and_output_type t output_idx ~type_ =
  { t with output_idx; output_type = type_ }

let extract : type a . p -> a Type.t -> a t option = fun p type_ ->
  let P t = p in
  match t.output_type, type_ with
  | Type.Unit, Type.Unit -> Some t
  | Type.Int32, Type.Int32 -> Some t
  | Type.Int64, Type.Int64 -> Some t
  | Type.Bool, Type.Bool -> Some t
  | Type.String, Type.String -> Some t
  | Type.Complex64, Type.Complex64 -> Some t
  | Type.Float, Type.Float -> Some t
  | Type.Double, Type.Double -> Some t
  | _, _ -> None

let extract_exn p type_ =
  Option.value_exn (extract p type_)

(* TODO noury: actually make weak *)
module Weak_table = struct
  type 'a node = 'a t
  type t = p Id_table.t

  let create () =
    Hashtbl.create (module Id) ()

  let set t ~key ~data =
    Hashtbl.set t ~key:(id key) ~data:(P data)

  let find t key =
    match Hashtbl.find t (id key) with
    | None -> None
    | Some packed_node ->
      extract packed_node (output_type key)
end
