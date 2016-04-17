open Core_kernel.Std

module Op_name : Identifiable = String_id

module Name : Identifiable = String_id

module Id : sig
  include Identifiable
  val create : unit -> t
end = struct
  include Int
  let create =
    let counter = ref 0 in
    fun () ->
      incr counter;
      !counter
end

module Type = struct
  (* We rely on all variants to be of the form | Variant : [ `variant ] t. *)
  type _ t =
    | Unit : [ `unit ] t
    | Float : [ `float ] t
    | Double : [ `double ] t
    | Int32 : [ `int32 ] t
    | Int64 : [ `int64 ] t
    | Complex64 : [ `complex64 ] t
    | Bool : [ `bool ] t
    | String : [ `string ] t

  type p = P : _ t -> p

  let to_dt_type = function
    | P Unit -> assert false
    | P Float -> `dt_float
    | P Double -> `dt_double
    | P Int32 -> `dt_int32
    | P Int64 -> `dt_int64
    | P Complex64 -> `dt_complex64
    | P Bool -> `dt_bool
    | P String -> `dt_string

  let of_dt_type = function
    | `dt_float -> Some (P Float)
    | `dt_double -> Some (P Double)
    | `dt_int32 -> Some (P Int32)
    | `dt_int64 -> Some (P Int64)
    | `dt_complex64 -> Some (P Complex64)
    | `dt_bool -> Some (P Bool)
    | `dt_string -> Some (P String)
    | _ -> None

  let to_string = function
    | P Unit -> "Unit"
    | P Float -> "Float"
    | P Double -> "Double"
    | P Int32 -> "Int32"
    | P Int64 -> "Int64"
    | P Complex64 -> "Complex64"
    | P Bool -> "Bool"
    | P String -> "String"
end

(* This is used for float/double/string, maybe we should introduce another GADT
   to handle this in a generic way ? *)
module Tensor = struct
  type 'a t =
    { type_ : Type.p (* Has to be Float or Double. *)
    ; shape : int list
    ; values : 'a list
    }
end

module Dim = struct
  type t =
    { size : int
    ; name : string option
    }

  let create ?name size = { size; name }
end

module Attr_list = struct
  type t =
    | String of string list
    | Int of int list
    | Float of float list
    | Bool of bool list
    | Type of Type.p list
    | Shape of Dim.t list list
end

type attr =
  | String of string
  | Int of int
  | Float of float
  | Bool of bool
  | Type of Type.p
  | List of Attr_list.t
  | Tensor_float of float Tensor.t
  | Tensor_int of int Tensor.t
  | Tensor_string of string Tensor.t
  | Shape of Dim.t list

type 'a t =
  { id : Id.t
  ; name : Name.t
  ; op_name : Op_name.t
  ; output_type : 'a Type.t
  ; inputs : p list
  ; attributes : (string * attr) list
  ; output_idx : int option (* Only used for multiple outputs. *)
  }
and p = P : _ t -> p

let create
      ~name
      ~op_name
      ~output_type
      ~inputs
      ~attributes
      ~output_idx
  =
  { id = Id.create ()
  ; name
  ; op_name
  ; output_type
  ; inputs
  ; attributes
  ; output_idx
  }

let id t = t.id
let name t = t.name
let op_name t = t.op_name
let output_type t = t.output_type
let inputs t = t.inputs
let attributes t = t.attributes
let output_idx t = t.output_idx
let unique_name t =
  sprintf "%s-%s" (Name.to_string t.name) (Id.to_string t.id)

let packed_name (P t) = t.name
let packed_inputs (P t) = t.inputs
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
  List.Assoc.find t.attributes str

let get_attr_bool t str =
  Option.bind (get_attr t str) (function
    | Bool b -> Some b
    | _ -> None)

let get_attr_string t str =
  Option.bind (get_attr t str) (function
    | String s -> Some s
    | _ -> None)

let get_attr_int_list t str =
  Option.bind (get_attr t str) (function
    | List (Int l) -> Some l
    | _ -> None)

let get_shape t =
  Option.bind (get_attr t "shape") (function
    | Shape shape -> Some shape
    | _ -> None)

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

(* CR noury: actually make weak *)
module Weak_table = struct
  type 'a node = 'a t
  type t = p Id.Table.t

  let create () =
    Id.Table.create ()

  let set t ~key ~data =
    Id.Table.set t ~key:(id key) ~data:(P data)

  let find t key =
    match Id.Table.find t (id key) with
    | None -> None
    | Some packed_node ->
      extract packed_node (output_type key)
end
