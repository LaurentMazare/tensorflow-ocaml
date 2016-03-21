open Core.Std

module Name : sig
  include Identifiable
  val make_fresh : name:string -> t
  val to_string : t -> string
end = struct
  include String
  let cnt = ref 0
  let make_fresh ~name =
    incr cnt;
    Printf.sprintf "%s-%d" name !cnt

  let to_string = Fn.id
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

(* This is used for float/double, maybe we should introduce another GADT to handle this
   in a generic way ? *)
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
  | Shape of Dim.t list

type 'a t =
  { name : Name.t
  ; op_name : string
  ; output_type : 'a Type.t
  ; inputs : p list
  ; attributes : (string * attr) list
  ; output_name : string option (* Only used for multiple outputs. *)
  }
and p = P : _ t -> p
