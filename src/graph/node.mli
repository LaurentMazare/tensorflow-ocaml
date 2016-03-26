open Core_kernel.Std

module Op_name : Identifiable

module Name : sig
  include Identifiable
  val make_fresh : name:string -> t
end

module Type : sig
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

  val to_string : p -> string

  val of_dt_type
    :  [> `dt_bool
       | `dt_complex64
       | `dt_double
       | `dt_float
       | `dt_int32
       | `dt_int64
       | `dt_string
       ]
    -> p option

  val to_dt_type
    :  p
    -> [> `dt_bool
       | `dt_complex64
       | `dt_double
       | `dt_float
       | `dt_int32
       | `dt_int64
       | `dt_string
       ]
end

module Tensor : sig
  type 'a t =
    { type_ : Type.p (* Has to be Float or Double. *)
    ; shape : int list
    ; values : 'a list
    }
end

module Dim : sig
  type t =
    { size : int
    ; name : string option
    }

  val create : ?name:string -> int -> t
end

module Attr_list : sig
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
  ; op_name : Op_name.t
  ; output_type : 'a Type.t
  ; inputs : p list
  ; attributes : (string * attr) list
  ; output_idx : int option (* Only used for multiple outputs. *)
  }
and p = P : _ t -> p

module Id :
sig
  include Identifiable
end

val packed_name : p -> Name.t
val packed_op_name : p -> Op_name.t
val packed_inputs : p -> p list
val packed_is_real : p -> bool
val packed_id : p -> Id.t

val id : _ t -> Id.t

val extract : p -> 'a Type.t -> 'a t option

module Weak_table :
sig
  type 'a t
  val create : unit -> 'a t
  val set : 'a t -> key:p -> data:'a -> unit
  val find : 'a t -> p -> 'a option
end
