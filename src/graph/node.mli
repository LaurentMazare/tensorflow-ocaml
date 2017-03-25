open Base
open Tensorflow_core

module Op_name : Identifiable.S

module Name : Identifiable.S

module Id : Identifiable.S

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

  val to_data_type : p -> Wrapper.data_type
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
  | Tensor_string of string Tensor.t
  | Shape of Dim.t list

type 'a t
type p = P : _ t -> p

type input = [ `single of p | `multi of p list ]

val create
  :  name:Name.t
  -> op_name:Op_name.t
  -> output_type:'a Type.t
  -> inputs:input list
  -> control_inputs:p list
  -> attributes:(string * attr) list
  -> output_idx:int option (* Only used for multiple outputs. *)
  -> 'a t

val name : _ t -> Name.t
val op_name : _ t -> Op_name.t
val output_type : 'a t -> 'a Type.t
val inputs : _ t -> input list
val flat_inputs : _ t -> p list
val control_inputs : _ t -> p list
val attributes : _ t -> (string * attr) list
val output_idx : _ t -> int option
val unique_name : _ t -> string

val packed_name : p -> Name.t
val packed_op_name : p -> Op_name.t
val packed_inputs : p -> input list
val packed_flat_inputs : p -> p list
val packed_is_real : p -> bool
val packed_id : p -> Id.t
val packed_output_idx : p -> int option

val get_attr_bool : _ t -> string -> bool option
val get_attr_string : _ t -> string -> string option
val get_attr_int : _ t -> string -> int option
val get_attr_int_list : _ t -> string -> int list option
val get_shape : _ t -> Dim.t list option

val set_output_idx : 'a t -> int option -> 'a t
(* This is a very unsafe function to use. *)
val set_output_idx_and_output_type
  :  'b t
  -> int option
  -> type_:'a Type.t
  -> 'a t

val id : _ t -> Id.t

val extract : p -> 'a Type.t -> 'a t option
val extract_exn : p -> 'a Type.t -> 'a t

module Weak_table : sig
  type 'a node = 'a t
  type t
  val create : unit -> t
  val set : t -> key:'a node -> data:'a node -> unit
  val find : t -> 'a node -> 'a node option
end
