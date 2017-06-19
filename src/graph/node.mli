open Base
open Tensorflow_core

module Op_name : Identifiable.S

module Name : Identifiable.S

module Id : Identifiable.S

module Type = Operation.Type
module Tensor = Operation.Tensor_attr
module Dim = Operation.Dim
module Attr_list = Operation.Attr_list
type attr = Operation.attr

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
