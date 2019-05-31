open Base
open Tensorflow_core

module Session : sig
  val default_session : unit -> Wrapper.Session.t
  val default_graph : unit -> Wrapper.Graph.t
  val get_and_reset_variable_initializations : unit -> Operation.t list
  val add_variable_initialization : Operation.t -> unit
end

module Op_name : Identifiable.S
module Name : Identifiable.S
module Id : Identifiable.S
module Type = Operation.Type
module Tensor = Operation.Tensor_attr
module Dim = Operation.Dim
module Attr_list = Operation.Attr_list

type attr = Operation.attr
type output = Wrapper.Graph.output
type 'a t
type p = P : _ t -> p

type input =
  [ `single of p
  | `multi of p list
  ]

val create
  :  name:Name.t
  -> op_name:Op_name.t
  -> output_type:'a Type.t
  -> inputs:input list
  -> control_inputs:p list
  -> attributes:(string * attr) list
  -> output_idx:int option (* Only used for multiple outputs. *)
  -> 'a t

val create_gradient : output -> output_type:'a Type.t -> 'a t
val name : _ t -> Name.t
val op_name : _ t -> Op_name.t
val output_type : 'a t -> 'a Type.t
val inputs : _ t -> input list
val flat_inputs : _ t -> p list
val control_inputs : _ t -> p list
val attributes : _ t -> (string * attr) list
val output_idx : _ t -> int option
val operation : _ t -> Operation.t
val output : _ t -> output
val packed_name : p -> Name.t
val packed_op_name : p -> Op_name.t
val packed_inputs : p -> input list
val packed_flat_inputs : p -> p list
val packed_is_real : p -> bool
val packed_id : p -> Id.t
val packed_output_idx : p -> int option
val packed_operation : p -> Operation.t
val packed_output : p -> output
val get_attr_bool : _ t -> string -> bool option
val get_attr_string : _ t -> string -> string option
val get_attr_int : _ t -> string -> int option
val get_attr_int_list : _ t -> string -> int list option
val shape : _ t -> int list
val set_output_idx : 'a t -> int option -> 'a t

(* This is a very unsafe function to use. *)
val set_output_idx_and_output_type : 'b t -> int option -> type_:'a Type.t -> 'a t
val id : _ t -> Id.t
val extract : p -> 'a Type.t -> 'a t option
val extract_exn : p -> 'a Type.t -> 'a t
