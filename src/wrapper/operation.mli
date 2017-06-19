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

module Tensor_attr : sig
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
  | Tensor_float of float Tensor_attr.t
  | Tensor_int of int Tensor_attr.t
  | Tensor_string of string Tensor_attr.t
  | Shape of Dim.t list

type t = Wrapper.Graph.operation

val create
  :  Wrapper.Graph.t
  -> op_name:string
  -> unique_name:string
  -> inputs:(t * int) list
  -> input_lists:(t * int) list list
  -> control_inputs:t list
  -> attributes:(string * attr) list
  -> t

