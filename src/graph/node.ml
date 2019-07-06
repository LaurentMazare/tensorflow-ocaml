open Base
open Tensorflow_core

module Session = struct
  type t =
    { session : Wrapper.Session.t
    ; graph : Wrapper.Graph.t (* This list is always topologically sorted. *)
    ; mutable variable_initializations : Wrapper.Graph.operation list
    }

  let create () =
    let graph = Wrapper.Graph.create () in
    match Wrapper.Session.create graph with
    | Error status ->
      Printf.failwithf
        "Unable to generate session: %s"
        (Wrapper.Status.message status)
        ()
    | Ok session -> { session; graph; variable_initializations = [] }

  let default_lazy = lazy (create ())
  let default () = Lazy.force default_lazy
  let default_session () = (default ()).session
  let default_graph () = (default ()).graph

  let get_and_reset_variable_initializations () =
    let t = default () in
    let variable_initializations = t.variable_initializations in
    t.variable_initializations <- [];
    List.rev variable_initializations

  let add_variable_initialization op =
    let t = default () in
    t.variable_initializations <- op :: t.variable_initializations
end

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

module Id_table = Hashtbl.M (Id)
module Type = Operation.Type
module Tensor = Operation.Tensor_attr
module Dim = Operation.Dim
module Attr_list = Operation.Attr_list

type attr = Operation.attr
type output = Wrapper.Graph.output

type 'a t =
  { id : Id.t
  ; name : Name.t
  ; op_name : Op_name.t
  ; output_type : 'a Type.t
  ; inputs : input list
  ; control_inputs : p list
  ; attributes : (string * attr) list
  ; output_idx : int option (* Only used for multiple outputs. *)
  ; operation : Operation.t
  ; output : output
  }

and p = P : _ t -> p

and input =
  [ `single of p
  | `multi of p list
  ]

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
let operation t = t.operation
let output t = t.output
let packed_name (P t) = t.name
let packed_inputs (P t) = t.inputs
let packed_flat_inputs (P t) = flat_inputs t
let packed_op_name (P t) = t.op_name

let packed_is_real (P t) =
  match t.output_type with
  | Type.Unit -> false
  | Type.Variant -> false
  | Type.Int32 -> false
  | Type.Int64 -> false
  | Type.Bool -> false
  | Type.String -> false
  | Type.Complex64 -> false
  | Type.Float -> true
  | Type.Double -> true

let packed_id (P t) = t.id
let packed_output_idx (P t) = t.output_idx
let packed_operation (P t) = t.operation
let packed_output (P t) = t.output

let create ~name ~op_name ~output_type ~inputs ~control_inputs ~attributes ~output_idx =
  let id = Id.create () in
  let op_inputs, op_input_lists =
    List.partition_map inputs ~f:(function
        | `single input ->
          `Fst
            (packed_operation input, packed_output_idx input |> Option.value ~default:0)
        | `multi inputs ->
          let input_lists =
            List.map inputs ~f:(fun input ->
                let index = packed_output_idx input |> Option.value ~default:0 in
                packed_operation input, index)
          in
          `Snd input_lists)
  in
  let operation =
    Operation.create
      (Session.default_graph ())
      ~op_name:(Op_name.to_string op_name)
      ~unique_name:(Printf.sprintf "%s-%s" (Name.to_string name) (Id.to_string id))
      ~control_inputs:(List.map control_inputs ~f:packed_operation)
      ~inputs:op_inputs
      ~input_lists:op_input_lists
      ~attributes
  in
  let output =
    Wrapper.Graph.create_output operation ~index:(Option.value output_idx ~default:0)
  in
  { id
  ; name
  ; op_name
  ; output_type
  ; inputs
  ; control_inputs
  ; attributes
  ; output_idx
  ; operation
  ; output
  }

let create_gradient output ~output_type =
  let id = Id.create () in
  let operation, output_idx =
    Wrapper.Graph.output_op_and_index (Session.default_graph ()) output
  in
  { id
  ; name = Name.of_string "gradient"
  ; op_name = Op_name.of_string "gradient"
  ; output_type
  ; inputs = []
  ; control_inputs = []
  ; attributes = []
  ; output_idx = Some output_idx
  ; operation
  ; output
  }

let get_attr t str = List.Assoc.find ~equal:String.equal t.attributes str

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

let shape t =
  Wrapper.Graph.shape (Session.default_graph ()) t.output |> Wrapper.Status.ok_exn

let set_output_idx t output_idx = { t with output_idx }

let set_output_idx_and_output_type t output_idx ~type_ =
  { t with output_idx; output_type = type_ }

let extract : type a. p -> a Type.t -> a t option =
 fun p type_ ->
  let (P t) = p in
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

let extract_exn p type_ = Option.value_exn (extract p type_)
