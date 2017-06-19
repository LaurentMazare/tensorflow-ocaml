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

  let to_data_type = function
    | P Unit -> assert false
    | P Float -> Wrapper.TF_FLOAT
    | P Double -> TF_DOUBLE
    | P Int32 -> TF_INT32
    | P Int64 -> TF_INT64
    | P Complex64 -> TF_COMPLEX
    | P Bool -> TF_BOOL
    | P String -> TF_STRING

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
module Tensor_attr = struct
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

type t = Wrapper.Graph.operation

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
  | Tensor_float of float Tensor_attr.t
  | Tensor_int of int Tensor_attr.t
  | Tensor_string of string Tensor_attr.t
  | Shape of Dim.t list

let add_attribute operation_description ~attr_name attr =
  match (attr : attr) with
  | String str ->
    Wrapper.Graph.set_attr_string operation_description ~attr_name str
  | Type dtype ->
    let dtype = Type.to_data_type dtype in
    Wrapper.Graph.set_attr_type operation_description ~attr_name dtype
  | Tensor_float tensor_float ->
    let set_attr kind =
      let tensor = Tensor.create kind (Array.of_list tensor_float.shape) in
      Tensor.copy_elt_list tensor tensor_float.values;
      Wrapper.Graph.set_attr_tensor operation_description ~attr_name (Tensor.P tensor)
      |> Wrapper.Status.ok_exn
    in
    begin
      match tensor_float.type_ with
      | Type.P Type.Float -> set_attr Float32
      | Type.P Type.Double -> set_attr Float64
      | Type.P _ -> assert false
    end
  | Tensor_int tensor_int ->
    let tensor =
      match tensor_int.type_ with
      | Type.P Type.Int32 ->
        let tensor = Tensor.create Int32 (Array.of_list tensor_int.shape) in
        Tensor.copy_elt_list tensor (List.map Int32.of_int tensor_int.values);
        Tensor.P tensor
      | Type.P Type.Int64 ->
        let tensor = Tensor.create Int64 (Array.of_list tensor_int.shape) in
        Tensor.copy_elt_list tensor (List.map Int64.of_int tensor_int.values);
        Tensor.P tensor
      | Type.P _ -> assert false
    in
    Wrapper.Graph.set_attr_tensor operation_description ~attr_name tensor
    |> Wrapper.Status.ok_exn
  | Int i ->
    Wrapper.Graph.set_attr_int operation_description ~attr_name i
  | Float f ->
    Wrapper.Graph.set_attr_float operation_description ~attr_name f
  | Bool b ->
    Wrapper.Graph.set_attr_bool operation_description ~attr_name b
  | Shape shape ->
    let shape = List.map (fun dim -> dim.Dim.size) shape in
    Wrapper.Graph.set_attr_shape operation_description ~attr_name shape
  | List (Int is) ->
    Wrapper.Graph.set_attr_int_list operation_description ~attr_name is
  | List (Float fs) ->
    Wrapper.Graph.set_attr_float_list operation_description ~attr_name fs
  | List (Bool bs) ->
    Wrapper.Graph.set_attr_bool_list operation_description ~attr_name bs
  | List (Type dtypes) ->
    let dtypes = List.map Type.to_data_type dtypes in
    Wrapper.Graph.set_attr_type_list operation_description ~attr_name dtypes
  | List (String _) -> failwith "List String attributes are not supported yet."
  | List (Shape _) -> failwith "List Shape attributes are not supported yet."
  | Tensor_string tensor_str ->
    Wrapper.Graph.set_attr_tensor_string operation_description tensor_str.values
      ~attr_name
      ~shape:tensor_str.shape
    |> Wrapper.Status.ok_exn

let create
      graph
      ~op_name
      ~unique_name
      ~inputs
      ~input_lists
      ~control_inputs
      ~attributes
  =
  let operation_description =
    Wrapper.Graph.new_operation graph
      ~op_name
      ~name:unique_name
  in
  List.iter
    (fun control_input ->
      Wrapper.Graph.add_control_input operation_description control_input)
    control_inputs;
  List.iter
    (fun (input, output_index) ->
      Wrapper.Graph.add_input
        operation_description
        input
        ~index:output_index)
    inputs;
  List.iter
    (fun inputs ->
      Wrapper.Graph.add_inputs operation_description inputs)
    input_lists;
  List.iter
    (fun (attr_name, attr) ->
      add_attribute operation_description ~attr_name attr)
    attributes;
  Wrapper.Graph.finish_operation operation_description
  |> Wrapper.Status.ok_exn
