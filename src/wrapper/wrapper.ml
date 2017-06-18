open Ctypes

module C = Tf_bindings.C(Tf_generated)
open C
let verbose = false
let force_full_major = false

type data_type =
  | TF_FLOAT
  | TF_DOUBLE
  | TF_INT32
  | TF_UINT8
  | TF_INT16
  | TF_INT8
  | TF_STRING
  | TF_COMPLEX
  | TF_INT64
  | TF_BOOL
  | TF_QINT8
  | TF_QUINT8
  | TF_QINT32
  | TF_BFLOAT16
  | TF_QINT16
  | TF_QUINT16
  | TF_UINT16
  | Unknown of int

let data_type_to_int = function
  | TF_FLOAT -> 1
  | TF_DOUBLE -> 2
  | TF_INT32 -> 3
  | TF_UINT8 -> 4
  | TF_INT16 -> 5
  | TF_INT8 -> 6
  | TF_STRING -> 7
  | TF_COMPLEX -> 8
  | TF_INT64 -> 9
  | TF_BOOL -> 10
  | TF_QINT8 -> 11
  | TF_QUINT8 -> 12
  | TF_QINT32 -> 13
  | TF_BFLOAT16 -> 14
  | TF_QINT16 -> 15
  | TF_QUINT16 -> 16
  | TF_UINT16 -> 17
  | Unknown n -> n

let int_to_data_type = function
  | 1 -> TF_FLOAT
  | 2 -> TF_DOUBLE
  | 3 -> TF_INT32
  | 4 -> TF_UINT8
  | 5 -> TF_INT16
  | 6 -> TF_INT8
  | 7 -> TF_STRING
  | 8 -> TF_COMPLEX
  | 9 -> TF_INT64
  | 10 -> TF_BOOL
  | 11 -> TF_QINT8
  | 12 -> TF_QUINT8
  | 13 -> TF_QINT32
  | 14 -> TF_BFLOAT16
  | 15 -> TF_QINT16
  | 16 -> TF_QUINT16
  | 17 -> TF_UINT16
  | n -> Unknown n

module Tensor = struct
  open C.Tf_tensor
  let sizeof = function
    | TF_FLOAT -> 4
    | TF_DOUBLE -> 8
    | TF_INT32 -> 4
    | TF_UINT16
    | TF_INT16 -> 2
    | TF_BOOL
    | TF_UINT8
    | TF_INT8 -> 1
    | TF_INT64 -> 8
    | TF_STRING
    | TF_COMPLEX
    | TF_QINT8
    | TF_QUINT8
    | TF_QINT32
    | TF_BFLOAT16
    | TF_QINT16
    | TF_QUINT16
    | Unknown _ -> failwith "Unsupported tensor type (sizeof)"

  let data_type_of_kind (type a) (type b) (kind : (a, b) Bigarray.kind) =
    match kind with
    | Bigarray.Float32 -> TF_FLOAT
    | Bigarray.Float64 -> TF_DOUBLE
    | Bigarray.Int64 -> TF_INT64
    | Bigarray.Int32 -> TF_INT32
    | Bigarray.Int8_unsigned -> TF_BOOL
    | _ -> failwith "Unsupported yet"

  module Id : sig
    type t
    val create : unit -> t
    val to_int : t -> int
    val of_int : int -> t
  end = struct
    type t = int
    let create =
      let cnt = ref 0 in
      fun () -> incr cnt; !cnt

    let to_int t = t
    let of_int t = t
  end

  let live_tensors = Hashtbl.create 1024
  let deallocate _ _ id =
    let id = raw_address_of_ptr id |> Nativeint.to_int |> Id.of_int in
    if verbose
    then Printf.printf "Deallocating tensor %d\n%!" (Id.to_int id);
    Hashtbl.remove live_tensors id

  let deallocate =
    coerce
      (Foreign.funptr (ptr void @-> size_t @-> ptr void @-> returning void))
      (static_funptr (ptr void @-> size_t @-> ptr void @-> returning void))
      deallocate

  let c_tensor_of_tensor packed_tensor =
    let Tensor.P tensor = packed_tensor in
    let id = Id.create () in
    let dim_array = Tensor.dims tensor in
    let dims =
      Array.to_list dim_array
      |> List.map Int64.of_int
      |> CArray.of_list int64_t
      |> CArray.start
    in
    let data_type = Tensor.kind tensor |> data_type_of_kind in
    let size = Array.fold_left ( * ) 1 dim_array * sizeof data_type in
    Hashtbl.add live_tensors id (`tensor packed_tensor);
    let num_dims = Tensor.num_dims tensor in
    let start =
      bigarray_start genarray (Tensor.to_bigarray tensor)
      |> to_voidp
    in
    if force_full_major
    then Gc.full_major ();
    tf_newtensor (data_type_to_int data_type)
      dims
      num_dims
      start
      (Unsigned.Size_t.of_int size)
      deallocate
      (Id.to_int id |> Nativeint.of_int |> ptr_of_raw_address)

  let tensor_of_c_tensor c_tensor =
    let num_dims = tf_numdims c_tensor in
    let dims = Array.init num_dims (fun i -> tf_dim c_tensor i) in
    let dims = (* Scalar hack: use a 1d bigarray as 0d bigarray are not supported. *)
      if Array.length dims = 0
      then [| 1 |]
      else dims
    in
    let apply_kind kind =
      let data = tf_tensordata c_tensor |> from_voidp (typ_of_bigarray_kind kind) in
      let data = bigarray_of_ptr genarray dims kind data in
      Gc.finalise (fun _data -> tf_deletetensor c_tensor) data;
      Tensor.P (Tensor.of_bigarray data ~scalar:(num_dims = 0))
    in
    match tf_tensortype c_tensor |> int_to_data_type with
    | TF_FLOAT -> apply_kind Bigarray.float32
    | TF_DOUBLE -> apply_kind Bigarray.float64
    | TF_INT32 -> apply_kind Bigarray.int32
    | TF_INT64 -> apply_kind Bigarray.int64
    | TF_BOOL -> apply_kind Bigarray.int8_unsigned
    | _ -> failwith "Unsupported tensor type"

  let varint_len int =
    let rec loop acc int =
      if int < 128
      then acc + 1
      else loop (acc + 1) (int / 128)
    in
    loop 0 int

  let c_tensor_of_strings strings ~shape =
    let nstrings = List.length strings in
    let bigarray, size =
      let start_offset_len = nstrings * 8 in
      let data_len, offsets =
        List.fold_left
          (fun (acc_length, offsets) str ->
            let length = String.length str in
            acc_length + length + varint_len length, acc_length :: offsets)
          (0, [])
          strings
      in
      let offsets = List.rev offsets |> Array.of_list in
      let bigarray =
        Bigarray.Array1.create Char Bigarray.c_layout (start_offset_len + data_len)
      in
      List.iteri
        (fun i str ->
          let offset = offsets.(i) in
          let length = String.length str in
          let rec loop acc length =
            bigarray.{ start_offset_len + offset + acc } <- Char.chr (length mod 128);
            if length < 128
            then acc + 1
            else loop (acc + 1) (length / 128)
          in
          let varint_len = loop 0 length in
          for j = 0 to length - 1 do
            bigarray.{ start_offset_len + offset + varint_len + j } <- str.[j]
          done;
          let offset = ref offset in
          for j = 0 to 7 do
            bigarray.{8*i + j} <- Char.chr (!offset mod 256);
            offset := !offset / 256;
          done)
        strings;
      Bigarray.genarray_of_array1 bigarray, Bigarray.Array1.dim bigarray
    in
    let id = Id.create () in
    let dims =
      List.map Int64.of_int shape
      |> CArray.of_list int64_t
      |> CArray.start
    in
    Hashtbl.add live_tensors id (`string_tensor bigarray);
    let start = bigarray_start genarray bigarray |> to_voidp in
    if force_full_major
    then Gc.full_major ();
    tf_newtensor (data_type_to_int TF_STRING)
      dims
      (List.length shape)
      start
      (Unsigned.Size_t.of_int size)
      deallocate
      (Id.to_int id |> Nativeint.of_int |> ptr_of_raw_address)
end

module Status = struct
  include C.Tf_status

  type code =
    | TF_OK
    | TF_CANCELLED
    | TF_UNKNOWN
    | TF_INVALID_ARGUMENT
    | TF_DEADLINE_EXCEEDED
    | TF_NOT_FOUND
    | TF_ALREADY_EXISTS
    | TF_PERMISSION_DENIED
    | TF_UNAUTHENTICATED
    | TF_RESOURCE_EXHAUSTED
    | TF_FAILED_PRECONDITION
    | TF_ABORTED
    | TF_OUT_OF_RANGE
    | TF_UNIMPLEMENTED
    | TF_INTERNAL
    | TF_UNAVAILABLE
    | TF_DATA_LOSS
    | Unknown of int

  (* CAUTION: this has to stay in sync with [tensor_c_api.h], maybe we should generate
     some stubs to assert this at compile time. *)
  let code_of_int = function
    | 0  -> TF_OK
    | 1  -> TF_CANCELLED
    | 2  -> TF_UNKNOWN
    | 3  -> TF_INVALID_ARGUMENT
    | 4  -> TF_DEADLINE_EXCEEDED
    | 5  -> TF_NOT_FOUND
    | 6  -> TF_ALREADY_EXISTS
    | 7  -> TF_PERMISSION_DENIED
    | 16 -> TF_UNAUTHENTICATED
    | 8  -> TF_RESOURCE_EXHAUSTED
    | 9  -> TF_FAILED_PRECONDITION
    | 10 -> TF_ABORTED
    | 11 -> TF_OUT_OF_RANGE
    | 12 -> TF_UNIMPLEMENTED
    | 13 -> TF_INTERNAL
    | 14 -> TF_UNAVAILABLE
    | 15 -> TF_DATA_LOSS
    | n  -> Unknown n

  let create () =
    let status = tf_newstatus () in
    Gc.finalise tf_deletestatus status;
    status

  let code t = tf_getcode t |> code_of_int

  let message = tf_message

  type 'a result =
    | Ok of 'a
    | Error of t

  let result_or_error status v =
    match code status with
    | TF_OK -> Ok v
    | _ -> Error status

  let ok_exn = function
    | Ok ok -> ok
    | Error status ->
      failwith
        (Printf.sprintf "%d %s" (tf_getcode status) (message status))
end

module Buffer = struct
  module Tf_buffer = C.Tf_buffer
  type t = Tf_buffer.t

  let create_from_string str =
    let t = Tf_buffer.tf_newbufferfromstring str (String.length str) in
    Gc.finalise Tf_buffer.tf_deletebuffer t;
    (t : t)
end

module Graph_import = struct
  module Tf_import_graph_def_options = C.Tf_import_graph_def_options
  type import_options = Tf_import_graph_def_options.t

  let create_import_options () =
    let import_options = Tf_import_graph_def_options.tf_newimportgraphdefoptions () in
    Gc.finalise Tf_import_graph_def_options.tf_deleteimportgraphdefoptions import_options;
    (import_options : import_options)

  let import graph str =
    let status = Status.create () in
    let import_options = create_import_options () in
    let buffer = Buffer.create_from_string str in
    C.Tf_graph.tf_graphimportgraphdef graph buffer import_options status;
    Status.result_or_error status ()
end

module Graph = struct
  open C
  type t = Tf_graph.t

  let create () =
    let t = Tf_graph.tf_newgraph () in
    Gc.finalise Tf_graph.tf_deletegraph t;
    t

  let keep_alive t =
    if to_voidp t = null
    then failwith "null pointer"

  type operation = Tf_graph.t * Tf_operation.t

  (* Keep a reference to the graph as it should not be deleted before tf_finishoperation
     has been succesfully called. *)
  type operation_description = Tf_graph.t * Tf_operationdescription.t

  type output = Tf_output.t structure

  let live_strings = ref []

  let ptr_of_string str =
    let len = String.length str in
    let carray = CArray.make Ctypes.char (1 + len) in
    live_strings := carray :: !live_strings;
    String.iteri (fun i char -> CArray.set carray i char) str;
    CArray.set carray len '\x00';
    CArray.start carray

  let create_output (_graph, op) ~index =
    let output = make Tf_output.t in
    setf output Tf_output.oper op;
    setf output Tf_output.index index;
    output

  let new_operation t ~op_name ~name =
    let od =
      Tf_operationdescription.tf_newoperation
        t
        (ptr_of_string op_name)
        (ptr_of_string name)
    in
    t, od

  let finish_operation (graph, od) =
    let status = Status.create () in
    let operation = Tf_operationdescription.tf_finishoperation od status in
    Status.result_or_error status (graph, operation)

  let add_control_input (graph, od) (graph', op)=
    if graph != graph'
    then failwith "Calling add_input on different graphs.";
    Tf_operationdescription.tf_addcontrolinput od op;
    keep_alive graph

  let add_input (graph, od) (graph', op) ~index =
    if graph != graph'
    then failwith "Calling add_input on different graphs.";
    let output = create_output (graph, op) ~index in
    Tf_operationdescription.tf_addinput od output;
    keep_alive graph

  let add_inputs (graph, od) op_and_indexes =
    let outputs =
      List.map (fun (op, index) -> create_output op ~index)
        op_and_indexes
      |> CArray.of_list Tf_output.t
      |> CArray.start
    in
    Tf_operationdescription.tf_addinputlist od outputs (List.length op_and_indexes);
    keep_alive graph

  let set_attr_int (graph, od) ~attr_name value =
    Tf_operationdescription.tf_setattrint
      od
      (ptr_of_string attr_name)
      (Int64.of_int value);
    keep_alive graph

  let set_attr_int_list (graph, od) ~attr_name values =
    let values = List.map Int64.of_int values in
    Tf_operationdescription.tf_setattrintlist
      od
      (ptr_of_string attr_name)
      CArray.(of_list int64_t values |> start)
      (List.length values);
    keep_alive graph

  let set_attr_float (graph, od) ~attr_name value =
    Tf_operationdescription.tf_setattrfloat
      od
      (ptr_of_string attr_name)
      value;
    keep_alive graph

  let set_attr_float_list (graph, od) ~attr_name values =
    Tf_operationdescription.tf_setattrfloatlist
      od
      (ptr_of_string attr_name)
      CArray.(of_list float values |> start)
      (List.length values);
    keep_alive graph

  let set_attr_bool (graph, od) ~attr_name value =
    let value =
      if value
      then Unsigned.UChar.one
      else Unsigned.UChar.zero
    in
    Tf_operationdescription.tf_setattrbool
      od
      (ptr_of_string attr_name)
      value;
    keep_alive graph

  let set_attr_bool_list (graph, od) ~attr_name values =
    let values =
      List.map
        (function
          | true -> Unsigned.UChar.one
          | false -> Unsigned.UChar.zero)
        values
    in
    Tf_operationdescription.tf_setattrboollist
      od
      (ptr_of_string attr_name)
      CArray.(of_list uchar values |> start)
      (List.length values);
    keep_alive graph

  let set_attr_string (graph, od) ~attr_name value =
    Tf_operationdescription.tf_setattrstring
      od
      (ptr_of_string attr_name)
      (ptr_of_string value)
      (String.length value);
    keep_alive graph

  let set_attr_type (graph, od) ~attr_name dtype =
    Tf_operationdescription.tf_setattrtype
      od
      (ptr_of_string attr_name)
      (data_type_to_int dtype);
    keep_alive graph

  let set_attr_type_list (graph, od) ~attr_name dtypes =
    let dtypes = List.map data_type_to_int dtypes in
    Tf_operationdescription.tf_setattrtypelist
      od
      (ptr_of_string attr_name)
      CArray.(of_list int dtypes |> start)
      (List.length dtypes);
    keep_alive graph

  let set_attr_tensor (graph, od) ~attr_name tensor =
    let tensor = Tensor.c_tensor_of_tensor tensor in
    let status = Status.create () in
    Tf_operationdescription.tf_setattrtensor
      od
      (ptr_of_string attr_name)
      tensor
      status;
    keep_alive graph;
    Status.result_or_error status ()

  let set_attr_tensor_string (graph, od) ~attr_name ~shape strings =
    let tensor = Tensor.c_tensor_of_strings strings ~shape in
    let status = Status.create () in
    Tf_operationdescription.tf_setattrtensor
      od
      (ptr_of_string attr_name)
      tensor
      status;
    keep_alive graph;
    Status.result_or_error status ()

  let set_attr_tensors (graph, od) ~attr_name tensors =
    let tensors = List.map Tensor.c_tensor_of_tensor tensors in
    let tensor_start = CArray.(of_list Tf_tensor.t tensors |> start) in
    let status = Status.create () in
    Tf_operationdescription.tf_setattrtensorlist
      od
      (ptr_of_string attr_name)
      tensor_start
      (List.length tensors)
      status;
    keep_alive graph;
    Status.result_or_error status ()

  let set_attr_shape (graph, od) ~attr_name shape =
    let num_dims = List.length shape in
    let shape = List.map Int64.of_int shape in
    let shape = CArray.(of_list int64_t shape |> start) in
    Tf_operationdescription.tf_setattrshape
      od
      (ptr_of_string attr_name)
      shape
      num_dims;
    keep_alive graph

  let import = Graph_import.import

  let find_operation t name =
    let operation = Tf_graph.tf_graphoperationbyname t name in
    if to_voidp operation = null
    then None
    else Some (t, operation)

  let shape t output =
    let status = Status.create () in
    let num_dims = Tf_graph.tf_graphgettensornumdims t output status in
    match Status.code status with
    | TF_OK ->
      let shape = CArray.make int64_t num_dims in
      let shape_start = CArray.start shape in
      Tf_graph.tf_graphgettensorshape t output shape_start num_dims status;
      let dims = CArray.to_list shape |> List.map Int64.to_int in
      Status.result_or_error status dims
    | _ -> Error status

  let add_gradients t ys ~xs =
    let status = Status.create () in
    let nx = List.length xs in
    let ny = List.length ys in
    let xs = CArray.(of_list Tf_output.t xs |> start) in
    let ys = CArray.(of_list Tf_output.t ys |> start) in
    let dys = CArray.make Tf_output.t nx in
    Tf_graph.tf_addgradients t ys ny xs nx (from_voidp Tf_output.t null) status (CArray.start dys);
    keep_alive t;
    match Status.result_or_error status () with
    | Ok () -> Status.Ok (CArray.to_list dys)
    | Error _ as err -> err
end

module Session_options = struct
  include Tf_sessionoptions

  let create () =
    let session_options = tf_newsessionoptions () in
    Gc.finalise tf_deletesessionoptions session_options;
    session_options
end

module Session = struct
  type t = Tf_graph.t * Tf_session.t

  let create ?session_options graph =
    let session_options =
      match session_options with
      | None -> Session_options.create ()
      | Some session_options -> session_options
    in
    let status = Status.create () in
    let session =
      Tf_session.tf_newsession graph session_options status
    in
    Gc.finalise
      (fun session ->
        Tf_session.tf_closesession session status;
        Tf_session.tf_deletesession session status)
      session;
    Status.result_or_error status (graph, session)

  let run ?(inputs = []) ?(outputs = []) ?(targets = []) (graph, t) =
    let status = Status.create () in
    let ninputs = List.length inputs in
    let input_outputs, input_tensors = List.split inputs in
    let input_outputs = CArray.(of_list Tf_output.t input_outputs |> start) in
    let input_tensors = List.map Tensor.c_tensor_of_tensor input_tensors in
    let output_outputs = CArray.(of_list Tf_output.t outputs |> start) in
    let output_len = List.length outputs in
    let output_tensors = CArray.make Tf_tensor.t output_len in
    let input_tensor_start = CArray.(of_list Tf_tensor.t input_tensors |> start) in
    let ntargets = List.length targets in
    let targets = List.map snd targets in
    let output_tensor_start = CArray.start output_tensors in
    let target_operations = CArray.(of_list Tf_operation.t targets |> start) in
    if force_full_major
    then Gc.full_major ();
    Tf_session.tf_sessionrun
      t
      null
      input_outputs
      input_tensor_start
      ninputs
      output_outputs
      output_tensor_start
      output_len
      target_operations
      ntargets
      null
      status;
    Graph.keep_alive graph;
    match Status.result_or_error status () with
    | Ok () ->
      let output_tensors =
        CArray.to_list output_tensors
        |> List.map Tensor.tensor_of_c_tensor
      in
      Status.Ok output_tensors
    | Error _ as err -> err
end
