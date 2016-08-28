open Ctypes
open Foreign

let from = Dl.dlopen ~filename:"libtensorflow-0.10.so" ~flags:[ RTLD_LAZY ]

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

module Tf_tensor = struct
  type t = unit ptr
  let t : t typ = ptr void

  let tf_newtensor =
    foreign "TF_NewTensor" ~from
      (int            (* data type *)
      @-> ptr int64_t (* dims *)
      @-> int         (* num dims *)
      @-> ptr void    (* data *)
      @-> size_t      (* len *)
      @-> funptr (ptr void @-> int @-> ptr void @-> returning void) (* deallocator *)
      @-> ptr void    (* deallocator arg *)
      @-> returning t)

  let tf_deletetensor =
    foreign "TF_DeleteTensor" ~from (t @-> returning void)

  let tf_numdims =
    foreign "TF_NumDims" ~from (t @-> returning int)

  let tf_dim =
    foreign "TF_Dim" ~from (t @-> int @-> returning int)

  let tf_tensorbytesize =
    foreign "TF_TensorByteSize" ~from (t @-> returning size_t)

  let tf_tensordata =
    foreign "TF_TensorData" ~from (t @-> returning (ptr void))

  let tf_tensortype =
    foreign "TF_TensorType" ~from (t @-> returning int)
end

module Tensor = struct
  include Tf_tensor

  let sizeof = function
    | TF_FLOAT -> 4
    | TF_DOUBLE -> 8
    | TF_INT32 -> 4
    | TF_UINT16
    | TF_INT16 -> 2
    | TF_UINT8
    | TF_INT8 -> 1
    | TF_INT64 -> 8
    | TF_STRING
    | TF_COMPLEX
    | TF_BOOL
    | TF_QINT8
    | TF_QUINT8
    | TF_QINT32
    | TF_BFLOAT16
    | TF_QINT16
    | TF_QUINT16
    | Unknown _ -> failwith "Unsupported tensor type"

  let data_type_of_kind (type a) (type b) (kind : (a, b) Bigarray.kind) =
    match kind with
    | Bigarray.Float32 -> TF_FLOAT
    | Bigarray.Float64 -> TF_DOUBLE
    | Bigarray.Int64 -> TF_INT64
    | Bigarray.Int32 -> TF_INT32
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
    Hashtbl.add live_tensors id packed_tensor;
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
    | _ -> failwith "Unsupported tensor type"
end

module Tf_status = struct
  type t = unit ptr
  let t : t typ = ptr void

  let tf_newstatus =
    foreign "TF_NewStatus" ~from (void @-> returning t)

  let tf_deletestatus =
    foreign "TF_DeleteStatus" ~from (t @-> returning void)

  let tf_setstatus =
    foreign "TF_SetStatus" ~from (t @-> int @-> string @-> returning void)

  let tf_getcode =
    foreign "TF_GetCode" ~from (t @-> returning int)

  let tf_message =
    foreign "TF_Message" ~from (t @-> returning string)
end

module Status = struct
  include Tf_status

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

  let set = tf_setstatus

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

module Tf_operation = struct
  type t = unit ptr
  let t : t typ = ptr void

  let tf_operationname =
    foreign "TF_OperationName" ~from (t @-> returning string)

  let tf_operationoptype =
    foreign "TF_OperationOpType" ~from (t @-> returning string)

  let tf_operationdevice =
    foreign "TF_OperationDevice" ~from (t @-> returning string)

  let tf_operationnumoutputs =
    foreign "TF_OperationNumOutputs" ~from (t @-> returning int)

  let tf_operationnuminputs =
    foreign "TF_OperationNumInputs" ~from (t @-> returning int)
end

module Tf_port = struct
  type t
  let t : t structure typ = structure "TF_Port"
  let oper = field t "oper" Tf_operation.t
  let index = field t "index" int
  let () = seal t
end

module Tf_graph = struct
  type t = unit ptr
  let t : t typ = ptr void

  let tf_newgraph =
    foreign "TF_NewGraph" ~from (void @-> returning t)

  let tf_deletegraph =
    foreign "TF_DeleteGraph" ~from (t @-> returning void)
end

module Tf_operationdescription = struct
  type t = unit ptr
  let t : t typ = ptr void

  let tf_newoperation =
    foreign "TF_NewOperation" ~from
      (Tf_graph.t
      @-> string
      @-> string
      @-> returning t)

  let tf_finishoperation =
    foreign "TF_FinishOperation" ~from
      (t
      @-> Tf_status.t
      @-> returning Tf_operation.t)

  let tf_addinput =
    foreign "TF_AddInput" ~from (t @-> Tf_port.t @-> returning void)

  let tf_addinputlist =
    foreign "TF_AddInputList" ~from
      (t
      @-> ptr Tf_port.t
      @-> int
      @-> returning void)

  let tf_addcontrolinput =
    foreign "TF_AddControlInput" ~from (t @-> Tf_operation.t @-> returning void)

  let tf_setattrstring =
    foreign "TF_SetAttrString" ~from
      (t @-> string @-> string @-> int @-> returning void)

  let tf_setattrstringlist =
    foreign "TF_SetAttrStringList" ~from
      (t @-> string @-> ptr (ptr char) @-> ptr int @-> int @-> returning void)

  let tf_setattrint =
    foreign "TF_SetAttrInt" ~from
      (t @-> string @-> int64_t @-> returning void)

  let tf_setattrintlist =
    foreign "TF_SetAttrIntList" ~from
      (t @-> string @-> ptr int64_t @-> int @-> returning void)

  let tf_setattrfloat =
    foreign "TF_SetAttrFloat" ~from
      (t @-> string @-> float @-> returning void)

  let tf_setattrfloatlist =
    foreign "TF_SetAttrFloatList" ~from
      (t @-> string @-> ptr float @-> int @-> returning void)

  let tf_setattrbool =
    foreign "TF_SetAttrBool" ~from
      (t @-> string @-> uchar @-> returning void)

  let tf_setattrboollist =
    foreign "TF_SetAttrBoolList" ~from
      (t @-> string @-> ptr uchar @-> int @-> returning void)

  let tf_setattrtype =
    foreign "TF_SetAttrType" ~from
      (t @-> string @-> int @-> returning void)

  let tf_setattrtypelist =
    foreign "TF_SetAttrTypeList" ~from
      (t @-> string @-> ptr int @-> int @-> returning void)

  let tf_setattrshape =
    foreign "TF_SetAttrShape" ~from
      (t @-> string @-> ptr int64_t @-> int @-> returning void)

  let tf_setattrshapelist =
    foreign "TF_SetAttrShapeList" ~from
      (t @-> string @-> ptr (ptr int64_t) @-> ptr int @-> int @-> returning void)

  let tf_setattrtensor =
    foreign "TF_SetAttrTensor" ~from
      (t @-> string @-> Tf_tensor.t @-> Tf_status.t @-> returning void)

  let tf_setattrtensorlist =
    foreign "TF_SetAttrTensorList" ~from
      (t @-> string @-> ptr Tf_tensor.t @-> int @-> Tf_status.t @-> returning void)
end

module Graph = struct
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

  type port = Tf_port.t structure

  let create_port (_graph, op) ~index =
    let port = make Tf_port.t in
    setf port Tf_port.oper op;
    setf port Tf_port.index index;
    port

  let new_operation t ~op_name ~name =
    let od = Tf_operationdescription.tf_newoperation t op_name name in
    t, od

  let finish_operation (graph, od) =
    let status = Status.create () in
    let operation = Tf_operationdescription.tf_finishoperation od status in
    Status.result_or_error status (graph, operation)

  let add_input (graph, od) (graph', op) ~index =
    if graph != graph'
    then failwith "Calling add_input on different graphs.";
    let port = create_port (graph, op) ~index in
    Tf_operationdescription.tf_addinput od port;
    keep_alive graph

  let add_inputs (graph, od) op_and_indexes =
    let ports =
      List.map (fun (op, index) -> create_port op ~index)
        op_and_indexes
      |> CArray.of_list Tf_port.t
      |> CArray.start
    in
    Tf_operationdescription.tf_addinputlist od ports (List.length op_and_indexes);
    keep_alive graph

  let set_attr_int (_, od) ~attr_name value =
    Tf_operationdescription.tf_setattrint od attr_name (Int64.of_int value)

  let set_attr_float (_, od) ~attr_name value =
    Tf_operationdescription.tf_setattrfloat od attr_name value

  let set_attr_bool (_, od) ~attr_name value =
    let value =
      if value
      then Unsigned.UChar.one
      else Unsigned.UChar.zero
    in
    Tf_operationdescription.tf_setattrbool od attr_name value

  let set_attr_string (_, od) ~attr_name value =
    Tf_operationdescription.tf_setattrstring od attr_name value (String.length value)

  let set_attr_type (_, od) ~attr_name dtype =
    Tf_operationdescription.tf_setattrtype od attr_name (data_type_to_int dtype)

  let set_attr_tensor (_, od) ~attr_name tensor =
    let tensor = Tensor.c_tensor_of_tensor tensor in
    let status = Status.create () in
    Tf_operationdescription.tf_setattrtensor od attr_name tensor status;
    Status.result_or_error status ()

  let set_attr_tensors (_, od) ~attr_name tensors =
    let tensors = List.map Tensor.c_tensor_of_tensor tensors in
    let tensor_start = CArray.(of_list Tf_tensor.t tensors |> start) in
    let status = Status.create () in
    Tf_operationdescription.tf_setattrtensorlist
      od
      attr_name
      tensor_start
      (List.length tensors)
      status;
    Status.result_or_error status ()

  let set_attr_shape (_, od) ~attr_name shape =
    let num_dims = List.length shape in
    let shape = List.map Int64.of_int shape in
    let shape = CArray.(of_list int64_t shape |> start) in
    Tf_operationdescription.tf_setattrshape od attr_name shape num_dims
end

module Tf_sessionoptions = struct
  type t = unit ptr
  let t : t typ = ptr void

  let tf_newsessionoptions =
    foreign "TF_NewSessionOptions" ~from (void @-> returning t)

  let tf_settarget =
    foreign "TF_SetTarget" ~from (t @-> string @-> returning void)

  let tf_setconfig =
    foreign "TF_SetConfig" ~from
      (t
      @-> ptr void
      @-> size_t
      @-> Tf_status.t
      @-> returning t)

  let tf_deletesessionoptions =
    foreign "TF_DeleteSessionOptions" ~from (t @-> returning void)
end

module Session_options = struct
  include Tf_sessionoptions

  let create () =
    let session_options = tf_newsessionoptions () in
    Gc.finalise tf_deletesessionoptions session_options;
    session_options
end

module Tf_sessionwithgraph = struct
  type t = unit ptr
  let t : t typ = ptr void

  let tf_newsessionwithgraph =
    foreign "TF_NewSessionWithGraph" ~from
      (Tf_graph.t @-> Tf_sessionoptions.t @-> Tf_status.t @-> returning t)

  let tf_closesessionwithgraph =
    foreign "TF_CloseSessionWithGraph" ~from (t @-> Tf_status.t @-> returning void)

  let tf_deletesessionwithgraph =
    foreign "TF_DeleteSessionWithGraph" ~from (t @-> Tf_status.t @-> returning void)

  let tf_sessionrun =
    foreign "TF_SessionRun" ~from
      (t
      @-> ptr void (* run_options *)
      (* Input tensors *)
      @-> ptr Tf_port.t
      @-> ptr Tf_tensor.t
      @-> int
      (* Output tensors *)
      @-> ptr Tf_port.t
      @-> ptr Tf_tensor.t
      @-> int
      (* Target nodes *)
      @-> ptr Tf_operation.t
      @-> int
      @-> ptr void (* run_metadata *)
      (* Output status *)
      @-> Tf_status.t
      @-> returning void)
end

module Tf_session = struct
  type t = unit ptr
  let t : t typ = ptr void

  let tf_newsession =
    foreign "TF_NewSession" ~from
      (Tf_sessionoptions.t @-> Tf_status.t @-> returning t)

  let tf_closesession =
    foreign "TF_CloseSession" ~from (t @-> Tf_status.t @-> returning void)

  let tf_deletesession =
    foreign "TF_DeleteSession" ~from (t @-> Tf_status.t @-> returning void)

  let tf_extendgraph =
    foreign "TF_ExtendGraph" ~from
      (t @-> string @-> size_t @-> Tf_status.t @-> returning void)

  (* We use [ptr (ptr char)] rather than [ptr string] as when using [CArray.of_list]
     to build the [ptr string], the reference to the C (copied) version of the string can
     get lost which will be problematic if the GC triggers just after. *)
  let tf_run =
    foreign "TF_Run" ~from
      (t
      @-> ptr void (* run_options *)
      (* Input tensors *)
      @-> ptr (ptr char)
      @-> ptr Tf_tensor.t
      @-> int
      (* Output tensors *)
      @-> ptr (ptr char)
      @-> ptr Tf_tensor.t
      @-> int
      (* Target nodes *)
      @-> ptr (ptr char)
      @-> int
      @-> ptr void (* run_metadata *)
      (* Output status *)
      @-> Tf_status.t
      @-> returning void)
end

module Session = struct
  include Tf_session

  let create ?session_options () =
    let session_options =
      match session_options with
      | None -> Session_options.create ()
      | Some session_options -> session_options
    in
    let status = Status.create () in
    let session = tf_newsession session_options status in
    Gc.finalise
      (fun session ->
        tf_closesession session status;
        tf_deletesession session status)
      session;
    Status.result_or_error status session

  let extend_graph t protobuf =
    let status = Status.create () in
    let protobuf = Protobuf.to_string protobuf in
    if force_full_major
    then Gc.full_major ();
    tf_extendgraph
      t
      protobuf
      (String.length protobuf |> Unsigned.Size_t.of_int)
      status;
    Status.result_or_error status ()

  let char_list_of_string s =
    let rec char_list_of_string idx acc =
      if idx < 0
      then acc
      else char_list_of_string (idx - 1) (s.[idx] :: acc)
    in
    char_list_of_string (String.length s - 1) [ Char.chr 0 ]

  (* [opaque_identity] is not available pre 4.03, this hack ensures that it
     exists. *)
  let opaque_identity x = x
  let _ = opaque_identity
  let opaque_identity = let open Sys in opaque_identity

  let ptr_ptr_char l =
    let l =
      List.map (fun s -> char_list_of_string s |> CArray.of_list char) l
    in
    let ptr =
      CArray.of_list (ptr char) (List.map CArray.start l)
      |> CArray.start
    in
    (* Keep a reference to l as it could be GCed otherwise, [opaque_identity] is
       required here so that the [ignore] is not optimized by flambda. *)
    Gc.finalise (fun _ -> ignore (opaque_identity l)) ptr;
    ptr

  let run ?(inputs = []) ?(outputs = []) ?(targets = []) t =
    let status = Status.create () in
    let ninputs = List.length inputs in
    let input_names, input_tensors = List.split inputs in
    let input_tensors = List.map Tensor.c_tensor_of_tensor input_tensors in
    let output_len = List.length outputs in
    let output_tensors = CArray.make Tf_tensor.t output_len in
    let input_names = ptr_ptr_char input_names in
    let input_tensor_start = CArray.(of_list Tf_tensor.t input_tensors |> start) in
    let outputs = ptr_ptr_char outputs in
    let ntargets = List.length targets in
    let targets = ptr_ptr_char targets in
    let output_tensor_start = CArray.start output_tensors in
    if force_full_major
    then Gc.full_major ();
    tf_run
      t
      null
      input_names
      input_tensor_start
      ninputs
      outputs
      output_tensor_start
      output_len
      targets
      ntargets
      null
      status;
    match Status.result_or_error status () with
    | Ok () ->
      let output_tensors =
        CArray.to_list output_tensors
        |> List.map Tensor.tensor_of_c_tensor
      in
      Status.Ok output_tensors
    | Error _ as err -> err
end

module Session_with_graph = struct
  type t = Tf_graph.t * Tf_sessionwithgraph.t

  let create ?session_options graph =
    let session_options =
      match session_options with
      | None -> Session_options.create ()
      | Some session_options -> session_options
    in
    let status = Status.create () in
    let session =
      Tf_sessionwithgraph.tf_newsessionwithgraph graph session_options status
    in
    Gc.finalise
      (fun session ->
        Tf_sessionwithgraph.tf_closesessionwithgraph session status;
        Tf_sessionwithgraph.tf_deletesessionwithgraph session status)
      session;
    Status.result_or_error status (graph, session)

  let run ?(inputs = []) ?(outputs = []) ?(targets = []) (graph, t) =
    let status = Status.create () in
    let ninputs = List.length inputs in
    let input_ports, input_tensors = List.split inputs in
    let input_ports = CArray.(of_list Tf_port.t input_ports |> start) in
    let input_tensors = List.map Tensor.c_tensor_of_tensor input_tensors in
    let output_ports = CArray.(of_list Tf_port.t outputs |> start) in
    let output_len = List.length outputs in
    let output_tensors = CArray.make Tf_tensor.t output_len in
    let input_tensor_start = CArray.(of_list Tf_tensor.t input_tensors |> start) in
    let ntargets = List.length targets in
    let targets = List.map snd targets in
    let output_tensor_start = CArray.start output_tensors in
    let target_operations = CArray.(of_list Tf_operation.t targets |> start) in
    if force_full_major
    then Gc.full_major ();
    Tf_sessionwithgraph.tf_sessionrun
      t
      null
      input_ports
      input_tensor_start
      ninputs
      output_ports
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


let () =
  ignore
    ( Tf_sessionoptions.tf_settarget
    , Tf_sessionoptions.tf_setconfig
    , Tf_tensor.tf_tensorbytesize
    , Status.set)
