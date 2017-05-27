open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  open F
  module Tf_tensor = struct
    type t = unit ptr
    let t : t typ = ptr void

    let tf_newtensor =
      foreign "TF_NewTensor"
        (int            (* data type *)
        @-> ptr int64_t (* dims *)
        @-> int         (* num dims *)
        @-> ptr void    (* data *)
        @-> size_t      (* len *)
        @-> static_funptr Ctypes.(ptr void @-> size_t @-> ptr void @-> returning void) (* deallocator *)
        @-> ptr void    (* deallocator arg *)
        @-> returning t)

    let tf_deletetensor =
      foreign "TF_DeleteTensor" (t @-> returning void)

    let tf_numdims =
      foreign "TF_NumDims" (t @-> returning int)

    let tf_dim =
      foreign "TF_Dim" (t @-> int @-> returning int)

    let tf_tensorbytesize =
      foreign "TF_TensorByteSize" (t @-> returning size_t)

    let tf_tensordata =
      foreign "TF_TensorData" (t @-> returning (ptr void))

    let tf_tensortype =
      foreign "TF_TensorType" (t @-> returning int)
  end

  module Tf_status = struct
    type t = unit ptr
    let t : t typ = ptr void

    let tf_newstatus =
      foreign "TF_NewStatus" (void @-> returning t)

    let tf_deletestatus =
      foreign "TF_DeleteStatus" (t @-> returning void)

    let tf_setstatus =
      foreign "TF_SetStatus" (t @-> int @-> string @-> returning void)

    let tf_getcode =
      foreign "TF_GetCode" (t @-> returning int)

    let tf_message =
      foreign "TF_Message" (t @-> returning string)
  end

  module Tf_operation = struct
    type t = unit ptr
    let t : t typ = ptr void

    let tf_operationname =
      foreign "TF_OperationName" (t @-> returning string)

    let tf_operationoptype =
      foreign "TF_OperationOpType" (t @-> returning string)

    let tf_operationdevice =
      foreign "TF_OperationDevice" (t @-> returning string)

    let tf_operationnumoutputs =
      foreign "TF_OperationNumOutputs" (t @-> returning int)

    let tf_operationnuminputs =
      foreign "TF_OperationNumInputs" (t @-> returning int)
  end

  module Tf_output = struct
    type t
    let t : t structure typ = structure "TF_Output"
    let oper = field t "oper" Tf_operation.t
    let index = field t "index" int
    let () = seal t
  end

  module Tf_import_graph_def_options = struct
    type t = unit ptr
    let t : t typ = ptr void

    let tf_newimportgraphdefoptions =
      foreign "TF_NewImportGraphDefOptions" (void @-> returning t)

    let tf_deleteimportgraphdefoptions =
      foreign "TF_DeleteImportGraphDefOptions" (t @-> returning void)
  end

  module Tf_buffer = struct
    type t = unit ptr
    let t : t typ = ptr void

    let tf_newbufferfromstring =
      foreign "TF_NewBufferFromString" (string @-> int @-> returning t)

    let tf_deletebuffer =
      foreign "TF_DeleteBuffer" (t @-> returning void)
  end

  module Tf_graph = struct
    type t = unit ptr
    let t : t typ = ptr void

    let tf_newgraph =
      foreign "TF_NewGraph" (void @-> returning t)

    let tf_deletegraph =
      foreign "TF_DeleteGraph" (t @-> returning void)

    let tf_graphimportgraphdef =
      foreign "TF_GraphImportGraphDef"
        (t
        @-> Tf_buffer.t
        @-> Tf_import_graph_def_options.t
        @-> Tf_status.t
        @-> returning void)

    let tf_graphoperationbyname =
      foreign "TF_GraphOperationByName"
        (t
        @-> string
        @-> returning Tf_operation.t)

    let tf_graphgettensornumdims =
      foreign "TF_GraphGetTensorNumDims"
        (t
        @-> Tf_output.t
        @-> Tf_status.t
        @-> returning int)

    let tf_graphgettensorshape =
      foreign "TF_GraphGetTensorShape"
        (t
        @-> Tf_output.t
        @-> ptr int64_t
        @-> int
        @-> Tf_status.t
        @-> returning void)

    let tf_addgradients =
      foreign "TF_AddGradients"
        (t
        @-> ptr Tf_output.t
        @-> int
        @-> ptr Tf_output.t
        @-> int
        @-> ptr Tf_output.t
        @-> Tf_status.t
        @-> ptr Tf_output.t
        @-> returning void)
  end

  module Tf_operationdescription = struct
    type t = unit ptr
    let t : t typ = ptr void

    let tf_newoperation =
      foreign "TF_NewOperation"
        (Tf_graph.t
        @-> ptr char
        @-> ptr char
        @-> returning t)

    let tf_finishoperation =
      foreign "TF_FinishOperation"
        (t
        @-> Tf_status.t
        @-> returning Tf_operation.t)

    let tf_addinput =
      foreign "TF_AddInput" (t @-> Tf_output.t @-> returning void)

    let tf_addinputlist =
      foreign "TF_AddInputList"
        (t
        @-> ptr Tf_output.t
        @-> int
        @-> returning void)

    let tf_addcontrolinput =
      foreign "TF_AddControlInput" (t @-> Tf_operation.t @-> returning void)

    let tf_setattrstring =
      foreign "TF_SetAttrString"
        (t @-> ptr char @-> ptr char @-> int @-> returning void)

    let tf_setattrstringlist =
      foreign "TF_SetAttrStringList"
        (t @-> ptr char @-> ptr (ptr void) @-> ptr size_t @-> int @-> returning void)

    let tf_setattrint =
      foreign "TF_SetAttrInt"
        (t @-> ptr char @-> int64_t @-> returning void)

    let tf_setattrintlist =
      foreign "TF_SetAttrIntList"
        (t @-> ptr char @-> ptr int64_t @-> int @-> returning void)

    let tf_setattrfloat =
      foreign "TF_SetAttrFloat"
        (t @-> ptr char @-> float @-> returning void)

    let tf_setattrfloatlist =
      foreign "TF_SetAttrFloatList"
        (t @-> ptr char @-> ptr float @-> int @-> returning void)

    let tf_setattrbool =
      foreign "TF_SetAttrBool"
        (t @-> ptr char @-> uchar @-> returning void)

    let tf_setattrboollist =
      foreign "TF_SetAttrBoolList"
        (t @-> ptr char @-> ptr uchar @-> int @-> returning void)

    let tf_setattrtype =
      foreign "TF_SetAttrType"
        (t @-> ptr char @-> int @-> returning void)

    let tf_setattrtypelist =
      foreign "TF_SetAttrTypeList"
        (t @-> ptr char @-> ptr int @-> int @-> returning void)

    let tf_setattrshape =
      foreign "TF_SetAttrShape"
        (t @-> ptr char @-> ptr int64_t @-> int @-> returning void)

    let tf_setattrshapelist =
      foreign "TF_SetAttrShapeList"
        (t @-> ptr char @-> ptr (ptr int64_t) @-> ptr int @-> int @-> returning void)

    let tf_setattrtensor =
      foreign "TF_SetAttrTensor"
        (t @-> ptr char @-> Tf_tensor.t @-> Tf_status.t @-> returning void)

    let tf_setattrtensorlist =
      foreign "TF_SetAttrTensorList"
        (t @-> ptr char @-> ptr Tf_tensor.t @-> int @-> Tf_status.t @-> returning void)
  end

  module Tf_sessionoptions = struct
    type t = unit ptr
    let t : t typ = ptr void

    let tf_newsessionoptions =
      foreign "TF_NewSessionOptions" (void @-> returning t)

    let tf_settarget =
      foreign "TF_SetTarget" (t @-> string @-> returning void)

    let tf_setconfig =
      foreign "TF_SetConfig"
        (t
        @-> ptr void
        @-> size_t
        @-> Tf_status.t
        @-> returning void)

    let tf_deletesessionoptions =
      foreign "TF_DeleteSessionOptions" (t @-> returning void)
  end

  module Tf_session = struct
    type t = unit ptr
    let t : t typ = ptr void

    let tf_newsession =
      foreign "TF_NewSession"
        (Tf_graph.t @-> Tf_sessionoptions.t @-> Tf_status.t @-> returning t)

    let tf_closesession =
      foreign "TF_CloseSession" (t @-> Tf_status.t @-> returning void)

    let tf_deletesession =
      foreign "TF_DeleteSession" (t @-> Tf_status.t @-> returning void)

    let tf_sessionrun =
      foreign "TF_SessionRun"
        (t
        @-> ptr void (* run_options *)
        (* Input tensors *)
        @-> ptr Tf_output.t
        @-> ptr Tf_tensor.t
        @-> int
        (* Output tensors *)
        @-> ptr Tf_output.t
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
end
