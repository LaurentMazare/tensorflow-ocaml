(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)
open Node

let abs
    ?(name = "Abs")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Abs"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let add
    ?(name = "Add")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Add"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let addN
    ?(name = "AddN")
    (inputs : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t list)
  =
  let attributes = [ "T", Type (P (List.hd inputs).output_type) ] in
  let attributes =
    ("N", Int (List.length inputs)) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "AddN"
  ; output_type = (List.hd inputs).output_type
  ; inputs = List.map (fun n -> P n) inputs
  ; attributes
  ; output_idx = None
  }

let adjustContrast
    ?(name = "AdjustContrast")
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (contrast_factor : [ `float ] t)
    (min_value : [ `float ] t)
    (max_value : [ `float ] t)
  =
  let attributes = [ "T", Type (P images.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "AdjustContrast"
  ; output_type = Type.Float
  ; inputs = [ P images; P contrast_factor; P min_value; P max_value ]
  ; attributes
  ; output_idx = None
  }

let adjustContrastv2
    ?(name = "AdjustContrastv2")
    (images : [ `float ] t)
    (contrast_factor : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "AdjustContrastv2"
  ; output_type = Type.Float
  ; inputs = [ P images; P contrast_factor ]
  ; attributes
  ; output_idx = None
  }

let all
    ?(name = "All")
    ?keep_dims
    (input : [ `bool ] t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "All"
  ; output_type = Type.Bool
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_idx = None
  }

let any
    ?(name = "Any")
    ?keep_dims
    (input : [ `bool ] t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Any"
  ; output_type = Type.Bool
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_idx = None
  }

let applyAdagrad
    ?(name = "ApplyAdagrad")
    ?use_locking
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ApplyAdagrad"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad ]
  ; attributes
  ; output_idx = None
  }

let applyAdam
    ?(name = "ApplyAdam")
    ?use_locking
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (m : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (v : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (beta1_power : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (beta2_power : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (beta1 : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (beta2 : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (epsilon : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ApplyAdam"
  ; output_type = var.output_type
  ; inputs = [ P var; P m; P v; P beta1_power; P beta2_power; P lr; P beta1; P beta2; P epsilon; P grad ]
  ; attributes
  ; output_idx = None
  }

let applyFtrl
    ?(name = "ApplyFtrl")
    ?use_locking
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (linear : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (l1 : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (l2 : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr_power : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ApplyFtrl"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P linear; P grad; P lr; P l1; P l2; P lr_power ]
  ; attributes
  ; output_idx = None
  }

let applyGradientDescent
    ?(name = "ApplyGradientDescent")
    ?use_locking
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (alpha : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (delta : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ApplyGradientDescent"
  ; output_type = var.output_type
  ; inputs = [ P var; P alpha; P delta ]
  ; attributes
  ; output_idx = None
  }

let applyMomentum
    ?(name = "ApplyMomentum")
    ?use_locking
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (momentum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ApplyMomentum"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P momentum ]
  ; attributes
  ; output_idx = None
  }

let applyRMSProp
    ?(name = "ApplyRMSProp")
    ?use_locking
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (ms : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (mom : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (rho : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (momentum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (epsilon : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ApplyRMSProp"
  ; output_type = var.output_type
  ; inputs = [ P var; P ms; P mom; P lr; P rho; P momentum; P epsilon; P grad ]
  ; attributes
  ; output_idx = None
  }

let argMax
    ?(name = "ArgMax")
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (dimension : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ArgMax"
  ; output_type = Type.Int64
  ; inputs = [ P input; P dimension ]
  ; attributes
  ; output_idx = None
  }

let argMin
    ?(name = "ArgMin")
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (dimension : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ArgMin"
  ; output_type = Type.Int64
  ; inputs = [ P input; P dimension ]
  ; attributes
  ; output_idx = None
  }

let assign
    ?(name = "Assign")
    ?validate_shape
    ?use_locking
    (ref : 't t)
    (value : 't t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  let attributes =
    match validate_shape with | None -> attributes | Some validate_shape -> ("validate_shape", Bool validate_shape) :: attributes
  in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Assign"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P value ]
  ; attributes
  ; output_idx = None
  }

let assignAdd
    ?(name = "AssignAdd")
    ?use_locking
    (ref : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (value : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "AssignAdd"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P value ]
  ; attributes
  ; output_idx = None
  }

let assignSub
    ?(name = "AssignSub")
    ?use_locking
    (ref : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (value : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "AssignSub"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P value ]
  ; attributes
  ; output_idx = None
  }

let avgPool
    ?(name = "AvgPool")
    ~ksize
    ~strides
    ~padding
    ?data_format
    (value : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  let attributes =
    ("ksize", List (Int ksize)) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "AvgPool"
  ; output_type = value.output_type
  ; inputs = [ P value ]
  ; attributes
  ; output_idx = None
  }

let avgPoolGrad
    ?(name = "AvgPoolGrad")
    ~ksize
    ~strides
    ~padding
    ?data_format
    (orig_input_shape : [ `int32 ] t)
    (grad : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P grad.output_type) ] in
  let attributes =
    ("ksize", List (Int ksize)) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "AvgPoolGrad"
  ; output_type = grad.output_type
  ; inputs = [ P orig_input_shape; P grad ]
  ; attributes
  ; output_idx = None
  }

let batchCholesky
    ?(name = "BatchCholesky")
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BatchCholesky"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let batchMatMul
    ?(name = "BatchMatMul")
    ?adj_x
    ?adj_y
    (x : ([< `float | `double | `int32 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  let attributes =
    match adj_x with | None -> attributes | Some adj_x -> ("adj_x", Bool adj_x) :: attributes
  in
  let attributes =
    match adj_y with | None -> attributes | Some adj_y -> ("adj_y", Bool adj_y) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BatchMatMul"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let batchMatrixDeterminant
    ?(name = "BatchMatrixDeterminant")
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BatchMatrixDeterminant"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let batchMatrixInverse
    ?(name = "BatchMatrixInverse")
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BatchMatrixInverse"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let batchMatrixSolve
    ?(name = "BatchMatrixSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BatchMatrixSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes
  ; output_idx = None
  }

let batchMatrixSolveLs
    ?(name = "BatchMatrixSolveLs")
    ?fast
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
    (l2_regularizer : [ `double ] t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  let attributes =
    match fast with | None -> attributes | Some fast -> ("fast", Bool fast) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BatchMatrixSolveLs"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs; P l2_regularizer ]
  ; attributes
  ; output_idx = None
  }

let batchMatrixTriangularSolve
    ?(name = "BatchMatrixTriangularSolve")
    ?lower
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  let attributes =
    match lower with | None -> attributes | Some lower -> ("lower", Bool lower) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BatchMatrixTriangularSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes
  ; output_idx = None
  }

let batchNormWithGlobalNormalization
    ?(name = "BatchNormWithGlobalNormalization")
    ~variance_epsilon
    ~scale_after_normalization
    (t : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (m : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (v : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (beta : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (gamma : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P t.output_type) ] in
  let attributes =
    ("variance_epsilon", Float variance_epsilon) :: attributes
  in
  let attributes =
    ("scale_after_normalization", Bool scale_after_normalization) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BatchNormWithGlobalNormalization"
  ; output_type = t.output_type
  ; inputs = [ P t; P m; P v; P beta; P gamma ]
  ; attributes
  ; output_idx = None
  }

let batchSelfAdjointEig
    ?(name = "BatchSelfAdjointEig")
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BatchSelfAdjointEig"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let biasAdd
    ?(name = "BiasAdd")
    ?data_format
    (value : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (bias : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BiasAdd"
  ; output_type = value.output_type
  ; inputs = [ P value; P bias ]
  ; attributes
  ; output_idx = None
  }

let biasAddGrad
    ?(name = "BiasAddGrad")
    ?data_format
    (out_backprop : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P out_backprop.output_type) ] in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BiasAddGrad"
  ; output_type = out_backprop.output_type
  ; inputs = [ P out_backprop ]
  ; attributes
  ; output_idx = None
  }

let biasAddV1
    ?(name = "BiasAddV1")
    (value : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (bias : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "BiasAddV1"
  ; output_type = value.output_type
  ; inputs = [ P value; P bias ]
  ; attributes
  ; output_idx = None
  }

let bitcast
    ?(name = "Bitcast")
    ~type_
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ;  "type", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Bitcast"
  ; output_type = type_
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let cast
    ?(name = "Cast")
    ~type_
    (x : 'srcT t)
  =
  let attributes = [ "SrcT", Type (P x.output_type) ;  "DstT", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Cast"
  ; output_type = type_
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let ceil
    ?(name = "Ceil")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Ceil"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let checkNumerics
    ?(name = "CheckNumerics")
    ~message
    (tensor : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P tensor.output_type) ] in
  let attributes =
    ("message", String message) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "CheckNumerics"
  ; output_type = tensor.output_type
  ; inputs = [ P tensor ]
  ; attributes
  ; output_idx = None
  }

let cholesky
    ?(name = "Cholesky")
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Cholesky"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let complex
    ?(name = "Complex")
    (real : [ `float ] t)
    (imag : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Complex"
  ; output_type = Type.Complex64
  ; inputs = [ P real; P imag ]
  ; attributes
  ; output_idx = None
  }

let complexAbs
    ?(name = "ComplexAbs")
    (x : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ComplexAbs"
  ; output_type = Type.Float
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let concat
    ?(name = "Concat")
    (concat_dim : [ `int32 ] t)
    (values : 't t list)
  =
  let attributes = [ "T", Type (P (List.hd values).output_type) ] in
  let attributes =
    ("N", Int (List.length values)) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Concat"
  ; output_type = (List.hd values).output_type
  ; inputs = [ P concat_dim ] @ List.map (fun n -> P n) values
  ; attributes
  ; output_idx = None
  }

let concatOffset
    ?(name = "ConcatOffset")
    (concat_dim : [ `int32 ] t)
    (shape : [ `int32 ] t list)
  =
  let attributes = [] in
  let attributes =
    ("N", Int (List.length shape)) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ConcatOffset"
  ; output_type = Type.Int32
  ; inputs = [ P concat_dim ] @ List.map (fun n -> P n) shape
  ; attributes
  ; output_idx = None
  }

let conj
    ?(name = "Conj")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Conj"
  ; output_type = Type.Complex64
  ; inputs = [ P in__ ]
  ; attributes
  ; output_idx = None
  }

let controlTrigger
    ?(name = "ControlTrigger")
    ()
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ControlTrigger"
  ; output_type = Type.Unit
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let conv2D
    ?(name = "Conv2D")
    ~strides
    ?use_cudnn_on_gpu
    ~padding
    ?data_format
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    match use_cudnn_on_gpu with | None -> attributes | Some use_cudnn_on_gpu -> ("use_cudnn_on_gpu", Bool use_cudnn_on_gpu) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Conv2D"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter ]
  ; attributes
  ; output_idx = None
  }

let conv2DBackpropFilter
    ?(name = "Conv2DBackpropFilter")
    ~strides
    ?use_cudnn_on_gpu
    ~padding
    ?data_format
    (input : ([< `float | `double ] as 't) t)
    (filter_sizes : [ `int32 ] t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    match use_cudnn_on_gpu with | None -> attributes | Some use_cudnn_on_gpu -> ("use_cudnn_on_gpu", Bool use_cudnn_on_gpu) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Conv2DBackpropFilter"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter_sizes; P out_backprop ]
  ; attributes
  ; output_idx = None
  }

let conv2DBackpropInput
    ?(name = "Conv2DBackpropInput")
    ~strides
    ?use_cudnn_on_gpu
    ~padding
    ?data_format
    (input_sizes : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P filter.output_type) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    match use_cudnn_on_gpu with | None -> attributes | Some use_cudnn_on_gpu -> ("use_cudnn_on_gpu", Bool use_cudnn_on_gpu) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Conv2DBackpropInput"
  ; output_type = filter.output_type
  ; inputs = [ P input_sizes; P filter; P out_backprop ]
  ; attributes
  ; output_idx = None
  }

let cos
    ?(name = "Cos")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Cos"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let countUpTo
    ?(name = "CountUpTo")
    ~limit
    (ref : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  let attributes =
    ("limit", Int limit) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "CountUpTo"
  ; output_type = ref.output_type
  ; inputs = [ P ref ]
  ; attributes
  ; output_idx = None
  }

let cross
    ?(name = "Cross")
    (a : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (b : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P a.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Cross"
  ; output_type = a.output_type
  ; inputs = [ P a; P b ]
  ; attributes
  ; output_idx = None
  }

let decodeJSONExample
    ?(name = "DecodeJSONExample")
    (json_examples : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DecodeJSONExample"
  ; output_type = Type.String
  ; inputs = [ P json_examples ]
  ; attributes
  ; output_idx = None
  }

let decodePng
    ?(name = "DecodePng")
    ~type_
    ?channels
    (contents : [ `string ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match channels with | None -> attributes | Some channels -> ("channels", Int channels) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DecodePng"
  ; output_type = type_
  ; inputs = [ P contents ]
  ; attributes
  ; output_idx = None
  }

let decodeRaw
    ?(name = "DecodeRaw")
    ~type_
    ?little_endian
    (bytes : [ `string ] t)
  =
  let attributes = [ "out_type", Type (P type_) ] in
  let attributes =
    match little_endian with | None -> attributes | Some little_endian -> ("little_endian", Bool little_endian) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DecodeRaw"
  ; output_type = type_
  ; inputs = [ P bytes ]
  ; attributes
  ; output_idx = None
  }

let depthToSpace
    ?(name = "DepthToSpace")
    ~block_size
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("block_size", Int block_size) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DepthToSpace"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let depthwiseConv2dNative
    ?(name = "DepthwiseConv2dNative")
    ~strides
    ~padding
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DepthwiseConv2dNative"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter ]
  ; attributes
  ; output_idx = None
  }

let depthwiseConv2dNativeBackpropFilter
    ?(name = "DepthwiseConv2dNativeBackpropFilter")
    ~strides
    ~padding
    (input : ([< `float | `double ] as 't) t)
    (filter_sizes : [ `int32 ] t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DepthwiseConv2dNativeBackpropFilter"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter_sizes; P out_backprop ]
  ; attributes
  ; output_idx = None
  }

let depthwiseConv2dNativeBackpropInput
    ?(name = "DepthwiseConv2dNativeBackpropInput")
    ~strides
    ~padding
    (input_sizes : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P filter.output_type) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DepthwiseConv2dNativeBackpropInput"
  ; output_type = filter.output_type
  ; inputs = [ P input_sizes; P filter; P out_backprop ]
  ; attributes
  ; output_idx = None
  }

let destroyTemporaryVariable
    ?(name = "DestroyTemporaryVariable")
    ~var_name
    (ref : 't t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  let attributes =
    ("var_name", String var_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DestroyTemporaryVariable"
  ; output_type = ref.output_type
  ; inputs = [ P ref ]
  ; attributes
  ; output_idx = None
  }

let diag
    ?(name = "Diag")
    (diagonal : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P diagonal.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Diag"
  ; output_type = diagonal.output_type
  ; inputs = [ P diagonal ]
  ; attributes
  ; output_idx = None
  }

let diagPart
    ?(name = "DiagPart")
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DiagPart"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let digamma
    ?(name = "Digamma")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Digamma"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let div
    ?(name = "Div")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Div"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let drawBoundingBoxes
    ?(name = "DrawBoundingBoxes")
    (images : [ `float ] t)
    (boxes : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DrawBoundingBoxes"
  ; output_type = Type.Float
  ; inputs = [ P images; P boxes ]
  ; attributes
  ; output_idx = None
  }

let dynamicPartition
    ?(name = "DynamicPartition")
    ~num_partitions
    (data : 't t)
    (partitions : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  let attributes =
    ("num_partitions", Int num_partitions) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DynamicPartition"
  ; output_type = data.output_type
  ; inputs = [ P data; P partitions ]
  ; attributes
  ; output_idx = None
  }

let dynamicStitch
    ?(name = "DynamicStitch")
    (indices : [ `int32 ] t list)
    (data : 't t list)
  =
  let attributes = [ "T", Type (P (List.hd data).output_type) ] in
  let attributes =
    ("N", Int (List.length indices)) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "DynamicStitch"
  ; output_type = (List.hd data).output_type
  ; inputs = List.map (fun n -> P n) indices @ List.map (fun n -> P n) data
  ; attributes
  ; output_idx = None
  }

let editDistance
    ?(name = "EditDistance")
    ?normalize
    (hypothesis_indices : [ `int64 ] t)
    (hypothesis_values : 't t)
    (hypothesis_shape : [ `int64 ] t)
    (truth_indices : [ `int64 ] t)
    (truth_values : 't t)
    (truth_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P hypothesis_values.output_type) ] in
  let attributes =
    match normalize with | None -> attributes | Some normalize -> ("normalize", Bool normalize) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "EditDistance"
  ; output_type = Type.Float
  ; inputs = [ P hypothesis_indices; P hypothesis_values; P hypothesis_shape; P truth_indices; P truth_values; P truth_shape ]
  ; attributes
  ; output_idx = None
  }

let elu
    ?(name = "Elu")
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Elu"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_idx = None
  }

let eluGrad
    ?(name = "EluGrad")
    (gradients : ([< `float | `double ] as 't) t)
    (outputs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "EluGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P outputs ]
  ; attributes
  ; output_idx = None
  }

let encodePng
    ?(name = "EncodePng")
    ?compression
    (image : 't t)
  =
  let attributes = [ "T", Type (P image.output_type) ] in
  let attributes =
    match compression with | None -> attributes | Some compression -> ("compression", Int compression) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "EncodePng"
  ; output_type = Type.String
  ; inputs = [ P image ]
  ; attributes
  ; output_idx = None
  }

let enter
    ?(name = "Enter")
    ~frame_name
    ?is_constant
    ?parallel_iterations
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  let attributes =
    ("frame_name", String frame_name) :: attributes
  in
  let attributes =
    match is_constant with | None -> attributes | Some is_constant -> ("is_constant", Bool is_constant) :: attributes
  in
  let attributes =
    match parallel_iterations with | None -> attributes | Some parallel_iterations -> ("parallel_iterations", Int parallel_iterations) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Enter"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_idx = None
  }

let equal
    ?(name = "Equal")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Equal"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let erf
    ?(name = "Erf")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Erf"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let erfc
    ?(name = "Erfc")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Erfc"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let exit
    ?(name = "Exit")
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Exit"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_idx = None
  }

let exp
    ?(name = "Exp")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Exp"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let expandDims
    ?(name = "ExpandDims")
    (input : 't t)
    (dim : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ExpandDims"
  ; output_type = input.output_type
  ; inputs = [ P input; P dim ]
  ; attributes
  ; output_idx = None
  }

let extractGlimpse
    ?(name = "ExtractGlimpse")
    ?centered
    ?normalized
    ?uniform_noise
    (input : [ `float ] t)
    (size : [ `int32 ] t)
    (offsets : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match centered with | None -> attributes | Some centered -> ("centered", Bool centered) :: attributes
  in
  let attributes =
    match normalized with | None -> attributes | Some normalized -> ("normalized", Bool normalized) :: attributes
  in
  let attributes =
    match uniform_noise with | None -> attributes | Some uniform_noise -> ("uniform_noise", Bool uniform_noise) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ExtractGlimpse"
  ; output_type = Type.Float
  ; inputs = [ P input; P size; P offsets ]
  ; attributes
  ; output_idx = None
  }

let fFT2D
    ?(name = "FFT2D")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "FFT2D"
  ; output_type = Type.Complex64
  ; inputs = [ P in__ ]
  ; attributes
  ; output_idx = None
  }

let fIFOQueue
    ?(name = "FIFOQueue")
    ~component_types
    ?shapes
    ?capacity
    ?container
    ?shared_name
    ()
  =
  let attributes = [] in
  let attributes =
    ("component_types", List (Type component_types)) :: attributes
  in
  let attributes =
    match shapes with | None -> attributes | Some shapes -> ("shapes", List (Shape shapes)) :: attributes
  in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "FIFOQueue"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let fact
    ?(name = "Fact")
    ()
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Fact"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let fill
    ?(name = "Fill")
    (dims : [ `int32 ] t)
    (value : 't t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Fill"
  ; output_type = value.output_type
  ; inputs = [ P dims; P value ]
  ; attributes
  ; output_idx = None
  }

let fixedLengthRecordReader
    ?(name = "FixedLengthRecordReader")
    ?header_bytes
    ~record_bytes
    ?footer_bytes
    ?container
    ?shared_name
    ()
  =
  let attributes = [] in
  let attributes =
    match header_bytes with | None -> attributes | Some header_bytes -> ("header_bytes", Int header_bytes) :: attributes
  in
  let attributes =
    ("record_bytes", Int record_bytes) :: attributes
  in
  let attributes =
    match footer_bytes with | None -> attributes | Some footer_bytes -> ("footer_bytes", Int footer_bytes) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "FixedLengthRecordReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let floor
    ?(name = "Floor")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Floor"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let gather
    ?(name = "Gather")
    ?validate_indices
    (params : 'tparams t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P indices.output_type) ;  "Tparams", Type (P params.output_type) ] in
  let attributes =
    match validate_indices with | None -> attributes | Some validate_indices -> ("validate_indices", Bool validate_indices) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Gather"
  ; output_type = params.output_type
  ; inputs = [ P params; P indices ]
  ; attributes
  ; output_idx = None
  }

let greater
    ?(name = "Greater")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Greater"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let greaterEqual
    ?(name = "GreaterEqual")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "GreaterEqual"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let hSVToRGB
    ?(name = "HSVToRGB")
    (images : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "HSVToRGB"
  ; output_type = Type.Float
  ; inputs = [ P images ]
  ; attributes
  ; output_idx = None
  }

let hashTable
    ?(name = "HashTable")
    ?container
    ?shared_name
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "HashTable"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let histogramSummary
    ?(name = "HistogramSummary")
    (tag : [ `string ] t)
    (values : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P values.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "HistogramSummary"
  ; output_type = Type.String
  ; inputs = [ P tag; P values ]
  ; attributes
  ; output_idx = None
  }

let iFFT2D
    ?(name = "IFFT2D")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "IFFT2D"
  ; output_type = Type.Complex64
  ; inputs = [ P in__ ]
  ; attributes
  ; output_idx = None
  }

let identity
    ?(name = "Identity")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Identity"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let identityReader
    ?(name = "IdentityReader")
    ?container
    ?shared_name
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "IdentityReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let imag
    ?(name = "Imag")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Imag"
  ; output_type = Type.Float
  ; inputs = [ P in__ ]
  ; attributes
  ; output_idx = None
  }

let imageSummary
    ?(name = "ImageSummary")
    ?max_images
    (tag : [ `string ] t)
    (tensor : ([< `float ] as 't) t)
  =
  let attributes = [ "T", Type (P tensor.output_type) ] in
  let attributes =
    match max_images with | None -> attributes | Some max_images -> ("max_images", Int max_images) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ImageSummary"
  ; output_type = Type.String
  ; inputs = [ P tag; P tensor ]
  ; attributes
  ; output_idx = None
  }

let inTopK
    ?(name = "InTopK")
    ~k
    (predictions : [ `float ] t)
    (targets : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P targets.output_type) ] in
  let attributes =
    ("k", Int k) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "InTopK"
  ; output_type = Type.Bool
  ; inputs = [ P predictions; P targets ]
  ; attributes
  ; output_idx = None
  }

let initializeTable
    ?(name = "InitializeTable")
    (table_handle : [ `string ] t)
    (keys : 'tkey t)
    (values : 'tval t)
  =
  let attributes = [ "Tval", Type (P values.output_type) ;  "Tkey", Type (P keys.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "InitializeTable"
  ; output_type = Type.Unit
  ; inputs = [ P table_handle; P keys; P values ]
  ; attributes
  ; output_idx = None
  }

let inv
    ?(name = "Inv")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Inv"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let invertPermutation
    ?(name = "InvertPermutation")
    (x : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "InvertPermutation"
  ; output_type = Type.Int32
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let isFinite
    ?(name = "IsFinite")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "IsFinite"
  ; output_type = Type.Bool
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let isInf
    ?(name = "IsInf")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "IsInf"
  ; output_type = Type.Bool
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let isNan
    ?(name = "IsNan")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "IsNan"
  ; output_type = Type.Bool
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let l2Loss
    ?(name = "L2Loss")
    (t : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P t.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "L2Loss"
  ; output_type = t.output_type
  ; inputs = [ P t ]
  ; attributes
  ; output_idx = None
  }

let lRN
    ?(name = "LRN")
    ?depth_radius
    ?bias
    ?alpha
    ?beta
    (input : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match depth_radius with | None -> attributes | Some depth_radius -> ("depth_radius", Int depth_radius) :: attributes
  in
  let attributes =
    match bias with | None -> attributes | Some bias -> ("bias", Float bias) :: attributes
  in
  let attributes =
    match alpha with | None -> attributes | Some alpha -> ("alpha", Float alpha) :: attributes
  in
  let attributes =
    match beta with | None -> attributes | Some beta -> ("beta", Float beta) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LRN"
  ; output_type = Type.Float
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let lRNGrad
    ?(name = "LRNGrad")
    ?depth_radius
    ?bias
    ?alpha
    ?beta
    (input_grads : [ `float ] t)
    (input_image : [ `float ] t)
    (output_image : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match depth_radius with | None -> attributes | Some depth_radius -> ("depth_radius", Int depth_radius) :: attributes
  in
  let attributes =
    match bias with | None -> attributes | Some bias -> ("bias", Float bias) :: attributes
  in
  let attributes =
    match alpha with | None -> attributes | Some alpha -> ("alpha", Float alpha) :: attributes
  in
  let attributes =
    match beta with | None -> attributes | Some beta -> ("beta", Float beta) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LRNGrad"
  ; output_type = Type.Float
  ; inputs = [ P input_grads; P input_image; P output_image ]
  ; attributes
  ; output_idx = None
  }

let less
    ?(name = "Less")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Less"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let lessEqual
    ?(name = "LessEqual")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LessEqual"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let lgamma
    ?(name = "Lgamma")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Lgamma"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let linSpace
    ?(name = "LinSpace")
    (start : ([< `float | `double ] as 't) t)
    (stop : ([< `float | `double ] as 't) t)
    (num : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P start.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LinSpace"
  ; output_type = start.output_type
  ; inputs = [ P start; P stop; P num ]
  ; attributes
  ; output_idx = None
  }

let log
    ?(name = "Log")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Log"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let logicalAnd
    ?(name = "LogicalAnd")
    (x : [ `bool ] t)
    (y : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LogicalAnd"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let logicalNot
    ?(name = "LogicalNot")
    (x : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LogicalNot"
  ; output_type = Type.Bool
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let logicalOr
    ?(name = "LogicalOr")
    (x : [ `bool ] t)
    (y : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LogicalOr"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let lookupTableFind
    ?(name = "LookupTableFind")
    (table_handle : [ `string ] t)
    (keys : 'tin t)
    (default_value : 'tout t)
  =
  let attributes = [ "Tin", Type (P keys.output_type) ;  "Tout", Type (P default_value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LookupTableFind"
  ; output_type = default_value.output_type
  ; inputs = [ P table_handle; P keys; P default_value ]
  ; attributes
  ; output_idx = None
  }

let lookupTableSize
    ?(name = "LookupTableSize")
    (table_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LookupTableSize"
  ; output_type = Type.Int64
  ; inputs = [ P table_handle ]
  ; attributes
  ; output_idx = None
  }

let loopCond
    ?(name = "LoopCond")
    (input : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "LoopCond"
  ; output_type = Type.Bool
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let matMul
    ?(name = "MatMul")
    ?transpose_a
    ?transpose_b
    (a : ([< `float | `double | `int32 | `complex64 ] as 't) t)
    (b : ([< `float | `double | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P a.output_type) ] in
  let attributes =
    match transpose_a with | None -> attributes | Some transpose_a -> ("transpose_a", Bool transpose_a) :: attributes
  in
  let attributes =
    match transpose_b with | None -> attributes | Some transpose_b -> ("transpose_b", Bool transpose_b) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MatMul"
  ; output_type = a.output_type
  ; inputs = [ P a; P b ]
  ; attributes
  ; output_idx = None
  }

let matchingFiles
    ?(name = "MatchingFiles")
    (pattern : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MatchingFiles"
  ; output_type = Type.String
  ; inputs = [ P pattern ]
  ; attributes
  ; output_idx = None
  }

let matrixDeterminant
    ?(name = "MatrixDeterminant")
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MatrixDeterminant"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let matrixInverse
    ?(name = "MatrixInverse")
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MatrixInverse"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let matrixSolve
    ?(name = "MatrixSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MatrixSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes
  ; output_idx = None
  }

let matrixSolveLs
    ?(name = "MatrixSolveLs")
    ?fast
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
    (l2_regularizer : [ `double ] t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  let attributes =
    match fast with | None -> attributes | Some fast -> ("fast", Bool fast) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MatrixSolveLs"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs; P l2_regularizer ]
  ; attributes
  ; output_idx = None
  }

let matrixTriangularSolve
    ?(name = "MatrixTriangularSolve")
    ?lower
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  let attributes =
    match lower with | None -> attributes | Some lower -> ("lower", Bool lower) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MatrixTriangularSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes
  ; output_idx = None
  }

let max
    ?(name = "Max")
    ?keep_dims
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Max"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_idx = None
  }

let maxPool
    ?(name = "MaxPool")
    ~ksize
    ~strides
    ~padding
    ?data_format
    (input : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("ksize", List (Int ksize)) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MaxPool"
  ; output_type = Type.Float
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let maxPoolGrad
    ?(name = "MaxPoolGrad")
    ~ksize
    ~strides
    ~padding
    ?data_format
    (orig_input : [ `float ] t)
    (orig_output : [ `float ] t)
    (grad : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("ksize", List (Int ksize)) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MaxPoolGrad"
  ; output_type = Type.Float
  ; inputs = [ P orig_input; P orig_output; P grad ]
  ; attributes
  ; output_idx = None
  }

let maxPoolGradWithArgmax
    ?(name = "MaxPoolGradWithArgmax")
    ~ksize
    ~strides
    ~padding
    (input : [ `float ] t)
    (grad : [ `float ] t)
    (argmax : ([< `int32 | `int64 ] as 'targmax) t)
  =
  let attributes = [ "Targmax", Type (P argmax.output_type) ] in
  let attributes =
    ("ksize", List (Int ksize)) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MaxPoolGradWithArgmax"
  ; output_type = Type.Float
  ; inputs = [ P input; P grad; P argmax ]
  ; attributes
  ; output_idx = None
  }

let maximum
    ?(name = "Maximum")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Maximum"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let mean
    ?(name = "Mean")
    ?keep_dims
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Mean"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_idx = None
  }

let mergeSummary
    ?(name = "MergeSummary")
    (inputs : [ `string ] t list)
  =
  let attributes = [] in
  let attributes =
    ("N", Int (List.length inputs)) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MergeSummary"
  ; output_type = Type.String
  ; inputs = List.map (fun n -> P n) inputs
  ; attributes
  ; output_idx = None
  }

let min
    ?(name = "Min")
    ?keep_dims
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Min"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_idx = None
  }

let minimum
    ?(name = "Minimum")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Minimum"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let mirrorPad
    ?(name = "MirrorPad")
    ~mode
    (input : 't t)
    (paddings : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("mode", String mode) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MirrorPad"
  ; output_type = input.output_type
  ; inputs = [ P input; P paddings ]
  ; attributes
  ; output_idx = None
  }

let mirrorPadGrad
    ?(name = "MirrorPadGrad")
    ~mode
    (input : 't t)
    (paddings : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("mode", String mode) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "MirrorPadGrad"
  ; output_type = input.output_type
  ; inputs = [ P input; P paddings ]
  ; attributes
  ; output_idx = None
  }

let mod_
    ?(name = "Mod")
    (x : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (y : ([< `int32 | `int64 | `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Mod"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let mul
    ?(name = "Mul")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Mul"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let neg
    ?(name = "Neg")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Neg"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let negTrain
    ?(name = "NegTrain")
    ~vocab_count
    ~num_negative_samples
    (w_in : [ `float ] t)
    (w_out : [ `float ] t)
    (examples : [ `int32 ] t)
    (labels : [ `int32 ] t)
    (lr : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("vocab_count", List (Int vocab_count)) :: attributes
  in
  let attributes =
    ("num_negative_samples", Int num_negative_samples) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "NegTrain"
  ; output_type = Type.Unit
  ; inputs = [ P w_in; P w_out; P examples; P labels; P lr ]
  ; attributes
  ; output_idx = None
  }

let nextIteration
    ?(name = "NextIteration")
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "NextIteration"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_idx = None
  }

let noOp
    ?(name = "NoOp")
    ()
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "NoOp"
  ; output_type = Type.Unit
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let notEqual
    ?(name = "NotEqual")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "NotEqual"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let oneHot
    ?(name = "OneHot")
    ?axis
    (indices : [ `int64 ] t)
    (depth : [ `int32 ] t)
    (on_value : 't t)
    (off_value : 't t)
  =
  let attributes = [ "T", Type (P on_value.output_type) ] in
  let attributes =
    match axis with | None -> attributes | Some axis -> ("axis", Int axis) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "OneHot"
  ; output_type = on_value.output_type
  ; inputs = [ P indices; P depth; P on_value; P off_value ]
  ; attributes
  ; output_idx = None
  }

let pack
    ?(name = "Pack")
    (values : 't t list)
  =
  let attributes = [ "T", Type (P (List.hd values).output_type) ] in
  let attributes =
    ("N", Int (List.length values)) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Pack"
  ; output_type = (List.hd values).output_type
  ; inputs = List.map (fun n -> P n) values
  ; attributes
  ; output_idx = None
  }

let pad
    ?(name = "Pad")
    (input : 't t)
    (paddings : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Pad"
  ; output_type = input.output_type
  ; inputs = [ P input; P paddings ]
  ; attributes
  ; output_idx = None
  }

let paddingFIFOQueue
    ?(name = "PaddingFIFOQueue")
    ~component_types
    ?shapes
    ?capacity
    ?container
    ?shared_name
    ()
  =
  let attributes = [] in
  let attributes =
    ("component_types", List (Type component_types)) :: attributes
  in
  let attributes =
    match shapes with | None -> attributes | Some shapes -> ("shapes", List (Shape shapes)) :: attributes
  in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "PaddingFIFOQueue"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let placeholder
    ?(name = "Placeholder")
    ~type_
    ?shape
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match shape with | None -> attributes | Some shape -> ("shape", Shape shape) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Placeholder"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let pow
    ?(name = "Pow")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Pow"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let prod
    ?(name = "Prod")
    ?keep_dims
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Prod"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_idx = None
  }

let queueClose
    ?(name = "QueueClose")
    ?cancel_pending_enqueues
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match cancel_pending_enqueues with | None -> attributes | Some cancel_pending_enqueues -> ("cancel_pending_enqueues", Bool cancel_pending_enqueues) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "QueueClose"
  ; output_type = Type.Unit
  ; inputs = [ P handle ]
  ; attributes
  ; output_idx = None
  }

let queueSize
    ?(name = "QueueSize")
    (handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "QueueSize"
  ; output_type = Type.Int32
  ; inputs = [ P handle ]
  ; attributes
  ; output_idx = None
  }

let rGBToHSV
    ?(name = "RGBToHSV")
    (images : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RGBToHSV"
  ; output_type = Type.Float
  ; inputs = [ P images ]
  ; attributes
  ; output_idx = None
  }

let randomCrop
    ?(name = "RandomCrop")
    ?seed
    ?seed2
    (image : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P image.output_type) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RandomCrop"
  ; output_type = image.output_type
  ; inputs = [ P image; P size ]
  ; attributes
  ; output_idx = None
  }

let randomShuffle
    ?(name = "RandomShuffle")
    ?seed
    ?seed2
    (value : 't t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RandomShuffle"
  ; output_type = value.output_type
  ; inputs = [ P value ]
  ; attributes
  ; output_idx = None
  }

let randomShuffleQueue
    ?(name = "RandomShuffleQueue")
    ~component_types
    ?shapes
    ?capacity
    ?min_after_dequeue
    ?seed
    ?seed2
    ?container
    ?shared_name
    ()
  =
  let attributes = [] in
  let attributes =
    ("component_types", List (Type component_types)) :: attributes
  in
  let attributes =
    match shapes with | None -> attributes | Some shapes -> ("shapes", List (Shape shapes)) :: attributes
  in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match min_after_dequeue with | None -> attributes | Some min_after_dequeue -> ("min_after_dequeue", Int min_after_dequeue) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RandomShuffleQueue"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let randomStandardNormal
    ?(name = "RandomStandardNormal")
    ~type_
    ?seed
    ?seed2
    (shape : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P shape.output_type) ;  "dtype", Type (P type_) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RandomStandardNormal"
  ; output_type = type_
  ; inputs = [ P shape ]
  ; attributes
  ; output_idx = None
  }

let randomUniform
    ?(name = "RandomUniform")
    ~type_
    ?seed
    ?seed2
    (shape : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P shape.output_type) ;  "dtype", Type (P type_) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RandomUniform"
  ; output_type = type_
  ; inputs = [ P shape ]
  ; attributes
  ; output_idx = None
  }

let randomUniformInt
    ?(name = "RandomUniformInt")
    ?seed
    ?seed2
    (shape : ([< `int32 | `int64 ] as 't) t)
    (minval : ([< `int32 | `int64 ] as 'tout) t)
    (maxval : ([< `int32 | `int64 ] as 'tout) t)
  =
  let attributes = [ "T", Type (P shape.output_type) ;  "Tout", Type (P minval.output_type) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RandomUniformInt"
  ; output_type = minval.output_type
  ; inputs = [ P shape; P minval; P maxval ]
  ; attributes
  ; output_idx = None
  }

let range
    ?(name = "Range")
    (start : [ `int32 ] t)
    (limit : [ `int32 ] t)
    (delta : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Range"
  ; output_type = Type.Int32
  ; inputs = [ P start; P limit; P delta ]
  ; attributes
  ; output_idx = None
  }

let rank
    ?(name = "Rank")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Rank"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let readFile
    ?(name = "ReadFile")
    (filename : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ReadFile"
  ; output_type = Type.String
  ; inputs = [ P filename ]
  ; attributes
  ; output_idx = None
  }

let readerNumRecordsProduced
    ?(name = "ReaderNumRecordsProduced")
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ReaderNumRecordsProduced"
  ; output_type = Type.Int64
  ; inputs = [ P reader_handle ]
  ; attributes
  ; output_idx = None
  }

let readerNumWorkUnitsCompleted
    ?(name = "ReaderNumWorkUnitsCompleted")
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ReaderNumWorkUnitsCompleted"
  ; output_type = Type.Int64
  ; inputs = [ P reader_handle ]
  ; attributes
  ; output_idx = None
  }

let readerReset
    ?(name = "ReaderReset")
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ReaderReset"
  ; output_type = Type.Unit
  ; inputs = [ P reader_handle ]
  ; attributes
  ; output_idx = None
  }

let readerRestoreState
    ?(name = "ReaderRestoreState")
    (reader_handle : [ `string ] t)
    (state : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ReaderRestoreState"
  ; output_type = Type.Unit
  ; inputs = [ P reader_handle; P state ]
  ; attributes
  ; output_idx = None
  }

let readerSerializeState
    ?(name = "ReaderSerializeState")
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ReaderSerializeState"
  ; output_type = Type.String
  ; inputs = [ P reader_handle ]
  ; attributes
  ; output_idx = None
  }

let real
    ?(name = "Real")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Real"
  ; output_type = Type.Float
  ; inputs = [ P in__ ]
  ; attributes
  ; output_idx = None
  }

let refEnter
    ?(name = "RefEnter")
    ~frame_name
    ?is_constant
    ?parallel_iterations
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  let attributes =
    ("frame_name", String frame_name) :: attributes
  in
  let attributes =
    match is_constant with | None -> attributes | Some is_constant -> ("is_constant", Bool is_constant) :: attributes
  in
  let attributes =
    match parallel_iterations with | None -> attributes | Some parallel_iterations -> ("parallel_iterations", Int parallel_iterations) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RefEnter"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_idx = None
  }

let refExit
    ?(name = "RefExit")
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RefExit"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_idx = None
  }

let refIdentity
    ?(name = "RefIdentity")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RefIdentity"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let refNextIteration
    ?(name = "RefNextIteration")
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RefNextIteration"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_idx = None
  }

let refSelect
    ?(name = "RefSelect")
    (index : [ `int32 ] t)
    (inputs : 't t list)
  =
  let attributes = [ "T", Type (P (List.hd inputs).output_type) ] in
  let attributes =
    ("N", Int (List.length inputs)) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RefSelect"
  ; output_type = (List.hd inputs).output_type
  ; inputs = [ P index ] @ List.map (fun n -> P n) inputs
  ; attributes
  ; output_idx = None
  }

let relu
    ?(name = "Relu")
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Relu"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_idx = None
  }

let relu6
    ?(name = "Relu6")
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Relu6"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_idx = None
  }

let relu6Grad
    ?(name = "Relu6Grad")
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Relu6Grad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes
  ; output_idx = None
  }

let reluGrad
    ?(name = "ReluGrad")
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ReluGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes
  ; output_idx = None
  }

let reshape
    ?(name = "Reshape")
    (tensor : 't t)
    (shape : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P tensor.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Reshape"
  ; output_type = tensor.output_type
  ; inputs = [ P tensor; P shape ]
  ; attributes
  ; output_idx = None
  }

let resizeArea
    ?(name = "ResizeArea")
    ?align_corners
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P images.output_type) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ResizeArea"
  ; output_type = Type.Float
  ; inputs = [ P images; P size ]
  ; attributes
  ; output_idx = None
  }

let resizeBicubic
    ?(name = "ResizeBicubic")
    ?align_corners
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P images.output_type) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ResizeBicubic"
  ; output_type = Type.Float
  ; inputs = [ P images; P size ]
  ; attributes
  ; output_idx = None
  }

let resizeBilinear
    ?(name = "ResizeBilinear")
    ?align_corners
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P images.output_type) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ResizeBilinear"
  ; output_type = Type.Float
  ; inputs = [ P images; P size ]
  ; attributes
  ; output_idx = None
  }

let resizeBilinearGrad
    ?(name = "ResizeBilinearGrad")
    ?align_corners
    (grads : [ `float ] t)
    (original_image : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P original_image.output_type) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ResizeBilinearGrad"
  ; output_type = original_image.output_type
  ; inputs = [ P grads; P original_image ]
  ; attributes
  ; output_idx = None
  }

let resizeNearestNeighbor
    ?(name = "ResizeNearestNeighbor")
    ?align_corners
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P images.output_type) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ResizeNearestNeighbor"
  ; output_type = images.output_type
  ; inputs = [ P images; P size ]
  ; attributes
  ; output_idx = None
  }

let resizeNearestNeighborGrad
    ?(name = "ResizeNearestNeighborGrad")
    ?align_corners
    (grads : ([< `int32 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P grads.output_type) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ResizeNearestNeighborGrad"
  ; output_type = grads.output_type
  ; inputs = [ P grads; P size ]
  ; attributes
  ; output_idx = None
  }

let restore
    ?(name = "Restore")
    ~type_
    ?preferred_shard
    (file_pattern : [ `string ] t)
    (tensor_name : [ `string ] t)
  =
  let attributes = [ "dt", Type (P type_) ] in
  let attributes =
    match preferred_shard with | None -> attributes | Some preferred_shard -> ("preferred_shard", Int preferred_shard) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Restore"
  ; output_type = type_
  ; inputs = [ P file_pattern; P tensor_name ]
  ; attributes
  ; output_idx = None
  }

let restoreSlice
    ?(name = "RestoreSlice")
    ~type_
    ?preferred_shard
    (file_pattern : [ `string ] t)
    (tensor_name : [ `string ] t)
    (shape_and_slice : [ `string ] t)
  =
  let attributes = [ "dt", Type (P type_) ] in
  let attributes =
    match preferred_shard with | None -> attributes | Some preferred_shard -> ("preferred_shard", Int preferred_shard) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "RestoreSlice"
  ; output_type = type_
  ; inputs = [ P file_pattern; P tensor_name; P shape_and_slice ]
  ; attributes
  ; output_idx = None
  }

let reverse
    ?(name = "Reverse")
    (tensor : ([< `int32 | `bool | `float | `double ] as 't) t)
    (dims : [ `bool ] t)
  =
  let attributes = [ "T", Type (P tensor.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Reverse"
  ; output_type = tensor.output_type
  ; inputs = [ P tensor; P dims ]
  ; attributes
  ; output_idx = None
  }

let reverseSequence
    ?(name = "ReverseSequence")
    ~seq_dim
    ?batch_dim
    (input : 't t)
    (seq_lengths : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("seq_dim", Int seq_dim) :: attributes
  in
  let attributes =
    match batch_dim with | None -> attributes | Some batch_dim -> ("batch_dim", Int batch_dim) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ReverseSequence"
  ; output_type = input.output_type
  ; inputs = [ P input; P seq_lengths ]
  ; attributes
  ; output_idx = None
  }

let rsqrt
    ?(name = "Rsqrt")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Rsqrt"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let scalarSummary
    ?(name = "ScalarSummary")
    (tags : [ `string ] t)
    (values : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P values.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ScalarSummary"
  ; output_type = Type.String
  ; inputs = [ P tags; P values ]
  ; attributes
  ; output_idx = None
  }

let scatterAdd
    ?(name = "ScatterAdd")
    ?use_locking
    (ref : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P indices.output_type) ;  "T", Type (P ref.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ScatterAdd"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P indices; P updates ]
  ; attributes
  ; output_idx = None
  }

let scatterSub
    ?(name = "ScatterSub")
    ?use_locking
    (ref : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P indices.output_type) ;  "T", Type (P ref.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ScatterSub"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P indices; P updates ]
  ; attributes
  ; output_idx = None
  }

let scatterUpdate
    ?(name = "ScatterUpdate")
    ?use_locking
    (ref : 't t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : 't t)
  =
  let attributes = [ "Tindices", Type (P indices.output_type) ;  "T", Type (P ref.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ScatterUpdate"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P indices; P updates ]
  ; attributes
  ; output_idx = None
  }

let segmentMax
    ?(name = "SegmentMax")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P segment_ids.output_type) ;  "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SegmentMax"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_idx = None
  }

let segmentMean
    ?(name = "SegmentMean")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P segment_ids.output_type) ;  "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SegmentMean"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_idx = None
  }

let segmentMin
    ?(name = "SegmentMin")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P segment_ids.output_type) ;  "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SegmentMin"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_idx = None
  }

let segmentProd
    ?(name = "SegmentProd")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P segment_ids.output_type) ;  "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SegmentProd"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_idx = None
  }

let segmentSum
    ?(name = "SegmentSum")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P segment_ids.output_type) ;  "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SegmentSum"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_idx = None
  }

let select
    ?(name = "Select")
    (condition : [ `bool ] t)
    (t : 't t)
    (e : 't t)
  =
  let attributes = [ "T", Type (P t.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Select"
  ; output_type = t.output_type
  ; inputs = [ P condition; P t; P e ]
  ; attributes
  ; output_idx = None
  }

let selfAdjointEig
    ?(name = "SelfAdjointEig")
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SelfAdjointEig"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let serializeManySparse
    ?(name = "SerializeManySparse")
    (sparse_indices : [ `int64 ] t)
    (sparse_values : 't t)
    (sparse_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P sparse_values.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SerializeManySparse"
  ; output_type = Type.String
  ; inputs = [ P sparse_indices; P sparse_values; P sparse_shape ]
  ; attributes
  ; output_idx = None
  }

let serializeSparse
    ?(name = "SerializeSparse")
    (sparse_indices : [ `int64 ] t)
    (sparse_values : 't t)
    (sparse_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P sparse_values.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SerializeSparse"
  ; output_type = Type.String
  ; inputs = [ P sparse_indices; P sparse_values; P sparse_shape ]
  ; attributes
  ; output_idx = None
  }

let shape
    ?(name = "Shape")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Shape"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let shapeN
    ?(name = "ShapeN")
    (input : 't t list)
  =
  let attributes = [ "T", Type (P (List.hd input).output_type) ] in
  let attributes =
    ("N", Int (List.length input)) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ShapeN"
  ; output_type = Type.Int32
  ; inputs = List.map (fun n -> P n) input
  ; attributes
  ; output_idx = None
  }

let shardedFilename
    ?(name = "ShardedFilename")
    (basename : [ `string ] t)
    (shard : [ `int32 ] t)
    (num_shards : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ShardedFilename"
  ; output_type = Type.String
  ; inputs = [ P basename; P shard; P num_shards ]
  ; attributes
  ; output_idx = None
  }

let shardedFilespec
    ?(name = "ShardedFilespec")
    (basename : [ `string ] t)
    (num_shards : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ShardedFilespec"
  ; output_type = Type.String
  ; inputs = [ P basename; P num_shards ]
  ; attributes
  ; output_idx = None
  }

let sigmoid
    ?(name = "Sigmoid")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Sigmoid"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let sign
    ?(name = "Sign")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Sign"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let sin
    ?(name = "Sin")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Sin"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let size
    ?(name = "Size")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Size"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let slice
    ?(name = "Slice")
    (input : 't t)
    (begin__ : ([< `int32 | `int64 ] as 'index) t)
    (size : ([< `int32 | `int64 ] as 'index) t)
  =
  let attributes = [ "Index", Type (P begin__.output_type) ;  "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Slice"
  ; output_type = input.output_type
  ; inputs = [ P input; P begin__; P size ]
  ; attributes
  ; output_idx = None
  }

let softmax
    ?(name = "Softmax")
    (logits : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P logits.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Softmax"
  ; output_type = logits.output_type
  ; inputs = [ P logits ]
  ; attributes
  ; output_idx = None
  }

let softplus
    ?(name = "Softplus")
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Softplus"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_idx = None
  }

let softplusGrad
    ?(name = "SoftplusGrad")
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SoftplusGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes
  ; output_idx = None
  }

let softsign
    ?(name = "Softsign")
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Softsign"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_idx = None
  }

let softsignGrad
    ?(name = "SoftsignGrad")
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SoftsignGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes
  ; output_idx = None
  }

let spaceToDepth
    ?(name = "SpaceToDepth")
    ~block_size
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("block_size", Int block_size) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SpaceToDepth"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let sparseApplyAdagrad
    ?(name = "SparseApplyAdagrad")
    ?use_locking
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P indices.output_type) ;  "T", Type (P var.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseApplyAdagrad"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P indices ]
  ; attributes
  ; output_idx = None
  }

let sparseApplyFtrl
    ?(name = "SparseApplyFtrl")
    ?use_locking
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (linear : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (l1 : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (l2 : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr_power : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P indices.output_type) ;  "T", Type (P var.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseApplyFtrl"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P linear; P grad; P indices; P lr; P l1; P l2; P lr_power ]
  ; attributes
  ; output_idx = None
  }

let sparseApplyMomentum
    ?(name = "SparseApplyMomentum")
    ?use_locking
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (momentum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P indices.output_type) ;  "T", Type (P var.output_type) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseApplyMomentum"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P indices; P momentum ]
  ; attributes
  ; output_idx = None
  }

let sparseMatMul
    ?(name = "SparseMatMul")
    ?transpose_a
    ?transpose_b
    ?a_is_sparse
    ?b_is_sparse
    (a : [ `float ] t)
    (b : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match transpose_a with | None -> attributes | Some transpose_a -> ("transpose_a", Bool transpose_a) :: attributes
  in
  let attributes =
    match transpose_b with | None -> attributes | Some transpose_b -> ("transpose_b", Bool transpose_b) :: attributes
  in
  let attributes =
    match a_is_sparse with | None -> attributes | Some a_is_sparse -> ("a_is_sparse", Bool a_is_sparse) :: attributes
  in
  let attributes =
    match b_is_sparse with | None -> attributes | Some b_is_sparse -> ("b_is_sparse", Bool b_is_sparse) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseMatMul"
  ; output_type = Type.Float
  ; inputs = [ P a; P b ]
  ; attributes
  ; output_idx = None
  }

let sparseSegmentMean
    ?(name = "SparseSegmentMean")
    (data : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseSegmentMean"
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
  ; attributes
  ; output_idx = None
  }

let sparseSegmentMeanGrad
    ?(name = "SparseSegmentMeanGrad")
    (grad : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
    (output_dim0 : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P grad.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseSegmentMeanGrad"
  ; output_type = grad.output_type
  ; inputs = [ P grad; P indices; P segment_ids; P output_dim0 ]
  ; attributes
  ; output_idx = None
  }

let sparseSegmentSqrtN
    ?(name = "SparseSegmentSqrtN")
    (data : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseSegmentSqrtN"
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
  ; attributes
  ; output_idx = None
  }

let sparseSegmentSqrtNGrad
    ?(name = "SparseSegmentSqrtNGrad")
    (grad : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
    (output_dim0 : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P grad.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseSegmentSqrtNGrad"
  ; output_type = grad.output_type
  ; inputs = [ P grad; P indices; P segment_ids; P output_dim0 ]
  ; attributes
  ; output_idx = None
  }

let sparseSegmentSum
    ?(name = "SparseSegmentSum")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseSegmentSum"
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
  ; attributes
  ; output_idx = None
  }

let sparseTensorDenseMatMul
    ?(name = "SparseTensorDenseMatMul")
    ?adjoint_a
    ?adjoint_b
    (a_indices : [ `int64 ] t)
    (a_values : 't t)
    (a_shape : [ `int64 ] t)
    (b : 't t)
  =
  let attributes = [ "T", Type (P a_values.output_type) ] in
  let attributes =
    match adjoint_a with | None -> attributes | Some adjoint_a -> ("adjoint_a", Bool adjoint_a) :: attributes
  in
  let attributes =
    match adjoint_b with | None -> attributes | Some adjoint_b -> ("adjoint_b", Bool adjoint_b) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseTensorDenseMatMul"
  ; output_type = a_values.output_type
  ; inputs = [ P a_indices; P a_values; P a_shape; P b ]
  ; attributes
  ; output_idx = None
  }

let sparseToDense
    ?(name = "SparseToDense")
    ?validate_indices
    (sparse_indices : ([< `int32 | `int64 ] as 'tindices) t)
    (output_shape : ([< `int32 | `int64 ] as 'tindices) t)
    (sparse_values : 't t)
    (default_value : 't t)
  =
  let attributes = [ "Tindices", Type (P sparse_indices.output_type) ;  "T", Type (P sparse_values.output_type) ] in
  let attributes =
    match validate_indices with | None -> attributes | Some validate_indices -> ("validate_indices", Bool validate_indices) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SparseToDense"
  ; output_type = sparse_values.output_type
  ; inputs = [ P sparse_indices; P output_shape; P sparse_values; P default_value ]
  ; attributes
  ; output_idx = None
  }

let split
    ?(name = "Split")
    ~num_split
    (split_dim : [ `int32 ] t)
    (value : 't t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  let attributes =
    ("num_split", Int num_split) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Split"
  ; output_type = value.output_type
  ; inputs = [ P split_dim; P value ]
  ; attributes
  ; output_idx = None
  }

let sqrt
    ?(name = "Sqrt")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Sqrt"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let square
    ?(name = "Square")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Square"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let squaredDifference
    ?(name = "SquaredDifference")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "SquaredDifference"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let squeeze
    ?(name = "Squeeze")
    ?squeeze_dims
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    match squeeze_dims with | None -> attributes | Some squeeze_dims -> ("squeeze_dims", List (Int squeeze_dims)) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Squeeze"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let stack
    ?(name = "Stack")
    ?stack_name
    ()
  =
  let attributes = [] in
  let attributes =
    match stack_name with | None -> attributes | Some stack_name -> ("stack_name", String stack_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Stack"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let stackClose
    ?(name = "StackClose")
    (handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "StackClose"
  ; output_type = Type.Unit
  ; inputs = [ P handle ]
  ; attributes
  ; output_idx = None
  }

let stackPop
    ?(name = "StackPop")
    ~type_
    (handle : [ `string ] t)
  =
  let attributes = [ "elem_type", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "StackPop"
  ; output_type = type_
  ; inputs = [ P handle ]
  ; attributes
  ; output_idx = None
  }

let stackPush
    ?(name = "StackPush")
    ?swap_memory
    (handle : [ `string ] t)
    (elem : 't t)
  =
  let attributes = [ "T", Type (P elem.output_type) ] in
  let attributes =
    match swap_memory with | None -> attributes | Some swap_memory -> ("swap_memory", Bool swap_memory) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "StackPush"
  ; output_type = elem.output_type
  ; inputs = [ P handle; P elem ]
  ; attributes
  ; output_idx = None
  }

let stopGradient
    ?(name = "StopGradient")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "StopGradient"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let stringToHashBucket
    ?(name = "StringToHashBucket")
    ~num_buckets
    (string_tensor : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_buckets", Int num_buckets) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "StringToHashBucket"
  ; output_type = Type.Int64
  ; inputs = [ P string_tensor ]
  ; attributes
  ; output_idx = None
  }

let stringToNumber
    ?(name = "StringToNumber")
    ~type_
    (string_tensor : [ `string ] t)
  =
  let attributes = [ "out_type", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "StringToNumber"
  ; output_type = type_
  ; inputs = [ P string_tensor ]
  ; attributes
  ; output_idx = None
  }

let sub
    ?(name = "Sub")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Sub"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_idx = None
  }

let sum
    ?(name = "Sum")
    ?keep_dims
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Sum"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_idx = None
  }

let tFRecordReader
    ?(name = "TFRecordReader")
    ?container
    ?shared_name
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TFRecordReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let tanh
    ?(name = "Tanh")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Tanh"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

let temporaryVariable
    ?(name = "TemporaryVariable")
    ~type_
    ~shape
    ?var_name
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let attributes =
    match var_name with | None -> attributes | Some var_name -> ("var_name", String var_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TemporaryVariable"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let tensorArray
    ?(name = "TensorArray")
    ?dynamic_size
    ?tensor_array_name
    (size : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match dynamic_size with | None -> attributes | Some dynamic_size -> ("dynamic_size", Bool dynamic_size) :: attributes
  in
  let attributes =
    match tensor_array_name with | None -> attributes | Some tensor_array_name -> ("tensor_array_name", String tensor_array_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TensorArray"
  ; output_type = Type.String
  ; inputs = [ P size ]
  ; attributes
  ; output_idx = None
  }

let tensorArrayClose
    ?(name = "TensorArrayClose")
    (handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TensorArrayClose"
  ; output_type = Type.Unit
  ; inputs = [ P handle ]
  ; attributes
  ; output_idx = None
  }

let tensorArrayGrad
    ?(name = "TensorArrayGrad")
    ~source
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("source", String source) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TensorArrayGrad"
  ; output_type = Type.String
  ; inputs = [ P handle; P flow_in ]
  ; attributes
  ; output_idx = None
  }

let tensorArrayPack
    ?(name = "TensorArrayPack")
    ~type_
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TensorArrayPack"
  ; output_type = type_
  ; inputs = [ P handle; P flow_in ]
  ; attributes
  ; output_idx = None
  }

let tensorArrayRead
    ?(name = "TensorArrayRead")
    ~type_
    (handle : [ `string ] t)
    (index : [ `int32 ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TensorArrayRead"
  ; output_type = type_
  ; inputs = [ P handle; P index; P flow_in ]
  ; attributes
  ; output_idx = None
  }

let tensorArraySize
    ?(name = "TensorArraySize")
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TensorArraySize"
  ; output_type = Type.Int32
  ; inputs = [ P handle; P flow_in ]
  ; attributes
  ; output_idx = None
  }

let tensorArraySplit
    ?(name = "TensorArraySplit")
    (handle : [ `string ] t)
    (value : 't t)
    (lengths : [ `int64 ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TensorArraySplit"
  ; output_type = Type.Float
  ; inputs = [ P handle; P value; P lengths; P flow_in ]
  ; attributes
  ; output_idx = None
  }

let tensorArrayUnpack
    ?(name = "TensorArrayUnpack")
    (handle : [ `string ] t)
    (value : 't t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TensorArrayUnpack"
  ; output_type = Type.Float
  ; inputs = [ P handle; P value; P flow_in ]
  ; attributes
  ; output_idx = None
  }

let tensorArrayWrite
    ?(name = "TensorArrayWrite")
    (handle : [ `string ] t)
    (index : [ `int32 ] t)
    (value : 't t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TensorArrayWrite"
  ; output_type = Type.Float
  ; inputs = [ P handle; P index; P value; P flow_in ]
  ; attributes
  ; output_idx = None
  }

let textLineReader
    ?(name = "TextLineReader")
    ?skip_header_lines
    ?container
    ?shared_name
    ()
  =
  let attributes = [] in
  let attributes =
    match skip_header_lines with | None -> attributes | Some skip_header_lines -> ("skip_header_lines", Int skip_header_lines) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TextLineReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let tile
    ?(name = "Tile")
    (input : 't t)
    (multiples : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Tile"
  ; output_type = input.output_type
  ; inputs = [ P input; P multiples ]
  ; attributes
  ; output_idx = None
  }

let tileGrad
    ?(name = "TileGrad")
    (input : 't t)
    (multiples : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TileGrad"
  ; output_type = input.output_type
  ; inputs = [ P input; P multiples ]
  ; attributes
  ; output_idx = None
  }

let transpose
    ?(name = "Transpose")
    (x : 't t)
    (perm : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Transpose"
  ; output_type = x.output_type
  ; inputs = [ P x; P perm ]
  ; attributes
  ; output_idx = None
  }

let truncatedNormal
    ?(name = "TruncatedNormal")
    ~type_
    ?seed
    ?seed2
    (shape : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P shape.output_type) ;  "dtype", Type (P type_) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "TruncatedNormal"
  ; output_type = type_
  ; inputs = [ P shape ]
  ; attributes
  ; output_idx = None
  }

let unpack
    ?(name = "Unpack")
    ~num
    (value : 't t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  let attributes =
    ("num", Int num) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Unpack"
  ; output_type = value.output_type
  ; inputs = [ P value ]
  ; attributes
  ; output_idx = None
  }

let unsortedSegmentSum
    ?(name = "UnsortedSegmentSum")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
    (num_segments : [ `int32 ] t)
  =
  let attributes = [ "Tindices", Type (P segment_ids.output_type) ;  "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "UnsortedSegmentSum"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids; P num_segments ]
  ; attributes
  ; output_idx = None
  }

let variable
    ?(name = "Variable")
    ~type_
    ~shape
    ?container
    ?shared_name
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Variable"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let where
    ?(name = "Where")
    (input : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "Where"
  ; output_type = Type.Int64
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let wholeFileReader
    ?(name = "WholeFileReader")
    ?container
    ?shared_name
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "WholeFileReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_idx = None
  }

let zerosLike
    ?(name = "ZerosLike")
    (x : 't t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_name.of_string "ZerosLike"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

