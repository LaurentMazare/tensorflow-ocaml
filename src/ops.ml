open Node

let abs
    ?(name = "Abs")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Abs"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let add
    ?(name = "Add")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Add"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let addN
    ?(name = "AddN")
    (inputs : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P inputs.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "AddN"
  ; output_type = inputs.output_type
  ; inputs = [ P inputs ]
  ; attributes
  ; output_name = None
  }

let adjustContrast
    ?(name = "AdjustContrast")
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (contrast_factor : [ `float ] t)
    (min_value : [ `float ] t)
    (max_value : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "AdjustContrast"
  ; output_type = Type.Float
  ; inputs = [ P images; P contrast_factor; P min_value; P max_value ]
  ; attributes
  ; output_name = None
  }

let adjustContrastv2
    ?(name = "AdjustContrastv2")
    (images : [ `float ] t)
    (contrast_factor : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "AdjustContrastv2"
  ; output_type = Type.Float
  ; inputs = [ P images; P contrast_factor ]
  ; attributes
  ; output_name = None
  }

let all
    ?(name = "All")
    (input : [ `bool ] t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "All"
  ; output_type = Type.Bool
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_name = None
  }

let any
    ?(name = "Any")
    (input : [ `bool ] t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Any"
  ; output_type = Type.Bool
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_name = None
  }

let applyAdagrad
    ?(name = "ApplyAdagrad")
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ApplyAdagrad"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad ]
  ; attributes
  ; output_name = None
  }

let applyAdam
    ?(name = "ApplyAdam")
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
  { name = Name.make_fresh ~name
  ; op_name = "ApplyAdam"
  ; output_type = var.output_type
  ; inputs = [ P var; P m; P v; P beta1_power; P beta2_power; P lr; P beta1; P beta2; P epsilon; P grad ]
  ; attributes
  ; output_name = None
  }

let applyFtrl
    ?(name = "ApplyFtrl")
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
  { name = Name.make_fresh ~name
  ; op_name = "ApplyFtrl"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P linear; P grad; P lr; P l1; P l2; P lr_power ]
  ; attributes
  ; output_name = None
  }

let applyGradientDescent
    ?(name = "ApplyGradientDescent")
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (alpha : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (delta : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ApplyGradientDescent"
  ; output_type = var.output_type
  ; inputs = [ P var; P alpha; P delta ]
  ; attributes
  ; output_name = None
  }

let applyMomentum
    ?(name = "ApplyMomentum")
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (momentum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ApplyMomentum"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P momentum ]
  ; attributes
  ; output_name = None
  }

let applyRMSProp
    ?(name = "ApplyRMSProp")
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
  { name = Name.make_fresh ~name
  ; op_name = "ApplyRMSProp"
  ; output_type = var.output_type
  ; inputs = [ P var; P ms; P mom; P lr; P rho; P momentum; P epsilon; P grad ]
  ; attributes
  ; output_name = None
  }

let argMax
    ?(name = "ArgMax")
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (dimension : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ArgMax"
  ; output_type = Type.Int64
  ; inputs = [ P input; P dimension ]
  ; attributes
  ; output_name = None
  }

let argMin
    ?(name = "ArgMin")
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (dimension : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ArgMin"
  ; output_type = Type.Int64
  ; inputs = [ P input; P dimension ]
  ; attributes
  ; output_name = None
  }

let assign
    ?(name = "Assign")
    (ref : 't t)
    (value : 't t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Assign"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P value ]
  ; attributes
  ; output_name = None
  }

let assignAdd
    ?(name = "AssignAdd")
    (ref : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (value : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "AssignAdd"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P value ]
  ; attributes
  ; output_name = None
  }

let assignSub
    ?(name = "AssignSub")
    (ref : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (value : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "AssignSub"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P value ]
  ; attributes
  ; output_name = None
  }

let avgPool
    ?(name = "AvgPool")
    ~padding
    ?data_format
    (value : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "AvgPool"
  ; output_type = value.output_type
  ; inputs = [ P value ]
  ; attributes
  ; output_name = None
  }

let avgPoolGrad
    ?(name = "AvgPoolGrad")
    ~padding
    ?data_format
    (orig_input_shape : [ `int32 ] t)
    (grad : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P grad.output_type) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "AvgPoolGrad"
  ; output_type = grad.output_type
  ; inputs = [ P orig_input_shape; P grad ]
  ; attributes
  ; output_name = None
  }

let batchCholesky
    ?(name = "BatchCholesky")
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BatchCholesky"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let batchMatMul
    ?(name = "BatchMatMul")
    (x : ([< `float | `double | `int32 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatMul"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let batchMatrixDeterminant
    ?(name = "BatchMatrixDeterminant")
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixDeterminant"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let batchMatrixInverse
    ?(name = "BatchMatrixInverse")
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixInverse"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let batchMatrixSolve
    ?(name = "BatchMatrixSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes
  ; output_name = None
  }

let batchMatrixSolveLs
    ?(name = "BatchMatrixSolveLs")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
    (l2_regularizer : [ `double ] t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixSolveLs"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs; P l2_regularizer ]
  ; attributes
  ; output_name = None
  }

let batchMatrixTriangularSolve
    ?(name = "BatchMatrixTriangularSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixTriangularSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes
  ; output_name = None
  }

let batchNormWithGlobalNormalization
    ?(name = "BatchNormWithGlobalNormalization")
    (t : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (m : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (v : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (beta : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (gamma : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P t.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BatchNormWithGlobalNormalization"
  ; output_type = t.output_type
  ; inputs = [ P t; P m; P v; P beta; P gamma ]
  ; attributes
  ; output_name = None
  }

let batchSelfAdjointEig
    ?(name = "BatchSelfAdjointEig")
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BatchSelfAdjointEig"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
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
  ; op_name = "BiasAdd"
  ; output_type = value.output_type
  ; inputs = [ P value; P bias ]
  ; attributes
  ; output_name = None
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
  ; op_name = "BiasAddGrad"
  ; output_type = out_backprop.output_type
  ; inputs = [ P out_backprop ]
  ; attributes
  ; output_name = None
  }

let biasAddV1
    ?(name = "BiasAddV1")
    (value : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (bias : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "BiasAddV1"
  ; output_type = value.output_type
  ; inputs = [ P value; P bias ]
  ; attributes
  ; output_name = None
  }

let bitcast
    ?(name = "Bitcast")
    ~type_
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "type", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Bitcast"
  ; output_type = type_
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let cast
    ?(name = "Cast")
    ~type_
    (x : 'srcT t)
  =
  let attributes = [ "DstT", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Cast"
  ; output_type = type_
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let ceil
    ?(name = "Ceil")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Ceil"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
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
  ; op_name = "CheckNumerics"
  ; output_type = tensor.output_type
  ; inputs = [ P tensor ]
  ; attributes
  ; output_name = None
  }

let cholesky
    ?(name = "Cholesky")
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Cholesky"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let complex
    ?(name = "Complex")
    (real : [ `float ] t)
    (imag : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Complex"
  ; output_type = Type.Complex64
  ; inputs = [ P real; P imag ]
  ; attributes
  ; output_name = None
  }

let complexAbs
    ?(name = "ComplexAbs")
    (x : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ComplexAbs"
  ; output_type = Type.Float
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let concat
    ?(name = "Concat")
    (concat_dim : [ `int32 ] t)
    (values : 't t)
  =
  let attributes = [ "T", Type (P values.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Concat"
  ; output_type = values.output_type
  ; inputs = [ P concat_dim; P values ]
  ; attributes
  ; output_name = None
  }

let concatOffset
    ?(name = "ConcatOffset")
    (concat_dim : [ `int32 ] t)
    (shape : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ConcatOffset"
  ; output_type = Type.Int32
  ; inputs = [ P concat_dim; P shape ]
  ; attributes
  ; output_name = None
  }

let conj
    ?(name = "Conj")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Conj"
  ; output_type = Type.Complex64
  ; inputs = [ P in__ ]
  ; attributes
  ; output_name = None
  }

let controlTrigger
    ?(name = "ControlTrigger")
    ()
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ControlTrigger"
  ; output_type = Type.Unit
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let conv2D
    ?(name = "Conv2D")
    ~padding
    ?data_format
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "Conv2D"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter ]
  ; attributes
  ; output_name = None
  }

let conv2DBackpropFilter
    ?(name = "Conv2DBackpropFilter")
    ~padding
    ?data_format
    (input : ([< `float | `double ] as 't) t)
    (filter_sizes : [ `int32 ] t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "Conv2DBackpropFilter"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter_sizes; P out_backprop ]
  ; attributes
  ; output_name = None
  }

let conv2DBackpropInput
    ?(name = "Conv2DBackpropInput")
    ~padding
    ?data_format
    (input_sizes : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P filter.output_type) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "Conv2DBackpropInput"
  ; output_type = filter.output_type
  ; inputs = [ P input_sizes; P filter; P out_backprop ]
  ; attributes
  ; output_name = None
  }

let cos
    ?(name = "Cos")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Cos"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let countUpTo
    ?(name = "CountUpTo")
    (ref : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "CountUpTo"
  ; output_type = ref.output_type
  ; inputs = [ P ref ]
  ; attributes
  ; output_name = None
  }

let cross
    ?(name = "Cross")
    (a : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (b : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P a.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Cross"
  ; output_type = a.output_type
  ; inputs = [ P a; P b ]
  ; attributes
  ; output_name = None
  }

let decodeJSONExample
    ?(name = "DecodeJSONExample")
    (json_examples : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "DecodeJSONExample"
  ; output_type = Type.String
  ; inputs = [ P json_examples ]
  ; attributes
  ; output_name = None
  }

let decodePng
    ?(name = "DecodePng")
    ~type_
    (contents : [ `string ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "DecodePng"
  ; output_type = type_
  ; inputs = [ P contents ]
  ; attributes
  ; output_name = None
  }

let decodeRaw
    ?(name = "DecodeRaw")
    ~type_
    (bytes : [ `string ] t)
  =
  let attributes = [ "out_type", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "DecodeRaw"
  ; output_type = type_
  ; inputs = [ P bytes ]
  ; attributes
  ; output_name = None
  }

let depthToSpace
    ?(name = "DepthToSpace")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "DepthToSpace"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let depthwiseConv2dNative
    ?(name = "DepthwiseConv2dNative")
    ~padding
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "DepthwiseConv2dNative"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter ]
  ; attributes
  ; output_name = None
  }

let depthwiseConv2dNativeBackpropFilter
    ?(name = "DepthwiseConv2dNativeBackpropFilter")
    ~padding
    (input : ([< `float | `double ] as 't) t)
    (filter_sizes : [ `int32 ] t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "DepthwiseConv2dNativeBackpropFilter"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter_sizes; P out_backprop ]
  ; attributes
  ; output_name = None
  }

let depthwiseConv2dNativeBackpropInput
    ?(name = "DepthwiseConv2dNativeBackpropInput")
    ~padding
    (input_sizes : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P filter.output_type) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "DepthwiseConv2dNativeBackpropInput"
  ; output_type = filter.output_type
  ; inputs = [ P input_sizes; P filter; P out_backprop ]
  ; attributes
  ; output_name = None
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
  ; op_name = "DestroyTemporaryVariable"
  ; output_type = ref.output_type
  ; inputs = [ P ref ]
  ; attributes
  ; output_name = None
  }

let diag
    ?(name = "Diag")
    (diagonal : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P diagonal.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Diag"
  ; output_type = diagonal.output_type
  ; inputs = [ P diagonal ]
  ; attributes
  ; output_name = None
  }

let diagPart
    ?(name = "DiagPart")
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "DiagPart"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let digamma
    ?(name = "Digamma")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Digamma"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let div
    ?(name = "Div")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Div"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let drawBoundingBoxes
    ?(name = "DrawBoundingBoxes")
    (images : [ `float ] t)
    (boxes : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "DrawBoundingBoxes"
  ; output_type = Type.Float
  ; inputs = [ P images; P boxes ]
  ; attributes
  ; output_name = None
  }

let dynamicPartition
    ?(name = "DynamicPartition")
    (data : 't t)
    (partitions : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "DynamicPartition"
  ; output_type = data.output_type
  ; inputs = [ P data; P partitions ]
  ; attributes
  ; output_name = None
  }

let dynamicStitch
    ?(name = "DynamicStitch")
    (indices : [ `int32 ] t)
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "DynamicStitch"
  ; output_type = data.output_type
  ; inputs = [ P indices; P data ]
  ; attributes
  ; output_name = None
  }

let editDistance
    ?(name = "EditDistance")
    (hypothesis_indices : [ `int64 ] t)
    (hypothesis_values : 't t)
    (hypothesis_shape : [ `int64 ] t)
    (truth_indices : [ `int64 ] t)
    (truth_values : 't t)
    (truth_shape : [ `int64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "EditDistance"
  ; output_type = Type.Float
  ; inputs = [ P hypothesis_indices; P hypothesis_values; P hypothesis_shape; P truth_indices; P truth_values; P truth_shape ]
  ; attributes
  ; output_name = None
  }

let elu
    ?(name = "Elu")
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Elu"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_name = None
  }

let eluGrad
    ?(name = "EluGrad")
    (gradients : ([< `float | `double ] as 't) t)
    (outputs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "EluGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P outputs ]
  ; attributes
  ; output_name = None
  }

let encodePng
    ?(name = "EncodePng")
    (image : 't t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "EncodePng"
  ; output_type = Type.String
  ; inputs = [ P image ]
  ; attributes
  ; output_name = None
  }

let enter
    ?(name = "Enter")
    ~frame_name
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  let attributes =
    ("frame_name", String frame_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "Enter"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_name = None
  }

let equal
    ?(name = "Equal")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Equal"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let erf
    ?(name = "Erf")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Erf"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let erfc
    ?(name = "Erfc")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Erfc"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let exit
    ?(name = "Exit")
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Exit"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_name = None
  }

let exp
    ?(name = "Exp")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Exp"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let expandDims
    ?(name = "ExpandDims")
    (input : 't t)
    (dim : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ExpandDims"
  ; output_type = input.output_type
  ; inputs = [ P input; P dim ]
  ; attributes
  ; output_name = None
  }

let extractGlimpse
    ?(name = "ExtractGlimpse")
    (input : [ `float ] t)
    (size : [ `int32 ] t)
    (offsets : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ExtractGlimpse"
  ; output_type = Type.Float
  ; inputs = [ P input; P size; P offsets ]
  ; attributes
  ; output_name = None
  }

let fFT2D
    ?(name = "FFT2D")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "FFT2D"
  ; output_type = Type.Complex64
  ; inputs = [ P in__ ]
  ; attributes
  ; output_name = None
  }

let fIFOQueue
    ?(name = "FIFOQueue")
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
  ; op_name = "FIFOQueue"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let fact
    ?(name = "Fact")
    ()
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Fact"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let fill
    ?(name = "Fill")
    (dims : [ `int32 ] t)
    (value : 't t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Fill"
  ; output_type = value.output_type
  ; inputs = [ P dims; P value ]
  ; attributes
  ; output_name = None
  }

let fixedLengthRecordReader
    ?(name = "FixedLengthRecordReader")
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
  ; op_name = "FixedLengthRecordReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let floor
    ?(name = "Floor")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Floor"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let gather
    ?(name = "Gather")
    (params : 'tparams t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tparams", Type (P params.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Gather"
  ; output_type = params.output_type
  ; inputs = [ P params; P indices ]
  ; attributes
  ; output_name = None
  }

let greater
    ?(name = "Greater")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Greater"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let greaterEqual
    ?(name = "GreaterEqual")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "GreaterEqual"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let hSVToRGB
    ?(name = "HSVToRGB")
    (images : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "HSVToRGB"
  ; output_type = Type.Float
  ; inputs = [ P images ]
  ; attributes
  ; output_name = None
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
  ; op_name = "HashTable"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let histogramSummary
    ?(name = "HistogramSummary")
    (tag : [ `string ] t)
    (values : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "HistogramSummary"
  ; output_type = Type.String
  ; inputs = [ P tag; P values ]
  ; attributes
  ; output_name = None
  }

let iFFT2D
    ?(name = "IFFT2D")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "IFFT2D"
  ; output_type = Type.Complex64
  ; inputs = [ P in__ ]
  ; attributes
  ; output_name = None
  }

let identity
    ?(name = "Identity")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Identity"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
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
  ; op_name = "IdentityReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let imag
    ?(name = "Imag")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Imag"
  ; output_type = Type.Float
  ; inputs = [ P in__ ]
  ; attributes
  ; output_name = None
  }

let imageSummary
    ?(name = "ImageSummary")
    (tag : [ `string ] t)
    (tensor : ([< `float ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ImageSummary"
  ; output_type = Type.String
  ; inputs = [ P tag; P tensor ]
  ; attributes
  ; output_name = None
  }

let inTopK
    ?(name = "InTopK")
    (predictions : [ `float ] t)
    (targets : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "InTopK"
  ; output_type = Type.Bool
  ; inputs = [ P predictions; P targets ]
  ; attributes
  ; output_name = None
  }

let initializeTable
    ?(name = "InitializeTable")
    (table_handle : [ `string ] t)
    (keys : 'tkey t)
    (values : 'tval t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "InitializeTable"
  ; output_type = Type.Unit
  ; inputs = [ P table_handle; P keys; P values ]
  ; attributes
  ; output_name = None
  }

let inv
    ?(name = "Inv")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Inv"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let invertPermutation
    ?(name = "InvertPermutation")
    (x : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "InvertPermutation"
  ; output_type = Type.Int32
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let isFinite
    ?(name = "IsFinite")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "IsFinite"
  ; output_type = Type.Bool
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let isInf
    ?(name = "IsInf")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "IsInf"
  ; output_type = Type.Bool
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let isNan
    ?(name = "IsNan")
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "IsNan"
  ; output_type = Type.Bool
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let l2Loss
    ?(name = "L2Loss")
    (t : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P t.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "L2Loss"
  ; output_type = t.output_type
  ; inputs = [ P t ]
  ; attributes
  ; output_name = None
  }

let lRN
    ?(name = "LRN")
    (input : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "LRN"
  ; output_type = Type.Float
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let lRNGrad
    ?(name = "LRNGrad")
    (input_grads : [ `float ] t)
    (input_image : [ `float ] t)
    (output_image : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "LRNGrad"
  ; output_type = Type.Float
  ; inputs = [ P input_grads; P input_image; P output_image ]
  ; attributes
  ; output_name = None
  }

let less
    ?(name = "Less")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Less"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let lessEqual
    ?(name = "LessEqual")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "LessEqual"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let lgamma
    ?(name = "Lgamma")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Lgamma"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let linSpace
    ?(name = "LinSpace")
    (start : ([< `float | `double ] as 't) t)
    (stop : ([< `float | `double ] as 't) t)
    (num : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P start.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "LinSpace"
  ; output_type = start.output_type
  ; inputs = [ P start; P stop; P num ]
  ; attributes
  ; output_name = None
  }

let log
    ?(name = "Log")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Log"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let logicalAnd
    ?(name = "LogicalAnd")
    (x : [ `bool ] t)
    (y : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "LogicalAnd"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let logicalNot
    ?(name = "LogicalNot")
    (x : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "LogicalNot"
  ; output_type = Type.Bool
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let logicalOr
    ?(name = "LogicalOr")
    (x : [ `bool ] t)
    (y : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "LogicalOr"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let lookupTableFind
    ?(name = "LookupTableFind")
    (table_handle : [ `string ] t)
    (keys : 'tin t)
    (default_value : 'tout t)
  =
  let attributes = [ "Tout", Type (P default_value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "LookupTableFind"
  ; output_type = default_value.output_type
  ; inputs = [ P table_handle; P keys; P default_value ]
  ; attributes
  ; output_name = None
  }

let lookupTableSize
    ?(name = "LookupTableSize")
    (table_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "LookupTableSize"
  ; output_type = Type.Int64
  ; inputs = [ P table_handle ]
  ; attributes
  ; output_name = None
  }

let loopCond
    ?(name = "LoopCond")
    (input : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "LoopCond"
  ; output_type = Type.Bool
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let matMul
    ?(name = "MatMul")
    (a : ([< `float | `double | `int32 | `complex64 ] as 't) t)
    (b : ([< `float | `double | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P a.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "MatMul"
  ; output_type = a.output_type
  ; inputs = [ P a; P b ]
  ; attributes
  ; output_name = None
  }

let matchingFiles
    ?(name = "MatchingFiles")
    (pattern : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "MatchingFiles"
  ; output_type = Type.String
  ; inputs = [ P pattern ]
  ; attributes
  ; output_name = None
  }

let matrixDeterminant
    ?(name = "MatrixDeterminant")
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "MatrixDeterminant"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let matrixInverse
    ?(name = "MatrixInverse")
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "MatrixInverse"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let matrixSolve
    ?(name = "MatrixSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "MatrixSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes
  ; output_name = None
  }

let matrixSolveLs
    ?(name = "MatrixSolveLs")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
    (l2_regularizer : [ `double ] t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "MatrixSolveLs"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs; P l2_regularizer ]
  ; attributes
  ; output_name = None
  }

let matrixTriangularSolve
    ?(name = "MatrixTriangularSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P matrix.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "MatrixTriangularSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes
  ; output_name = None
  }

let max
    ?(name = "Max")
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Max"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_name = None
  }

let maxPool
    ?(name = "MaxPool")
    ~padding
    ?data_format
    (input : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "MaxPool"
  ; output_type = Type.Float
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let maxPoolGrad
    ?(name = "MaxPoolGrad")
    ~padding
    ?data_format
    (orig_input : [ `float ] t)
    (orig_output : [ `float ] t)
    (grad : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "MaxPoolGrad"
  ; output_type = Type.Float
  ; inputs = [ P orig_input; P orig_output; P grad ]
  ; attributes
  ; output_name = None
  }

let maxPoolGradWithArgmax
    ?(name = "MaxPoolGradWithArgmax")
    ~padding
    (input : [ `float ] t)
    (grad : [ `float ] t)
    (argmax : ([< `int32 | `int64 ] as 'targmax) t)
  =
  let attributes = [] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "MaxPoolGradWithArgmax"
  ; output_type = Type.Float
  ; inputs = [ P input; P grad; P argmax ]
  ; attributes
  ; output_name = None
  }

let maximum
    ?(name = "Maximum")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Maximum"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let mean
    ?(name = "Mean")
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Mean"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_name = None
  }

let mergeSummary
    ?(name = "MergeSummary")
    (inputs : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "MergeSummary"
  ; output_type = Type.String
  ; inputs = [ P inputs ]
  ; attributes
  ; output_name = None
  }

let min
    ?(name = "Min")
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Min"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_name = None
  }

let minimum
    ?(name = "Minimum")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Minimum"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
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
  ; op_name = "MirrorPad"
  ; output_type = input.output_type
  ; inputs = [ P input; P paddings ]
  ; attributes
  ; output_name = None
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
  ; op_name = "MirrorPadGrad"
  ; output_type = input.output_type
  ; inputs = [ P input; P paddings ]
  ; attributes
  ; output_name = None
  }

let mod_
    ?(name = "Mod")
    (x : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (y : ([< `int32 | `int64 | `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Mod"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let mul
    ?(name = "Mul")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Mul"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let neg
    ?(name = "Neg")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Neg"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let negTrain
    ?(name = "NegTrain")
    (w_in : [ `float ] t)
    (w_out : [ `float ] t)
    (examples : [ `int32 ] t)
    (labels : [ `int32 ] t)
    (lr : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "NegTrain"
  ; output_type = Type.Unit
  ; inputs = [ P w_in; P w_out; P examples; P labels; P lr ]
  ; attributes
  ; output_name = None
  }

let nextIteration
    ?(name = "NextIteration")
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "NextIteration"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_name = None
  }

let noOp
    ?(name = "NoOp")
    ()
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "NoOp"
  ; output_type = Type.Unit
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let notEqual
    ?(name = "NotEqual")
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "NotEqual"
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let oneHot
    ?(name = "OneHot")
    (indices : [ `int64 ] t)
    (depth : [ `int32 ] t)
    (on_value : 't t)
    (off_value : 't t)
  =
  let attributes = [ "T", Type (P on_value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "OneHot"
  ; output_type = on_value.output_type
  ; inputs = [ P indices; P depth; P on_value; P off_value ]
  ; attributes
  ; output_name = None
  }

let pack
    ?(name = "Pack")
    (values : 't t)
  =
  let attributes = [ "T", Type (P values.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Pack"
  ; output_type = values.output_type
  ; inputs = [ P values ]
  ; attributes
  ; output_name = None
  }

let pad
    ?(name = "Pad")
    (input : 't t)
    (paddings : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Pad"
  ; output_type = input.output_type
  ; inputs = [ P input; P paddings ]
  ; attributes
  ; output_name = None
  }

let paddingFIFOQueue
    ?(name = "PaddingFIFOQueue")
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
  ; op_name = "PaddingFIFOQueue"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
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
  ; op_name = "Placeholder"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let pow
    ?(name = "Pow")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Pow"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let prod
    ?(name = "Prod")
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Prod"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_name = None
  }

let queueClose
    ?(name = "QueueClose")
    (handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "QueueClose"
  ; output_type = Type.Unit
  ; inputs = [ P handle ]
  ; attributes
  ; output_name = None
  }

let queueSize
    ?(name = "QueueSize")
    (handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "QueueSize"
  ; output_type = Type.Int32
  ; inputs = [ P handle ]
  ; attributes
  ; output_name = None
  }

let rGBToHSV
    ?(name = "RGBToHSV")
    (images : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "RGBToHSV"
  ; output_type = Type.Float
  ; inputs = [ P images ]
  ; attributes
  ; output_name = None
  }

let randomCrop
    ?(name = "RandomCrop")
    (image : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P image.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RandomCrop"
  ; output_type = image.output_type
  ; inputs = [ P image; P size ]
  ; attributes
  ; output_name = None
  }

let randomShuffle
    ?(name = "RandomShuffle")
    (value : 't t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RandomShuffle"
  ; output_type = value.output_type
  ; inputs = [ P value ]
  ; attributes
  ; output_name = None
  }

let randomShuffleQueue
    ?(name = "RandomShuffleQueue")
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
  ; op_name = "RandomShuffleQueue"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let randomStandardNormal
    ?(name = "RandomStandardNormal")
    ~type_
    (shape : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RandomStandardNormal"
  ; output_type = type_
  ; inputs = [ P shape ]
  ; attributes
  ; output_name = None
  }

let randomUniform
    ?(name = "RandomUniform")
    ~type_
    (shape : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RandomUniform"
  ; output_type = type_
  ; inputs = [ P shape ]
  ; attributes
  ; output_name = None
  }

let randomUniformInt
    ?(name = "RandomUniformInt")
    (shape : ([< `int32 | `int64 ] as 't) t)
    (minval : ([< `int32 | `int64 ] as 'tout) t)
    (maxval : ([< `int32 | `int64 ] as 'tout) t)
  =
  let attributes = [ "Tout", Type (P minval.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RandomUniformInt"
  ; output_type = minval.output_type
  ; inputs = [ P shape; P minval; P maxval ]
  ; attributes
  ; output_name = None
  }

let range
    ?(name = "Range")
    (start : [ `int32 ] t)
    (limit : [ `int32 ] t)
    (delta : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Range"
  ; output_type = Type.Int32
  ; inputs = [ P start; P limit; P delta ]
  ; attributes
  ; output_name = None
  }

let rank
    ?(name = "Rank")
    (input : 't t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Rank"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let readFile
    ?(name = "ReadFile")
    (filename : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ReadFile"
  ; output_type = Type.String
  ; inputs = [ P filename ]
  ; attributes
  ; output_name = None
  }

let readerNumRecordsProduced
    ?(name = "ReaderNumRecordsProduced")
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ReaderNumRecordsProduced"
  ; output_type = Type.Int64
  ; inputs = [ P reader_handle ]
  ; attributes
  ; output_name = None
  }

let readerNumWorkUnitsCompleted
    ?(name = "ReaderNumWorkUnitsCompleted")
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ReaderNumWorkUnitsCompleted"
  ; output_type = Type.Int64
  ; inputs = [ P reader_handle ]
  ; attributes
  ; output_name = None
  }

let readerReset
    ?(name = "ReaderReset")
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ReaderReset"
  ; output_type = Type.Unit
  ; inputs = [ P reader_handle ]
  ; attributes
  ; output_name = None
  }

let readerRestoreState
    ?(name = "ReaderRestoreState")
    (reader_handle : [ `string ] t)
    (state : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ReaderRestoreState"
  ; output_type = Type.Unit
  ; inputs = [ P reader_handle; P state ]
  ; attributes
  ; output_name = None
  }

let readerSerializeState
    ?(name = "ReaderSerializeState")
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ReaderSerializeState"
  ; output_type = Type.String
  ; inputs = [ P reader_handle ]
  ; attributes
  ; output_name = None
  }

let real
    ?(name = "Real")
    (in__ : [ `complex64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Real"
  ; output_type = Type.Float
  ; inputs = [ P in__ ]
  ; attributes
  ; output_name = None
  }

let refEnter
    ?(name = "RefEnter")
    ~frame_name
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  let attributes =
    ("frame_name", String frame_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "RefEnter"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_name = None
  }

let refExit
    ?(name = "RefExit")
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RefExit"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_name = None
  }

let refIdentity
    ?(name = "RefIdentity")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RefIdentity"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let refNextIteration
    ?(name = "RefNextIteration")
    (data : 't t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RefNextIteration"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes
  ; output_name = None
  }

let refSelect
    ?(name = "RefSelect")
    (index : [ `int32 ] t)
    (inputs : 't t)
  =
  let attributes = [ "T", Type (P inputs.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RefSelect"
  ; output_type = inputs.output_type
  ; inputs = [ P index; P inputs ]
  ; attributes
  ; output_name = None
  }

let relu
    ?(name = "Relu")
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Relu"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_name = None
  }

let relu6
    ?(name = "Relu6")
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Relu6"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_name = None
  }

let relu6Grad
    ?(name = "Relu6Grad")
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Relu6Grad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes
  ; output_name = None
  }

let reluGrad
    ?(name = "ReluGrad")
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ReluGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes
  ; output_name = None
  }

let reshape
    ?(name = "Reshape")
    (tensor : 't t)
    (shape : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P tensor.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Reshape"
  ; output_type = tensor.output_type
  ; inputs = [ P tensor; P shape ]
  ; attributes
  ; output_name = None
  }

let resizeArea
    ?(name = "ResizeArea")
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ResizeArea"
  ; output_type = Type.Float
  ; inputs = [ P images; P size ]
  ; attributes
  ; output_name = None
  }

let resizeBicubic
    ?(name = "ResizeBicubic")
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ResizeBicubic"
  ; output_type = Type.Float
  ; inputs = [ P images; P size ]
  ; attributes
  ; output_name = None
  }

let resizeBilinear
    ?(name = "ResizeBilinear")
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ResizeBilinear"
  ; output_type = Type.Float
  ; inputs = [ P images; P size ]
  ; attributes
  ; output_name = None
  }

let resizeBilinearGrad
    ?(name = "ResizeBilinearGrad")
    (grads : [ `float ] t)
    (original_image : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P original_image.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ResizeBilinearGrad"
  ; output_type = original_image.output_type
  ; inputs = [ P grads; P original_image ]
  ; attributes
  ; output_name = None
  }

let resizeNearestNeighbor
    ?(name = "ResizeNearestNeighbor")
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P images.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ResizeNearestNeighbor"
  ; output_type = images.output_type
  ; inputs = [ P images; P size ]
  ; attributes
  ; output_name = None
  }

let resizeNearestNeighborGrad
    ?(name = "ResizeNearestNeighborGrad")
    (grads : ([< `int32 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P grads.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ResizeNearestNeighborGrad"
  ; output_type = grads.output_type
  ; inputs = [ P grads; P size ]
  ; attributes
  ; output_name = None
  }

let restore
    ?(name = "Restore")
    ~type_
    (file_pattern : [ `string ] t)
    (tensor_name : [ `string ] t)
  =
  let attributes = [ "dt", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Restore"
  ; output_type = type_
  ; inputs = [ P file_pattern; P tensor_name ]
  ; attributes
  ; output_name = None
  }

let restoreSlice
    ?(name = "RestoreSlice")
    ~type_
    (file_pattern : [ `string ] t)
    (tensor_name : [ `string ] t)
    (shape_and_slice : [ `string ] t)
  =
  let attributes = [ "dt", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "RestoreSlice"
  ; output_type = type_
  ; inputs = [ P file_pattern; P tensor_name; P shape_and_slice ]
  ; attributes
  ; output_name = None
  }

let reverse
    ?(name = "Reverse")
    (tensor : ([< `int32 | `bool | `float | `double ] as 't) t)
    (dims : [ `bool ] t)
  =
  let attributes = [ "T", Type (P tensor.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Reverse"
  ; output_type = tensor.output_type
  ; inputs = [ P tensor; P dims ]
  ; attributes
  ; output_name = None
  }

let reverseSequence
    ?(name = "ReverseSequence")
    (input : 't t)
    (seq_lengths : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ReverseSequence"
  ; output_type = input.output_type
  ; inputs = [ P input; P seq_lengths ]
  ; attributes
  ; output_name = None
  }

let rsqrt
    ?(name = "Rsqrt")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Rsqrt"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let scalarSummary
    ?(name = "ScalarSummary")
    (tags : [ `string ] t)
    (values : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ScalarSummary"
  ; output_type = Type.String
  ; inputs = [ P tags; P values ]
  ; attributes
  ; output_name = None
  }

let scatterAdd
    ?(name = "ScatterAdd")
    (ref : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ScatterAdd"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P indices; P updates ]
  ; attributes
  ; output_name = None
  }

let scatterSub
    ?(name = "ScatterSub")
    (ref : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ScatterSub"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P indices; P updates ]
  ; attributes
  ; output_name = None
  }

let scatterUpdate
    ?(name = "ScatterUpdate")
    (ref : 't t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : 't t)
  =
  let attributes = [ "T", Type (P ref.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ScatterUpdate"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P indices; P updates ]
  ; attributes
  ; output_name = None
  }

let segmentMax
    ?(name = "SegmentMax")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SegmentMax"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_name = None
  }

let segmentMean
    ?(name = "SegmentMean")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SegmentMean"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_name = None
  }

let segmentMin
    ?(name = "SegmentMin")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SegmentMin"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_name = None
  }

let segmentProd
    ?(name = "SegmentProd")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SegmentProd"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_name = None
  }

let segmentSum
    ?(name = "SegmentSum")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SegmentSum"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes
  ; output_name = None
  }

let select
    ?(name = "Select")
    (condition : [ `bool ] t)
    (t : 't t)
    (e : 't t)
  =
  let attributes = [ "T", Type (P t.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Select"
  ; output_type = t.output_type
  ; inputs = [ P condition; P t; P e ]
  ; attributes
  ; output_name = None
  }

let selfAdjointEig
    ?(name = "SelfAdjointEig")
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SelfAdjointEig"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let serializeManySparse
    ?(name = "SerializeManySparse")
    (sparse_indices : [ `int64 ] t)
    (sparse_values : 't t)
    (sparse_shape : [ `int64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "SerializeManySparse"
  ; output_type = Type.String
  ; inputs = [ P sparse_indices; P sparse_values; P sparse_shape ]
  ; attributes
  ; output_name = None
  }

let serializeSparse
    ?(name = "SerializeSparse")
    (sparse_indices : [ `int64 ] t)
    (sparse_values : 't t)
    (sparse_shape : [ `int64 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "SerializeSparse"
  ; output_type = Type.String
  ; inputs = [ P sparse_indices; P sparse_values; P sparse_shape ]
  ; attributes
  ; output_name = None
  }

let shape
    ?(name = "Shape")
    (input : 't t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Shape"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let shapeN
    ?(name = "ShapeN")
    (input : 't t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ShapeN"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let shardedFilename
    ?(name = "ShardedFilename")
    (basename : [ `string ] t)
    (shard : [ `int32 ] t)
    (num_shards : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ShardedFilename"
  ; output_type = Type.String
  ; inputs = [ P basename; P shard; P num_shards ]
  ; attributes
  ; output_name = None
  }

let shardedFilespec
    ?(name = "ShardedFilespec")
    (basename : [ `string ] t)
    (num_shards : [ `int32 ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "ShardedFilespec"
  ; output_type = Type.String
  ; inputs = [ P basename; P num_shards ]
  ; attributes
  ; output_name = None
  }

let sigmoid
    ?(name = "Sigmoid")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Sigmoid"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let sign
    ?(name = "Sign")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Sign"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let sin
    ?(name = "Sin")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Sin"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let size
    ?(name = "Size")
    (input : 't t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Size"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let slice
    ?(name = "Slice")
    (input : 't t)
    (begin__ : ([< `int32 | `int64 ] as 'index) t)
    (size : ([< `int32 | `int64 ] as 'index) t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Slice"
  ; output_type = input.output_type
  ; inputs = [ P input; P begin__; P size ]
  ; attributes
  ; output_name = None
  }

let softmax
    ?(name = "Softmax")
    (logits : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P logits.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Softmax"
  ; output_type = logits.output_type
  ; inputs = [ P logits ]
  ; attributes
  ; output_name = None
  }

let softplus
    ?(name = "Softplus")
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Softplus"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_name = None
  }

let softplusGrad
    ?(name = "SoftplusGrad")
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SoftplusGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes
  ; output_name = None
  }

let softsign
    ?(name = "Softsign")
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P features.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Softsign"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes
  ; output_name = None
  }

let softsignGrad
    ?(name = "SoftsignGrad")
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P gradients.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SoftsignGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes
  ; output_name = None
  }

let spaceToDepth
    ?(name = "SpaceToDepth")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SpaceToDepth"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let sparseApplyAdagrad
    ?(name = "SparseApplyAdagrad")
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SparseApplyAdagrad"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P indices ]
  ; attributes
  ; output_name = None
  }

let sparseApplyFtrl
    ?(name = "SparseApplyFtrl")
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
  let attributes = [ "T", Type (P var.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SparseApplyFtrl"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P linear; P grad; P indices; P lr; P l1; P l2; P lr_power ]
  ; attributes
  ; output_name = None
  }

let sparseApplyMomentum
    ?(name = "SparseApplyMomentum")
    (var : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (accum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (lr : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (grad : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (momentum : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P var.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SparseApplyMomentum"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P indices; P momentum ]
  ; attributes
  ; output_name = None
  }

let sparseMatMul
    ?(name = "SparseMatMul")
    (a : [ `float ] t)
    (b : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "SparseMatMul"
  ; output_type = Type.Float
  ; inputs = [ P a; P b ]
  ; attributes
  ; output_name = None
  }

let sparseSegmentMean
    ?(name = "SparseSegmentMean")
    (data : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SparseSegmentMean"
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
  ; attributes
  ; output_name = None
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
  ; op_name = "SparseSegmentMeanGrad"
  ; output_type = grad.output_type
  ; inputs = [ P grad; P indices; P segment_ids; P output_dim0 ]
  ; attributes
  ; output_name = None
  }

let sparseSegmentSqrtN
    ?(name = "SparseSegmentSqrtN")
    (data : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SparseSegmentSqrtN"
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
  ; attributes
  ; output_name = None
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
  ; op_name = "SparseSegmentSqrtNGrad"
  ; output_type = grad.output_type
  ; inputs = [ P grad; P indices; P segment_ids; P output_dim0 ]
  ; attributes
  ; output_name = None
  }

let sparseSegmentSum
    ?(name = "SparseSegmentSum")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SparseSegmentSum"
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
  ; attributes
  ; output_name = None
  }

let sparseTensorDenseMatMul
    ?(name = "SparseTensorDenseMatMul")
    (a_indices : [ `int64 ] t)
    (a_values : 't t)
    (a_shape : [ `int64 ] t)
    (b : 't t)
  =
  let attributes = [ "T", Type (P a_values.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SparseTensorDenseMatMul"
  ; output_type = a_values.output_type
  ; inputs = [ P a_indices; P a_values; P a_shape; P b ]
  ; attributes
  ; output_name = None
  }

let sparseToDense
    ?(name = "SparseToDense")
    (sparse_indices : ([< `int32 | `int64 ] as 'tindices) t)
    (output_shape : ([< `int32 | `int64 ] as 'tindices) t)
    (sparse_values : 't t)
    (default_value : 't t)
  =
  let attributes = [ "T", Type (P sparse_values.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SparseToDense"
  ; output_type = sparse_values.output_type
  ; inputs = [ P sparse_indices; P output_shape; P sparse_values; P default_value ]
  ; attributes
  ; output_name = None
  }

let split
    ?(name = "Split")
    (split_dim : [ `int32 ] t)
    (value : 't t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Split"
  ; output_type = value.output_type
  ; inputs = [ P split_dim; P value ]
  ; attributes
  ; output_name = None
  }

let sqrt
    ?(name = "Sqrt")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Sqrt"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let square
    ?(name = "Square")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Square"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

let squaredDifference
    ?(name = "SquaredDifference")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "SquaredDifference"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let squeeze
    ?(name = "Squeeze")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Squeeze"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
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
  ; op_name = "Stack"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let stackClose
    ?(name = "StackClose")
    (handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "StackClose"
  ; output_type = Type.Unit
  ; inputs = [ P handle ]
  ; attributes
  ; output_name = None
  }

let stackPop
    ?(name = "StackPop")
    ~type_
    (handle : [ `string ] t)
  =
  let attributes = [ "elem_type", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "StackPop"
  ; output_type = type_
  ; inputs = [ P handle ]
  ; attributes
  ; output_name = None
  }

let stackPush
    ?(name = "StackPush")
    (handle : [ `string ] t)
    (elem : 't t)
  =
  let attributes = [ "T", Type (P elem.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "StackPush"
  ; output_type = elem.output_type
  ; inputs = [ P handle; P elem ]
  ; attributes
  ; output_name = None
  }

let stopGradient
    ?(name = "StopGradient")
    (input : 't t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "StopGradient"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
  }

let stringToHashBucket
    ?(name = "StringToHashBucket")
    (string_tensor : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "StringToHashBucket"
  ; output_type = Type.Int64
  ; inputs = [ P string_tensor ]
  ; attributes
  ; output_name = None
  }

let stringToNumber
    ?(name = "StringToNumber")
    ~type_
    (string_tensor : [ `string ] t)
  =
  let attributes = [ "out_type", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "StringToNumber"
  ; output_type = type_
  ; inputs = [ P string_tensor ]
  ; attributes
  ; output_name = None
  }

let sub
    ?(name = "Sub")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Sub"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes
  ; output_name = None
  }

let sum
    ?(name = "Sum")
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Sum"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes
  ; output_name = None
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
  ; op_name = "TFRecordReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let tanh
    ?(name = "Tanh")
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Tanh"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
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
  ; op_name = "TemporaryVariable"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let tensorArray
    ?(name = "TensorArray")
    ?tensor_array_name
    (size : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match tensor_array_name with | None -> attributes | Some tensor_array_name -> ("tensor_array_name", String tensor_array_name) :: attributes
  in
  { name = Name.make_fresh ~name
  ; op_name = "TensorArray"
  ; output_type = Type.String
  ; inputs = [ P size ]
  ; attributes
  ; output_name = None
  }

let tensorArrayClose
    ?(name = "TensorArrayClose")
    (handle : [ `string ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "TensorArrayClose"
  ; output_type = Type.Unit
  ; inputs = [ P handle ]
  ; attributes
  ; output_name = None
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
  ; op_name = "TensorArrayGrad"
  ; output_type = Type.String
  ; inputs = [ P handle; P flow_in ]
  ; attributes
  ; output_name = None
  }

let tensorArrayPack
    ?(name = "TensorArrayPack")
    ~type_
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "TensorArrayPack"
  ; output_type = type_
  ; inputs = [ P handle; P flow_in ]
  ; attributes
  ; output_name = None
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
  ; op_name = "TensorArrayRead"
  ; output_type = type_
  ; inputs = [ P handle; P index; P flow_in ]
  ; attributes
  ; output_name = None
  }

let tensorArraySize
    ?(name = "TensorArraySize")
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "TensorArraySize"
  ; output_type = Type.Int32
  ; inputs = [ P handle; P flow_in ]
  ; attributes
  ; output_name = None
  }

let tensorArraySplit
    ?(name = "TensorArraySplit")
    (handle : [ `string ] t)
    (value : 't t)
    (lengths : [ `int64 ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "TensorArraySplit"
  ; output_type = Type.Float
  ; inputs = [ P handle; P value; P lengths; P flow_in ]
  ; attributes
  ; output_name = None
  }

let tensorArrayUnpack
    ?(name = "TensorArrayUnpack")
    (handle : [ `string ] t)
    (value : 't t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "TensorArrayUnpack"
  ; output_type = Type.Float
  ; inputs = [ P handle; P value; P flow_in ]
  ; attributes
  ; output_name = None
  }

let tensorArrayWrite
    ?(name = "TensorArrayWrite")
    (handle : [ `string ] t)
    (index : [ `int32 ] t)
    (value : 't t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "TensorArrayWrite"
  ; output_type = Type.Float
  ; inputs = [ P handle; P index; P value; P flow_in ]
  ; attributes
  ; output_name = None
  }

let textLineReader
    ?(name = "TextLineReader")
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
  ; op_name = "TextLineReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let tile
    ?(name = "Tile")
    (input : 't t)
    (multiples : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Tile"
  ; output_type = input.output_type
  ; inputs = [ P input; P multiples ]
  ; attributes
  ; output_name = None
  }

let tileGrad
    ?(name = "TileGrad")
    (input : 't t)
    (multiples : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P input.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "TileGrad"
  ; output_type = input.output_type
  ; inputs = [ P input; P multiples ]
  ; attributes
  ; output_name = None
  }

let transpose
    ?(name = "Transpose")
    (x : 't t)
    (perm : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Transpose"
  ; output_type = x.output_type
  ; inputs = [ P x; P perm ]
  ; attributes
  ; output_name = None
  }

let truncatedNormal
    ?(name = "TruncatedNormal")
    ~type_
    (shape : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  { name = Name.make_fresh ~name
  ; op_name = "TruncatedNormal"
  ; output_type = type_
  ; inputs = [ P shape ]
  ; attributes
  ; output_name = None
  }

let unpack
    ?(name = "Unpack")
    (value : 't t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "Unpack"
  ; output_type = value.output_type
  ; inputs = [ P value ]
  ; attributes
  ; output_name = None
  }

let unsortedSegmentSum
    ?(name = "UnsortedSegmentSum")
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
    (num_segments : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P data.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "UnsortedSegmentSum"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids; P num_segments ]
  ; attributes
  ; output_name = None
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
  ; op_name = "Variable"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let where
    ?(name = "Where")
    (input : [ `bool ] t)
  =
  let attributes = [] in
  { name = Name.make_fresh ~name
  ; op_name = "Where"
  ; output_type = Type.Int64
  ; inputs = [ P input ]
  ; attributes
  ; output_name = None
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
  ; op_name = "WholeFileReader"
  ; output_type = Type.String
  ; inputs = [  ]
  ; attributes
  ; output_name = None
  }

let zerosLike
    ?(name = "ZerosLike")
    (x : 't t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = "ZerosLike"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_name = None
  }

