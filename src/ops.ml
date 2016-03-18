open Node

let abs
    ?(name = "Abs")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Abs"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let add
    ?(name = "Add")
    (x : ([< `float | `double | `int32 ] as 't) t)
    (y : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Add"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let addN
    ?(name = "AddN")
    (inputs : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "AddN"
  ; output_type = inputs.output_type
  ; inputs = [ P inputs ]
  ; attributes = [
      "T", Type (P inputs.output_type);
    ]
  }

let adjustContrast
    ?(name = "AdjustContrast")
    (images : ([< `int32 | `float | `double ] as 't) t)
    (contrast_factor : [ `float ] t)
    (min_value : [ `float ] t)
    (max_value : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "AdjustContrast"
  ; output_type = Type.Float
  ; inputs = [ P images; P contrast_factor; P min_value; P max_value ]
  ; attributes = [
    ]
  }

let adjustContrastv2
    ?(name = "AdjustContrastv2")
    (images : [ `float ] t)
    (contrast_factor : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "AdjustContrastv2"
  ; output_type = Type.Float
  ; inputs = [ P images; P contrast_factor ]
  ; attributes = [
    ]
  }

let applyAdagrad
    ?(name = "ApplyAdagrad")
    (var : ([< `float | `double | `int32 ] as 't) t)
    (accum : ([< `float | `double | `int32 ] as 't) t)
    (lr : ([< `float | `double | `int32 ] as 't) t)
    (grad : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ApplyAdagrad"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad ]
  ; attributes = [
      "T", Type (P var.output_type);
    ]
  }

let applyAdam
    ?(name = "ApplyAdam")
    (var : ([< `float | `double | `int32 ] as 't) t)
    (m : ([< `float | `double | `int32 ] as 't) t)
    (v : ([< `float | `double | `int32 ] as 't) t)
    (beta1_power : ([< `float | `double | `int32 ] as 't) t)
    (beta2_power : ([< `float | `double | `int32 ] as 't) t)
    (lr : ([< `float | `double | `int32 ] as 't) t)
    (beta1 : ([< `float | `double | `int32 ] as 't) t)
    (beta2 : ([< `float | `double | `int32 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 ] as 't) t)
    (grad : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ApplyAdam"
  ; output_type = var.output_type
  ; inputs = [ P var; P m; P v; P beta1_power; P beta2_power; P lr; P beta1; P beta2; P epsilon; P grad ]
  ; attributes = [
      "T", Type (P var.output_type);
    ]
  }

let applyFtrl
    ?(name = "ApplyFtrl")
    (var : ([< `float | `double | `int32 ] as 't) t)
    (accum : ([< `float | `double | `int32 ] as 't) t)
    (linear : ([< `float | `double | `int32 ] as 't) t)
    (grad : ([< `float | `double | `int32 ] as 't) t)
    (lr : ([< `float | `double | `int32 ] as 't) t)
    (l1 : ([< `float | `double | `int32 ] as 't) t)
    (l2 : ([< `float | `double | `int32 ] as 't) t)
    (lr_power : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ApplyFtrl"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P linear; P grad; P lr; P l1; P l2; P lr_power ]
  ; attributes = [
      "T", Type (P var.output_type);
    ]
  }

let applyGradientDescent
    ?(name = "ApplyGradientDescent")
    (var : ([< `float | `double | `int32 ] as 't) t)
    (alpha : ([< `float | `double | `int32 ] as 't) t)
    (delta : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ApplyGradientDescent"
  ; output_type = var.output_type
  ; inputs = [ P var; P alpha; P delta ]
  ; attributes = [
      "T", Type (P var.output_type);
    ]
  }

let applyMomentum
    ?(name = "ApplyMomentum")
    (var : ([< `float | `double | `int32 ] as 't) t)
    (accum : ([< `float | `double | `int32 ] as 't) t)
    (lr : ([< `float | `double | `int32 ] as 't) t)
    (grad : ([< `float | `double | `int32 ] as 't) t)
    (momentum : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ApplyMomentum"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P momentum ]
  ; attributes = [
      "T", Type (P var.output_type);
    ]
  }

let applyRMSProp
    ?(name = "ApplyRMSProp")
    (var : ([< `float | `double | `int32 ] as 't) t)
    (ms : ([< `float | `double | `int32 ] as 't) t)
    (mom : ([< `float | `double | `int32 ] as 't) t)
    (lr : ([< `float | `double | `int32 ] as 't) t)
    (rho : ([< `float | `double | `int32 ] as 't) t)
    (momentum : ([< `float | `double | `int32 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 ] as 't) t)
    (grad : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ApplyRMSProp"
  ; output_type = var.output_type
  ; inputs = [ P var; P ms; P mom; P lr; P rho; P momentum; P epsilon; P grad ]
  ; attributes = [
      "T", Type (P var.output_type);
    ]
  }

let assign
    ?(name = "Assign")
    (ref : 't t)
    (value : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Assign"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P value ]
  ; attributes = [
      "T", Type (P ref.output_type);
    ]
  }

let assignAdd
    ?(name = "AssignAdd")
    (ref : ([< `float | `double | `int32 ] as 't) t)
    (value : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "AssignAdd"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P value ]
  ; attributes = [
      "T", Type (P ref.output_type);
    ]
  }

let assignSub
    ?(name = "AssignSub")
    (ref : ([< `float | `double | `int32 ] as 't) t)
    (value : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "AssignSub"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P value ]
  ; attributes = [
      "T", Type (P ref.output_type);
    ]
  }

let avgPool
    ?(name = "AvgPool")
    (value : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "AvgPool"
  ; output_type = value.output_type
  ; inputs = [ P value ]
  ; attributes = [
      "T", Type (P value.output_type);
    ]
  }

let avgPoolGrad
    ?(name = "AvgPoolGrad")
    (orig_input_shape : [ `int32 ] t)
    (grad : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "AvgPoolGrad"
  ; output_type = grad.output_type
  ; inputs = [ P orig_input_shape; P grad ]
  ; attributes = [
      "T", Type (P grad.output_type);
    ]
  }

let batchCholesky
    ?(name = "BatchCholesky")
    (input : ([< `double | `float ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BatchCholesky"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let batchMatMul
    ?(name = "BatchMatMul")
    (x : ([< `float | `double | `int32 ] as 't) t)
    (y : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatMul"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let batchMatrixDeterminant
    ?(name = "BatchMatrixDeterminant")
    (input : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixDeterminant"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let batchMatrixInverse
    ?(name = "BatchMatrixInverse")
    (input : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixInverse"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let batchMatrixSolve
    ?(name = "BatchMatrixSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes = [
      "T", Type (P matrix.output_type);
    ]
  }

let batchMatrixSolveLs
    ?(name = "BatchMatrixSolveLs")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
    (l2_regularizer : [ `double ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixSolveLs"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs; P l2_regularizer ]
  ; attributes = [
      "T", Type (P matrix.output_type);
    ]
  }

let batchMatrixTriangularSolve
    ?(name = "BatchMatrixTriangularSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BatchMatrixTriangularSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes = [
      "T", Type (P matrix.output_type);
    ]
  }

let batchNormWithGlobalNormalization
    ?(name = "BatchNormWithGlobalNormalization")
    (t : ([< `float | `double | `int32 ] as 't) t)
    (m : ([< `float | `double | `int32 ] as 't) t)
    (v : ([< `float | `double | `int32 ] as 't) t)
    (beta : ([< `float | `double | `int32 ] as 't) t)
    (gamma : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BatchNormWithGlobalNormalization"
  ; output_type = t.output_type
  ; inputs = [ P t; P m; P v; P beta; P gamma ]
  ; attributes = [
      "T", Type (P t.output_type);
    ]
  }

let batchSelfAdjointEig
    ?(name = "BatchSelfAdjointEig")
    (input : ([< `double | `float ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BatchSelfAdjointEig"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let biasAdd
    ?(name = "BiasAdd")
    (value : ([< `float | `double | `int32 ] as 't) t)
    (bias : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BiasAdd"
  ; output_type = value.output_type
  ; inputs = [ P value; P bias ]
  ; attributes = [
      "T", Type (P value.output_type);
    ]
  }

let biasAddGrad
    ?(name = "BiasAddGrad")
    (out_backprop : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BiasAddGrad"
  ; output_type = out_backprop.output_type
  ; inputs = [ P out_backprop ]
  ; attributes = [
      "T", Type (P out_backprop.output_type);
    ]
  }

let biasAddV1
    ?(name = "BiasAddV1")
    (value : ([< `float | `double | `int32 ] as 't) t)
    (bias : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "BiasAddV1"
  ; output_type = value.output_type
  ; inputs = [ P value; P bias ]
  ; attributes = [
      "T", Type (P value.output_type);
    ]
  }

let bitcast
    ?(name = "Bitcast")
    ~type_
    (input : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Bitcast"
  ; output_type = type_
  ; inputs = [ P input ]
  ; attributes = [
      "type", Type (P type_);
    ]
  }

let cast
    ?(name = "Cast")
    ~type_
    (x : 'srcT t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Cast"
  ; output_type = type_
  ; inputs = [ P x ]
  ; attributes = [
      "DstT", Type (P type_);
    ]
  }

let ceil
    ?(name = "Ceil")
    (x : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Ceil"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let checkNumerics
    ?(name = "CheckNumerics")
    (tensor : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "CheckNumerics"
  ; output_type = tensor.output_type
  ; inputs = [ P tensor ]
  ; attributes = [
      "T", Type (P tensor.output_type);
    ]
  }

let cholesky
    ?(name = "Cholesky")
    (input : ([< `double | `float ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Cholesky"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let concat
    ?(name = "Concat")
    (concat_dim : [ `int32 ] t)
    (values : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Concat"
  ; output_type = values.output_type
  ; inputs = [ P concat_dim; P values ]
  ; attributes = [
      "T", Type (P values.output_type);
    ]
  }

let concatOffset
    ?(name = "ConcatOffset")
    (concat_dim : [ `int32 ] t)
    (shape : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ConcatOffset"
  ; output_type = Type.Int32
  ; inputs = [ P concat_dim; P shape ]
  ; attributes = [
    ]
  }

let const
    ?(name = "Const")
    ~type_
  =
  { name = Name.make_fresh ~name
  ; op_name = "Const"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes = [
      "dtype", Type (P type_);
    ]
  }

let controlTrigger
    ?(name = "ControlTrigger")
    ()
  =
  { name = Name.make_fresh ~name
  ; op_name = "ControlTrigger"
  ; output_type = Type.Unit
  ; inputs = [  ]
  ; attributes = [
    ]
  }

let conv2D
    ?(name = "Conv2D")
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Conv2D"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let conv2DBackpropFilter
    ?(name = "Conv2DBackpropFilter")
    (input : ([< `float | `double ] as 't) t)
    (filter_sizes : [ `int32 ] t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Conv2DBackpropFilter"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter_sizes; P out_backprop ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let conv2DBackpropInput
    ?(name = "Conv2DBackpropInput")
    (input_sizes : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Conv2DBackpropInput"
  ; output_type = filter.output_type
  ; inputs = [ P input_sizes; P filter; P out_backprop ]
  ; attributes = [
      "T", Type (P filter.output_type);
    ]
  }

let cos
    ?(name = "Cos")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Cos"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let countUpTo
    ?(name = "CountUpTo")
    (ref : ([< `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "CountUpTo"
  ; output_type = ref.output_type
  ; inputs = [ P ref ]
  ; attributes = [
      "T", Type (P ref.output_type);
    ]
  }

let cross
    ?(name = "Cross")
    (a : ([< `float | `double | `int32 ] as 't) t)
    (b : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Cross"
  ; output_type = a.output_type
  ; inputs = [ P a; P b ]
  ; attributes = [
      "T", Type (P a.output_type);
    ]
  }

let depthToSpace
    ?(name = "DepthToSpace")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "DepthToSpace"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let depthwiseConv2dNative
    ?(name = "DepthwiseConv2dNative")
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "DepthwiseConv2dNative"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let depthwiseConv2dNativeBackpropFilter
    ?(name = "DepthwiseConv2dNativeBackpropFilter")
    (input : ([< `float | `double ] as 't) t)
    (filter_sizes : [ `int32 ] t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "DepthwiseConv2dNativeBackpropFilter"
  ; output_type = input.output_type
  ; inputs = [ P input; P filter_sizes; P out_backprop ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let depthwiseConv2dNativeBackpropInput
    ?(name = "DepthwiseConv2dNativeBackpropInput")
    (input_sizes : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "DepthwiseConv2dNativeBackpropInput"
  ; output_type = filter.output_type
  ; inputs = [ P input_sizes; P filter; P out_backprop ]
  ; attributes = [
      "T", Type (P filter.output_type);
    ]
  }

let destroyTemporaryVariable
    ?(name = "DestroyTemporaryVariable")
    (ref : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "DestroyTemporaryVariable"
  ; output_type = ref.output_type
  ; inputs = [ P ref ]
  ; attributes = [
      "T", Type (P ref.output_type);
    ]
  }

let diag
    ?(name = "Diag")
    (diagonal : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Diag"
  ; output_type = diagonal.output_type
  ; inputs = [ P diagonal ]
  ; attributes = [
      "T", Type (P diagonal.output_type);
    ]
  }

let diagPart
    ?(name = "DiagPart")
    (input : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "DiagPart"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let digamma
    ?(name = "Digamma")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Digamma"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let div
    ?(name = "Div")
    (x : ([< `float | `double | `int32 ] as 't) t)
    (y : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Div"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let drawBoundingBoxes
    ?(name = "DrawBoundingBoxes")
    (images : [ `float ] t)
    (boxes : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "DrawBoundingBoxes"
  ; output_type = Type.Float
  ; inputs = [ P images; P boxes ]
  ; attributes = [
    ]
  }

let dynamicPartition
    ?(name = "DynamicPartition")
    (data : 't t)
    (partitions : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "DynamicPartition"
  ; output_type = data.output_type
  ; inputs = [ P data; P partitions ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let dynamicStitch
    ?(name = "DynamicStitch")
    (indices : [ `int32 ] t)
    (data : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "DynamicStitch"
  ; output_type = data.output_type
  ; inputs = [ P indices; P data ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let elu
    ?(name = "Elu")
    (features : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Elu"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes = [
      "T", Type (P features.output_type);
    ]
  }

let eluGrad
    ?(name = "EluGrad")
    (gradients : ([< `float | `double ] as 't) t)
    (outputs : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "EluGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P outputs ]
  ; attributes = [
      "T", Type (P gradients.output_type);
    ]
  }

let enter
    ?(name = "Enter")
    (data : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Enter"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let erf
    ?(name = "Erf")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Erf"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let erfc
    ?(name = "Erfc")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Erfc"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let exit
    ?(name = "Exit")
    (data : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Exit"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let exp
    ?(name = "Exp")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Exp"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let expandDims
    ?(name = "ExpandDims")
    (input : 't t)
    (dim : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ExpandDims"
  ; output_type = input.output_type
  ; inputs = [ P input; P dim ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let extractGlimpse
    ?(name = "ExtractGlimpse")
    (input : [ `float ] t)
    (size : [ `int32 ] t)
    (offsets : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ExtractGlimpse"
  ; output_type = Type.Float
  ; inputs = [ P input; P size; P offsets ]
  ; attributes = [
    ]
  }

let fill
    ?(name = "Fill")
    (dims : [ `int32 ] t)
    (value : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Fill"
  ; output_type = value.output_type
  ; inputs = [ P dims; P value ]
  ; attributes = [
      "T", Type (P value.output_type);
    ]
  }

let floor
    ?(name = "Floor")
    (x : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Floor"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let gather
    ?(name = "Gather")
    (params : 'tparams t)
    (indices : ([< `int32 ] as 'tindices) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Gather"
  ; output_type = params.output_type
  ; inputs = [ P params; P indices ]
  ; attributes = [
      "Tparams", Type (P params.output_type);
    ]
  }

let hSVToRGB
    ?(name = "HSVToRGB")
    (images : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "HSVToRGB"
  ; output_type = Type.Float
  ; inputs = [ P images ]
  ; attributes = [
    ]
  }

let identity
    ?(name = "Identity")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Identity"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let inv
    ?(name = "Inv")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Inv"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let invertPermutation
    ?(name = "InvertPermutation")
    (x : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "InvertPermutation"
  ; output_type = Type.Int32
  ; inputs = [ P x ]
  ; attributes = [
    ]
  }

let l2Loss
    ?(name = "L2Loss")
    (t : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "L2Loss"
  ; output_type = t.output_type
  ; inputs = [ P t ]
  ; attributes = [
      "T", Type (P t.output_type);
    ]
  }

let lRN
    ?(name = "LRN")
    (input : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "LRN"
  ; output_type = Type.Float
  ; inputs = [ P input ]
  ; attributes = [
    ]
  }

let lRNGrad
    ?(name = "LRNGrad")
    (input_grads : [ `float ] t)
    (input_image : [ `float ] t)
    (output_image : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "LRNGrad"
  ; output_type = Type.Float
  ; inputs = [ P input_grads; P input_image; P output_image ]
  ; attributes = [
    ]
  }

let lgamma
    ?(name = "Lgamma")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Lgamma"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let linSpace
    ?(name = "LinSpace")
    (start : ([< `float | `double ] as 't) t)
    (stop : ([< `float | `double ] as 't) t)
    (num : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "LinSpace"
  ; output_type = start.output_type
  ; inputs = [ P start; P stop; P num ]
  ; attributes = [
      "T", Type (P start.output_type);
    ]
  }

let log
    ?(name = "Log")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Log"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let matMul
    ?(name = "MatMul")
    (a : ([< `float | `double | `int32 ] as 't) t)
    (b : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MatMul"
  ; output_type = a.output_type
  ; inputs = [ P a; P b ]
  ; attributes = [
      "T", Type (P a.output_type);
    ]
  }

let matrixDeterminant
    ?(name = "MatrixDeterminant")
    (input : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MatrixDeterminant"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let matrixInverse
    ?(name = "MatrixInverse")
    (input : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MatrixInverse"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let matrixSolve
    ?(name = "MatrixSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MatrixSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes = [
      "T", Type (P matrix.output_type);
    ]
  }

let matrixSolveLs
    ?(name = "MatrixSolveLs")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
    (l2_regularizer : [ `double ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MatrixSolveLs"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs; P l2_regularizer ]
  ; attributes = [
      "T", Type (P matrix.output_type);
    ]
  }

let matrixTriangularSolve
    ?(name = "MatrixTriangularSolve")
    (matrix : ([< `float | `double ] as 't) t)
    (rhs : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MatrixTriangularSolve"
  ; output_type = matrix.output_type
  ; inputs = [ P matrix; P rhs ]
  ; attributes = [
      "T", Type (P matrix.output_type);
    ]
  }

let max
    ?(name = "Max")
    (input : ([< `float | `double | `int32 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Max"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let maxPool
    ?(name = "MaxPool")
    (input : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MaxPool"
  ; output_type = Type.Float
  ; inputs = [ P input ]
  ; attributes = [
    ]
  }

let maxPoolGrad
    ?(name = "MaxPoolGrad")
    (orig_input : [ `float ] t)
    (orig_output : [ `float ] t)
    (grad : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MaxPoolGrad"
  ; output_type = Type.Float
  ; inputs = [ P orig_input; P orig_output; P grad ]
  ; attributes = [
    ]
  }

let maxPoolGradWithArgmax
    ?(name = "MaxPoolGradWithArgmax")
    (input : [ `float ] t)
    (grad : [ `float ] t)
    (argmax : ([< `int32 ] as 'targmax) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MaxPoolGradWithArgmax"
  ; output_type = Type.Float
  ; inputs = [ P input; P grad; P argmax ]
  ; attributes = [
    ]
  }

let maximum
    ?(name = "Maximum")
    (x : ([< `float | `double | `int32 ] as 't) t)
    (y : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Maximum"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let mean
    ?(name = "Mean")
    (input : ([< `float | `double | `int32 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Mean"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let min
    ?(name = "Min")
    (input : ([< `float | `double | `int32 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Min"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let minimum
    ?(name = "Minimum")
    (x : ([< `float | `double | `int32 ] as 't) t)
    (y : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Minimum"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let mirrorPad
    ?(name = "MirrorPad")
    (input : 't t)
    (paddings : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MirrorPad"
  ; output_type = input.output_type
  ; inputs = [ P input; P paddings ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let mirrorPadGrad
    ?(name = "MirrorPadGrad")
    (input : 't t)
    (paddings : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "MirrorPadGrad"
  ; output_type = input.output_type
  ; inputs = [ P input; P paddings ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let mod_
    ?(name = "Mod")
    (x : ([< `int32 | `float | `double ] as 't) t)
    (y : ([< `int32 | `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Mod"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let mul
    ?(name = "Mul")
    (x : ([< `float | `double | `int32 ] as 't) t)
    (y : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Mul"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let neg
    ?(name = "Neg")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Neg"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let negTrain
    ?(name = "NegTrain")
    (w_in : [ `float ] t)
    (w_out : [ `float ] t)
    (examples : [ `int32 ] t)
    (labels : [ `int32 ] t)
    (lr : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "NegTrain"
  ; output_type = Type.Unit
  ; inputs = [ P w_in; P w_out; P examples; P labels; P lr ]
  ; attributes = [
    ]
  }

let nextIteration
    ?(name = "NextIteration")
    (data : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "NextIteration"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let noOp
    ?(name = "NoOp")
    ()
  =
  { name = Name.make_fresh ~name
  ; op_name = "NoOp"
  ; output_type = Type.Unit
  ; inputs = [  ]
  ; attributes = [
    ]
  }

let pack
    ?(name = "Pack")
    (values : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Pack"
  ; output_type = values.output_type
  ; inputs = [ P values ]
  ; attributes = [
      "T", Type (P values.output_type);
    ]
  }

let pad
    ?(name = "Pad")
    (input : 't t)
    (paddings : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Pad"
  ; output_type = input.output_type
  ; inputs = [ P input; P paddings ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let placeholder
    ?(name = "Placeholder")
    ~type_
  =
  { name = Name.make_fresh ~name
  ; op_name = "Placeholder"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes = [
      "dtype", Type (P type_);
    ]
  }

let pow
    ?(name = "Pow")
    (x : ([< `float | `double | `int32 ] as 't) t)
    (y : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Pow"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let prod
    ?(name = "Prod")
    (input : ([< `float | `double | `int32 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Prod"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let rGBToHSV
    ?(name = "RGBToHSV")
    (images : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RGBToHSV"
  ; output_type = Type.Float
  ; inputs = [ P images ]
  ; attributes = [
    ]
  }

let randomShuffle
    ?(name = "RandomShuffle")
    (value : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RandomShuffle"
  ; output_type = value.output_type
  ; inputs = [ P value ]
  ; attributes = [
      "T", Type (P value.output_type);
    ]
  }

let randomStandardNormal
    ?(name = "RandomStandardNormal")
    ~type_
    (shape : ([< `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RandomStandardNormal"
  ; output_type = type_
  ; inputs = [ P shape ]
  ; attributes = [
      "dtype", Type (P type_);
    ]
  }

let randomUniform
    ?(name = "RandomUniform")
    ~type_
    (shape : ([< `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RandomUniform"
  ; output_type = type_
  ; inputs = [ P shape ]
  ; attributes = [
      "dtype", Type (P type_);
    ]
  }

let randomUniformInt
    ?(name = "RandomUniformInt")
    (shape : ([< `int32 ] as 't) t)
    (minval : ([< `int32 ] as 'tout) t)
    (maxval : ([< `int32 ] as 'tout) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RandomUniformInt"
  ; output_type = minval.output_type
  ; inputs = [ P shape; P minval; P maxval ]
  ; attributes = [
      "Tout", Type (P minval.output_type);
    ]
  }

let range
    ?(name = "Range")
    (start : [ `int32 ] t)
    (limit : [ `int32 ] t)
    (delta : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Range"
  ; output_type = Type.Int32
  ; inputs = [ P start; P limit; P delta ]
  ; attributes = [
    ]
  }

let rank
    ?(name = "Rank")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Rank"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes = [
    ]
  }

let refEnter
    ?(name = "RefEnter")
    (data : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RefEnter"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let refExit
    ?(name = "RefExit")
    (data : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RefExit"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let refIdentity
    ?(name = "RefIdentity")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RefIdentity"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let refNextIteration
    ?(name = "RefNextIteration")
    (data : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RefNextIteration"
  ; output_type = data.output_type
  ; inputs = [ P data ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let refSelect
    ?(name = "RefSelect")
    (index : [ `int32 ] t)
    (inputs : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "RefSelect"
  ; output_type = inputs.output_type
  ; inputs = [ P index; P inputs ]
  ; attributes = [
      "T", Type (P inputs.output_type);
    ]
  }

let relu
    ?(name = "Relu")
    (features : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Relu"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes = [
      "T", Type (P features.output_type);
    ]
  }

let relu6
    ?(name = "Relu6")
    (features : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Relu6"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes = [
      "T", Type (P features.output_type);
    ]
  }

let relu6Grad
    ?(name = "Relu6Grad")
    (gradients : ([< `float | `double | `int32 ] as 't) t)
    (features : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Relu6Grad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes = [
      "T", Type (P gradients.output_type);
    ]
  }

let reluGrad
    ?(name = "ReluGrad")
    (gradients : ([< `float | `double | `int32 ] as 't) t)
    (features : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ReluGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes = [
      "T", Type (P gradients.output_type);
    ]
  }

let reshape
    ?(name = "Reshape")
    (tensor : 't t)
    (shape : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Reshape"
  ; output_type = tensor.output_type
  ; inputs = [ P tensor; P shape ]
  ; attributes = [
      "T", Type (P tensor.output_type);
    ]
  }

let resizeArea
    ?(name = "ResizeArea")
    (images : ([< `int32 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ResizeArea"
  ; output_type = Type.Float
  ; inputs = [ P images; P size ]
  ; attributes = [
    ]
  }

let resizeBicubic
    ?(name = "ResizeBicubic")
    (images : ([< `int32 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ResizeBicubic"
  ; output_type = Type.Float
  ; inputs = [ P images; P size ]
  ; attributes = [
    ]
  }

let resizeBilinear
    ?(name = "ResizeBilinear")
    (images : ([< `int32 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ResizeBilinear"
  ; output_type = Type.Float
  ; inputs = [ P images; P size ]
  ; attributes = [
    ]
  }

let resizeBilinearGrad
    ?(name = "ResizeBilinearGrad")
    (grads : [ `float ] t)
    (original_image : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ResizeBilinearGrad"
  ; output_type = original_image.output_type
  ; inputs = [ P grads; P original_image ]
  ; attributes = [
      "T", Type (P original_image.output_type);
    ]
  }

let resizeNearestNeighbor
    ?(name = "ResizeNearestNeighbor")
    (images : ([< `int32 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ResizeNearestNeighbor"
  ; output_type = images.output_type
  ; inputs = [ P images; P size ]
  ; attributes = [
      "T", Type (P images.output_type);
    ]
  }

let resizeNearestNeighborGrad
    ?(name = "ResizeNearestNeighborGrad")
    (grads : ([< `int32 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ResizeNearestNeighborGrad"
  ; output_type = grads.output_type
  ; inputs = [ P grads; P size ]
  ; attributes = [
      "T", Type (P grads.output_type);
    ]
  }

let rsqrt
    ?(name = "Rsqrt")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Rsqrt"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let scatterAdd
    ?(name = "ScatterAdd")
    (ref : ([< `float | `double | `int32 ] as 't) t)
    (indices : ([< `int32 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ScatterAdd"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P indices; P updates ]
  ; attributes = [
      "T", Type (P ref.output_type);
    ]
  }

let scatterSub
    ?(name = "ScatterSub")
    (ref : ([< `float | `double | `int32 ] as 't) t)
    (indices : ([< `int32 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ScatterSub"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P indices; P updates ]
  ; attributes = [
      "T", Type (P ref.output_type);
    ]
  }

let scatterUpdate
    ?(name = "ScatterUpdate")
    (ref : 't t)
    (indices : ([< `int32 ] as 'tindices) t)
    (updates : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ScatterUpdate"
  ; output_type = ref.output_type
  ; inputs = [ P ref; P indices; P updates ]
  ; attributes = [
      "T", Type (P ref.output_type);
    ]
  }

let segmentMax
    ?(name = "SegmentMax")
    (data : ([< `float | `double | `int32 ] as 't) t)
    (segment_ids : ([< `int32 ] as 'tindices) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SegmentMax"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let segmentMean
    ?(name = "SegmentMean")
    (data : ([< `float | `double | `int32 ] as 't) t)
    (segment_ids : ([< `int32 ] as 'tindices) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SegmentMean"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let segmentMin
    ?(name = "SegmentMin")
    (data : ([< `float | `double | `int32 ] as 't) t)
    (segment_ids : ([< `int32 ] as 'tindices) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SegmentMin"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let segmentProd
    ?(name = "SegmentProd")
    (data : ([< `float | `double | `int32 ] as 't) t)
    (segment_ids : ([< `int32 ] as 'tindices) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SegmentProd"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let segmentSum
    ?(name = "SegmentSum")
    (data : ([< `float | `double | `int32 ] as 't) t)
    (segment_ids : ([< `int32 ] as 'tindices) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SegmentSum"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let selfAdjointEig
    ?(name = "SelfAdjointEig")
    (input : ([< `double | `float ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SelfAdjointEig"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let shape
    ?(name = "Shape")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Shape"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes = [
    ]
  }

let shapeN
    ?(name = "ShapeN")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ShapeN"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes = [
    ]
  }

let sigmoid
    ?(name = "Sigmoid")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Sigmoid"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let sign
    ?(name = "Sign")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Sign"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let sin
    ?(name = "Sin")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Sin"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let size
    ?(name = "Size")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Size"
  ; output_type = Type.Int32
  ; inputs = [ P input ]
  ; attributes = [
    ]
  }

let slice
    ?(name = "Slice")
    (input : 't t)
    (begin__ : ([< `int32 ] as 'index) t)
    (size : ([< `int32 ] as 'index) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Slice"
  ; output_type = input.output_type
  ; inputs = [ P input; P begin__; P size ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let softmax
    ?(name = "Softmax")
    (logits : ([< `float | `double ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Softmax"
  ; output_type = logits.output_type
  ; inputs = [ P logits ]
  ; attributes = [
      "T", Type (P logits.output_type);
    ]
  }

let softplus
    ?(name = "Softplus")
    (features : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Softplus"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes = [
      "T", Type (P features.output_type);
    ]
  }

let softplusGrad
    ?(name = "SoftplusGrad")
    (gradients : ([< `float | `double | `int32 ] as 't) t)
    (features : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SoftplusGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes = [
      "T", Type (P gradients.output_type);
    ]
  }

let softsign
    ?(name = "Softsign")
    (features : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Softsign"
  ; output_type = features.output_type
  ; inputs = [ P features ]
  ; attributes = [
      "T", Type (P features.output_type);
    ]
  }

let softsignGrad
    ?(name = "SoftsignGrad")
    (gradients : ([< `float | `double | `int32 ] as 't) t)
    (features : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SoftsignGrad"
  ; output_type = gradients.output_type
  ; inputs = [ P gradients; P features ]
  ; attributes = [
      "T", Type (P gradients.output_type);
    ]
  }

let spaceToDepth
    ?(name = "SpaceToDepth")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SpaceToDepth"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let sparseApplyAdagrad
    ?(name = "SparseApplyAdagrad")
    (var : ([< `float | `double | `int32 ] as 't) t)
    (accum : ([< `float | `double | `int32 ] as 't) t)
    (lr : ([< `float | `double | `int32 ] as 't) t)
    (grad : ([< `float | `double | `int32 ] as 't) t)
    (indices : ([< `int32 ] as 'tindices) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseApplyAdagrad"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P indices ]
  ; attributes = [
      "T", Type (P var.output_type);
    ]
  }

let sparseApplyFtrl
    ?(name = "SparseApplyFtrl")
    (var : ([< `float | `double | `int32 ] as 't) t)
    (accum : ([< `float | `double | `int32 ] as 't) t)
    (linear : ([< `float | `double | `int32 ] as 't) t)
    (grad : ([< `float | `double | `int32 ] as 't) t)
    (indices : ([< `int32 ] as 'tindices) t)
    (lr : ([< `float | `double | `int32 ] as 't) t)
    (l1 : ([< `float | `double | `int32 ] as 't) t)
    (l2 : ([< `float | `double | `int32 ] as 't) t)
    (lr_power : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseApplyFtrl"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P linear; P grad; P indices; P lr; P l1; P l2; P lr_power ]
  ; attributes = [
      "T", Type (P var.output_type);
    ]
  }

let sparseApplyMomentum
    ?(name = "SparseApplyMomentum")
    (var : ([< `float | `double | `int32 ] as 't) t)
    (accum : ([< `float | `double | `int32 ] as 't) t)
    (lr : ([< `float | `double | `int32 ] as 't) t)
    (grad : ([< `float | `double | `int32 ] as 't) t)
    (indices : ([< `int32 ] as 'tindices) t)
    (momentum : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseApplyMomentum"
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P indices; P momentum ]
  ; attributes = [
      "T", Type (P var.output_type);
    ]
  }

let sparseMatMul
    ?(name = "SparseMatMul")
    (a : [ `float ] t)
    (b : [ `float ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseMatMul"
  ; output_type = Type.Float
  ; inputs = [ P a; P b ]
  ; attributes = [
    ]
  }

let sparseSegmentMean
    ?(name = "SparseSegmentMean")
    (data : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseSegmentMean"
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let sparseSegmentMeanGrad
    ?(name = "SparseSegmentMeanGrad")
    (grad : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
    (output_dim0 : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseSegmentMeanGrad"
  ; output_type = grad.output_type
  ; inputs = [ P grad; P indices; P segment_ids; P output_dim0 ]
  ; attributes = [
      "T", Type (P grad.output_type);
    ]
  }

let sparseSegmentSqrtN
    ?(name = "SparseSegmentSqrtN")
    (data : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseSegmentSqrtN"
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let sparseSegmentSqrtNGrad
    ?(name = "SparseSegmentSqrtNGrad")
    (grad : ([< `float | `double ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
    (output_dim0 : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseSegmentSqrtNGrad"
  ; output_type = grad.output_type
  ; inputs = [ P grad; P indices; P segment_ids; P output_dim0 ]
  ; attributes = [
      "T", Type (P grad.output_type);
    ]
  }

let sparseSegmentSum
    ?(name = "SparseSegmentSum")
    (data : ([< `float | `double | `int32 ] as 't) t)
    (indices : [ `int32 ] t)
    (segment_ids : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseSegmentSum"
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let sparseToDense
    ?(name = "SparseToDense")
    (sparse_indices : ([< `int32 ] as 'tindices) t)
    (output_shape : ([< `int32 ] as 'tindices) t)
    (sparse_values : 't t)
    (default_value : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SparseToDense"
  ; output_type = sparse_values.output_type
  ; inputs = [ P sparse_indices; P output_shape; P sparse_values; P default_value ]
  ; attributes = [
      "T", Type (P sparse_values.output_type);
    ]
  }

let split
    ?(name = "Split")
    (split_dim : [ `int32 ] t)
    (value : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Split"
  ; output_type = value.output_type
  ; inputs = [ P split_dim; P value ]
  ; attributes = [
      "T", Type (P value.output_type);
    ]
  }

let sqrt
    ?(name = "Sqrt")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Sqrt"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let square
    ?(name = "Square")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Square"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let squaredDifference
    ?(name = "SquaredDifference")
    (x : ([< `float | `double | `int32 ] as 't) t)
    (y : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "SquaredDifference"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let squeeze
    ?(name = "Squeeze")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Squeeze"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let stopGradient
    ?(name = "StopGradient")
    (input : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "StopGradient"
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let sub
    ?(name = "Sub")
    (x : ([< `float | `double | `int32 ] as 't) t)
    (y : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Sub"
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let sum
    ?(name = "Sum")
    (input : ([< `float | `double | `int32 ] as 't) t)
    (reduction_indices : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Sum"
  ; output_type = input.output_type
  ; inputs = [ P input; P reduction_indices ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let tanh
    ?(name = "Tanh")
    (x : ([< `float | `double | `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Tanh"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let temporaryVariable
    ?(name = "TemporaryVariable")
    ~type_
  =
  { name = Name.make_fresh ~name
  ; op_name = "TemporaryVariable"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes = [
      "dtype", Type (P type_);
    ]
  }

let tile
    ?(name = "Tile")
    (input : 't t)
    (multiples : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Tile"
  ; output_type = input.output_type
  ; inputs = [ P input; P multiples ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let tileGrad
    ?(name = "TileGrad")
    (input : 't t)
    (multiples : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "TileGrad"
  ; output_type = input.output_type
  ; inputs = [ P input; P multiples ]
  ; attributes = [
      "T", Type (P input.output_type);
    ]
  }

let transpose
    ?(name = "Transpose")
    (x : 't t)
    (perm : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Transpose"
  ; output_type = x.output_type
  ; inputs = [ P x; P perm ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

let truncatedNormal
    ?(name = "TruncatedNormal")
    ~type_
    (shape : ([< `int32 ] as 't) t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "TruncatedNormal"
  ; output_type = type_
  ; inputs = [ P shape ]
  ; attributes = [
      "dtype", Type (P type_);
    ]
  }

let unpack
    ?(name = "Unpack")
    (value : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "Unpack"
  ; output_type = value.output_type
  ; inputs = [ P value ]
  ; attributes = [
      "T", Type (P value.output_type);
    ]
  }

let unsortedSegmentSum
    ?(name = "UnsortedSegmentSum")
    (data : ([< `float | `double | `int32 ] as 't) t)
    (segment_ids : ([< `int32 ] as 'tindices) t)
    (num_segments : [ `int32 ] t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "UnsortedSegmentSum"
  ; output_type = data.output_type
  ; inputs = [ P data; P segment_ids; P num_segments ]
  ; attributes = [
      "T", Type (P data.output_type);
    ]
  }

let variable
    ?(name = "Variable")
    ~type_
  =
  { name = Name.make_fresh ~name
  ; op_name = "Variable"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes = [
      "dtype", Type (P type_);
    ]
  }

let zerosLike
    ?(name = "ZerosLike")
    (x : 't t)
  =
  { name = Name.make_fresh ~name
  ; op_name = "ZerosLike"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [
      "T", Type (P x.output_type);
    ]
  }

