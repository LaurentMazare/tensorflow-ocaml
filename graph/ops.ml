open Node

let abs
    ?(name = "Abs")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let add
    ?(name = "Add")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let addN
    ?(name = "AddN")
    (inputs : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = inputs.output_type
    ; inputs = [ P inputs ]
    ; attributes = []
    }

let adjustContrast
    ?(name = "AdjustContrast")
    (images : ([< `float | `double ] as 't) Node.t)
    (contrast_factor : [ `float ] Node.t)
    (min_value : [ `float ] Node.t)
    (max_value : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P images; P contrast_factor; P min_value; P max_value ]
    ; attributes = []
    }

let adjustContrastv2
    ?(name = "AdjustContrastv2")
    (images : [ `float ] Node.t)
    (contrast_factor : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P images; P contrast_factor ]
    ; attributes = []
    }

let applyAdagrad
    ?(name = "ApplyAdagrad")
    (var : ([< `float | `double ] as 't) Node.t)
    (accum : ([< `float | `double ] as 't) Node.t)
    (lr : ([< `float | `double ] as 't) Node.t)
    (grad : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = var.output_type
    ; inputs = [ P var; P accum; P lr; P grad ]
    ; attributes = []
    }

let applyAdam
    ?(name = "ApplyAdam")
    (var : ([< `float | `double ] as 't) Node.t)
    (m : ([< `float | `double ] as 't) Node.t)
    (v : ([< `float | `double ] as 't) Node.t)
    (beta1_power : ([< `float | `double ] as 't) Node.t)
    (beta2_power : ([< `float | `double ] as 't) Node.t)
    (lr : ([< `float | `double ] as 't) Node.t)
    (beta1 : ([< `float | `double ] as 't) Node.t)
    (beta2 : ([< `float | `double ] as 't) Node.t)
    (epsilon : ([< `float | `double ] as 't) Node.t)
    (grad : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = var.output_type
    ; inputs = [ P var; P m; P v; P beta1_power; P beta2_power; P lr; P beta1; P beta2; P epsilon; P grad ]
    ; attributes = []
    }

let applyFtrl
    ?(name = "ApplyFtrl")
    (var : ([< `float | `double ] as 't) Node.t)
    (accum : ([< `float | `double ] as 't) Node.t)
    (linear : ([< `float | `double ] as 't) Node.t)
    (grad : ([< `float | `double ] as 't) Node.t)
    (lr : ([< `float | `double ] as 't) Node.t)
    (l1 : ([< `float | `double ] as 't) Node.t)
    (l2 : ([< `float | `double ] as 't) Node.t)
    (lr_power : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = var.output_type
    ; inputs = [ P var; P accum; P linear; P grad; P lr; P l1; P l2; P lr_power ]
    ; attributes = []
    }

let applyGradientDescent
    ?(name = "ApplyGradientDescent")
    (var : ([< `float | `double ] as 't) Node.t)
    (alpha : ([< `float | `double ] as 't) Node.t)
    (delta : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = var.output_type
    ; inputs = [ P var; P alpha; P delta ]
    ; attributes = []
    }

let applyMomentum
    ?(name = "ApplyMomentum")
    (var : ([< `float | `double ] as 't) Node.t)
    (accum : ([< `float | `double ] as 't) Node.t)
    (lr : ([< `float | `double ] as 't) Node.t)
    (grad : ([< `float | `double ] as 't) Node.t)
    (momentum : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = var.output_type
    ; inputs = [ P var; P accum; P lr; P grad; P momentum ]
    ; attributes = []
    }

let applyRMSProp
    ?(name = "ApplyRMSProp")
    (var : ([< `float | `double ] as 't) Node.t)
    (ms : ([< `float | `double ] as 't) Node.t)
    (mom : ([< `float | `double ] as 't) Node.t)
    (lr : ([< `float | `double ] as 't) Node.t)
    (rho : ([< `float | `double ] as 't) Node.t)
    (momentum : ([< `float | `double ] as 't) Node.t)
    (epsilon : ([< `float | `double ] as 't) Node.t)
    (grad : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = var.output_type
    ; inputs = [ P var; P ms; P mom; P lr; P rho; P momentum; P epsilon; P grad ]
    ; attributes = []
    }

let assign
    ?(name = "Assign")
    (ref : 't Node.t)
    (value : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = ref.output_type
    ; inputs = [ P ref; P value ]
    ; attributes = []
    }

let assignAdd
    ?(name = "AssignAdd")
    (ref : ([< `float | `double ] as 't) Node.t)
    (value : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = ref.output_type
    ; inputs = [ P ref; P value ]
    ; attributes = []
    }

let assignSub
    ?(name = "AssignSub")
    (ref : ([< `float | `double ] as 't) Node.t)
    (value : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = ref.output_type
    ; inputs = [ P ref; P value ]
    ; attributes = []
    }

let avgPool
    ?(name = "AvgPool")
    (value : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = value.output_type
    ; inputs = [ P value ]
    ; attributes = []
    }

let batchCholesky
    ?(name = "BatchCholesky")
    (input : ([< `double | `float ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let batchMatMul
    ?(name = "BatchMatMul")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let batchMatrixDeterminant
    ?(name = "BatchMatrixDeterminant")
    (input : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let batchMatrixInverse
    ?(name = "BatchMatrixInverse")
    (input : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let batchMatrixSolve
    ?(name = "BatchMatrixSolve")
    (matrix : ([< `float | `double ] as 't) Node.t)
    (rhs : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = matrix.output_type
    ; inputs = [ P matrix; P rhs ]
    ; attributes = []
    }

let batchMatrixSolveLs
    ?(name = "BatchMatrixSolveLs")
    (matrix : ([< `float | `double ] as 't) Node.t)
    (rhs : ([< `float | `double ] as 't) Node.t)
    (l2_regularizer : [ `double ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = matrix.output_type
    ; inputs = [ P matrix; P rhs; P l2_regularizer ]
    ; attributes = []
    }

let batchMatrixTriangularSolve
    ?(name = "BatchMatrixTriangularSolve")
    (matrix : ([< `float | `double ] as 't) Node.t)
    (rhs : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = matrix.output_type
    ; inputs = [ P matrix; P rhs ]
    ; attributes = []
    }

let batchNormWithGlobalNormalization
    ?(name = "BatchNormWithGlobalNormalization")
    (t : ([< `float | `double ] as 't) Node.t)
    (m : ([< `float | `double ] as 't) Node.t)
    (v : ([< `float | `double ] as 't) Node.t)
    (beta : ([< `float | `double ] as 't) Node.t)
    (gamma : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = t.output_type
    ; inputs = [ P t; P m; P v; P beta; P gamma ]
    ; attributes = []
    }

let batchSelfAdjointEig
    ?(name = "BatchSelfAdjointEig")
    (input : ([< `double | `float ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let biasAdd
    ?(name = "BiasAdd")
    (value : ([< `float | `double ] as 't) Node.t)
    (bias : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = value.output_type
    ; inputs = [ P value; P bias ]
    ; attributes = []
    }

let biasAddGrad
    ?(name = "BiasAddGrad")
    (out_backprop : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = out_backprop.output_type
    ; inputs = [ P out_backprop ]
    ; attributes = []
    }

let biasAddV1
    ?(name = "BiasAddV1")
    (value : ([< `float | `double ] as 't) Node.t)
    (bias : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = value.output_type
    ; inputs = [ P value; P bias ]
    ; attributes = []
    }

let bitcast
    ?(name = "Bitcast")
    (input : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = TODO: add a parameter
    ; inputs = [ P input ]
    ; attributes = []
    }

let cast
    ?(name = "Cast")
    (x : 'srcT Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = TODO: add a parameter
    ; inputs = [ P x ]
    ; attributes = []
    }

let ceil
    ?(name = "Ceil")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let checkNumerics
    ?(name = "CheckNumerics")
    (tensor : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = tensor.output_type
    ; inputs = [ P tensor ]
    ; attributes = []
    }

let cholesky
    ?(name = "Cholesky")
    (input : ([< `double | `float ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let const
    ?(name = "Const")
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = TODO: add a parameter
    ; inputs = [  ]
    ; attributes = []
    }

let controlTrigger
    ?(name = "ControlTrigger")
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Unit)
    ; inputs = [  ]
    ; attributes = []
    }

let conv2D
    ?(name = "Conv2D")
    (input : ([< `float | `double ] as 't) Node.t)
    (filter : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input; P filter ]
    ; attributes = []
    }

let cos
    ?(name = "Cos")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let countUpTo
    ?(name = "CountUpTo")
    (ref : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = ref.output_type
    ; inputs = [ P ref ]
    ; attributes = []
    }

let cross
    ?(name = "Cross")
    (a : ([< `float | `double ] as 't) Node.t)
    (b : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = a.output_type
    ; inputs = [ P a; P b ]
    ; attributes = []
    }

let depthToSpace
    ?(name = "DepthToSpace")
    (input : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let depthwiseConv2dNative
    ?(name = "DepthwiseConv2dNative")
    (input : ([< `float | `double ] as 't) Node.t)
    (filter : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input; P filter ]
    ; attributes = []
    }

let destroyTemporaryVariable
    ?(name = "DestroyTemporaryVariable")
    (ref : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = ref.output_type
    ; inputs = [ P ref ]
    ; attributes = []
    }

let diag
    ?(name = "Diag")
    (diagonal : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = diagonal.output_type
    ; inputs = [ P diagonal ]
    ; attributes = []
    }

let diagPart
    ?(name = "DiagPart")
    (input : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let digamma
    ?(name = "Digamma")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let div
    ?(name = "Div")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let drawBoundingBoxes
    ?(name = "DrawBoundingBoxes")
    (images : [ `float ] Node.t)
    (boxes : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P images; P boxes ]
    ; attributes = []
    }

let elu
    ?(name = "Elu")
    (features : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = features.output_type
    ; inputs = [ P features ]
    ; attributes = []
    }

let eluGrad
    ?(name = "EluGrad")
    (gradients : ([< `float | `double ] as 't) Node.t)
    (outputs : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = gradients.output_type
    ; inputs = [ P gradients; P outputs ]
    ; attributes = []
    }

let enter
    ?(name = "Enter")
    (data : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data ]
    ; attributes = []
    }

let erf
    ?(name = "Erf")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let erfc
    ?(name = "Erfc")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let exit
    ?(name = "Exit")
    (data : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data ]
    ; attributes = []
    }

let exp
    ?(name = "Exp")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let floor
    ?(name = "Floor")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let gather
    ?(name = "Gather")
    (params : 'tparams Node.t)
    (indices : 'tindices Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = params.output_type
    ; inputs = [ P params; P indices ]
    ; attributes = []
    }

let hSVToRGB
    ?(name = "HSVToRGB")
    (images : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P images ]
    ; attributes = []
    }

let identity
    ?(name = "Identity")
    (input : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let inv
    ?(name = "Inv")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let l2Loss
    ?(name = "L2Loss")
    (t : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = t.output_type
    ; inputs = [ P t ]
    ; attributes = []
    }

let lRN
    ?(name = "LRN")
    (input : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P input ]
    ; attributes = []
    }

let lRNGrad
    ?(name = "LRNGrad")
    (input_grads : [ `float ] Node.t)
    (input_image : [ `float ] Node.t)
    (output_image : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P input_grads; P input_image; P output_image ]
    ; attributes = []
    }

let lgamma
    ?(name = "Lgamma")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let log
    ?(name = "Log")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let matMul
    ?(name = "MatMul")
    (a : ([< `float | `double ] as 't) Node.t)
    (b : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = a.output_type
    ; inputs = [ P a; P b ]
    ; attributes = []
    }

let matrixDeterminant
    ?(name = "MatrixDeterminant")
    (input : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let matrixInverse
    ?(name = "MatrixInverse")
    (input : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let matrixSolve
    ?(name = "MatrixSolve")
    (matrix : ([< `float | `double ] as 't) Node.t)
    (rhs : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = matrix.output_type
    ; inputs = [ P matrix; P rhs ]
    ; attributes = []
    }

let matrixSolveLs
    ?(name = "MatrixSolveLs")
    (matrix : ([< `float | `double ] as 't) Node.t)
    (rhs : ([< `float | `double ] as 't) Node.t)
    (l2_regularizer : [ `double ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = matrix.output_type
    ; inputs = [ P matrix; P rhs; P l2_regularizer ]
    ; attributes = []
    }

let matrixTriangularSolve
    ?(name = "MatrixTriangularSolve")
    (matrix : ([< `float | `double ] as 't) Node.t)
    (rhs : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = matrix.output_type
    ; inputs = [ P matrix; P rhs ]
    ; attributes = []
    }

let maxPool
    ?(name = "MaxPool")
    (input : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P input ]
    ; attributes = []
    }

let maxPoolGrad
    ?(name = "MaxPoolGrad")
    (orig_input : [ `float ] Node.t)
    (orig_output : [ `float ] Node.t)
    (grad : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P orig_input; P orig_output; P grad ]
    ; attributes = []
    }

let maxPoolGradWithArgmax
    ?(name = "MaxPoolGradWithArgmax")
    (input : [ `float ] Node.t)
    (grad : [ `float ] Node.t)
    (argmax : 'targmax Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P input; P grad; P argmax ]
    ; attributes = []
    }

let maximum
    ?(name = "Maximum")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let minimum
    ?(name = "Minimum")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let mod_
    ?(name = "Mod")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let mul
    ?(name = "Mul")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let neg
    ?(name = "Neg")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let nextIteration
    ?(name = "NextIteration")
    (data : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data ]
    ; attributes = []
    }

let noOp
    ?(name = "NoOp")
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Unit)
    ; inputs = [  ]
    ; attributes = []
    }

let pack
    ?(name = "Pack")
    (values : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = values.output_type
    ; inputs = [ P values ]
    ; attributes = []
    }

let placeholder
    ?(name = "Placeholder")
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = TODO: add a parameter
    ; inputs = [  ]
    ; attributes = []
    }

let pow
    ?(name = "Pow")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let rGBToHSV
    ?(name = "RGBToHSV")
    (images : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P images ]
    ; attributes = []
    }

let randomShuffle
    ?(name = "RandomShuffle")
    (value : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = value.output_type
    ; inputs = [ P value ]
    ; attributes = []
    }

let randomStandardNormal
    ?(name = "RandomStandardNormal")
    (shape : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = TODO: add a parameter
    ; inputs = [ P shape ]
    ; attributes = []
    }

let randomUniform
    ?(name = "RandomUniform")
    (shape : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = TODO: add a parameter
    ; inputs = [ P shape ]
    ; attributes = []
    }

let randomUniformInt
    ?(name = "RandomUniformInt")
    (shape : 't Node.t)
    (minval : 'tout Node.t)
    (maxval : 'tout Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = minval.output_type
    ; inputs = [ P shape; P minval; P maxval ]
    ; attributes = []
    }

let refEnter
    ?(name = "RefEnter")
    (data : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data ]
    ; attributes = []
    }

let refExit
    ?(name = "RefExit")
    (data : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data ]
    ; attributes = []
    }

let refIdentity
    ?(name = "RefIdentity")
    (input : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let refNextIteration
    ?(name = "RefNextIteration")
    (data : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data ]
    ; attributes = []
    }

let relu
    ?(name = "Relu")
    (features : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = features.output_type
    ; inputs = [ P features ]
    ; attributes = []
    }

let relu6
    ?(name = "Relu6")
    (features : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = features.output_type
    ; inputs = [ P features ]
    ; attributes = []
    }

let relu6Grad
    ?(name = "Relu6Grad")
    (gradients : ([< `float | `double ] as 't) Node.t)
    (features : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = gradients.output_type
    ; inputs = [ P gradients; P features ]
    ; attributes = []
    }

let reluGrad
    ?(name = "ReluGrad")
    (gradients : ([< `float | `double ] as 't) Node.t)
    (features : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = gradients.output_type
    ; inputs = [ P gradients; P features ]
    ; attributes = []
    }

let resizeBilinearGrad
    ?(name = "ResizeBilinearGrad")
    (grads : [ `float ] Node.t)
    (original_image : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = original_image.output_type
    ; inputs = [ P grads; P original_image ]
    ; attributes = []
    }

let rsqrt
    ?(name = "Rsqrt")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let scatterAdd
    ?(name = "ScatterAdd")
    (ref : ([< `float | `double ] as 't) Node.t)
    (indices : 'tindices Node.t)
    (updates : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = ref.output_type
    ; inputs = [ P ref; P indices; P updates ]
    ; attributes = []
    }

let scatterSub
    ?(name = "ScatterSub")
    (ref : ([< `float | `double ] as 't) Node.t)
    (indices : 'tindices Node.t)
    (updates : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = ref.output_type
    ; inputs = [ P ref; P indices; P updates ]
    ; attributes = []
    }

let scatterUpdate
    ?(name = "ScatterUpdate")
    (ref : 't Node.t)
    (indices : 'tindices Node.t)
    (updates : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = ref.output_type
    ; inputs = [ P ref; P indices; P updates ]
    ; attributes = []
    }

let segmentMax
    ?(name = "SegmentMax")
    (data : ([< `float | `double ] as 't) Node.t)
    (segment_ids : 'tindices Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data; P segment_ids ]
    ; attributes = []
    }

let segmentMean
    ?(name = "SegmentMean")
    (data : ([< `float | `double ] as 't) Node.t)
    (segment_ids : 'tindices Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data; P segment_ids ]
    ; attributes = []
    }

let segmentMin
    ?(name = "SegmentMin")
    (data : ([< `float | `double ] as 't) Node.t)
    (segment_ids : 'tindices Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data; P segment_ids ]
    ; attributes = []
    }

let segmentProd
    ?(name = "SegmentProd")
    (data : ([< `float | `double ] as 't) Node.t)
    (segment_ids : 'tindices Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data; P segment_ids ]
    ; attributes = []
    }

let segmentSum
    ?(name = "SegmentSum")
    (data : ([< `float | `double ] as 't) Node.t)
    (segment_ids : 'tindices Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = data.output_type
    ; inputs = [ P data; P segment_ids ]
    ; attributes = []
    }

let selfAdjointEig
    ?(name = "SelfAdjointEig")
    (input : ([< `double | `float ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let sigmoid
    ?(name = "Sigmoid")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let sign
    ?(name = "Sign")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let sin
    ?(name = "Sin")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let slice
    ?(name = "Slice")
    (input : 't Node.t)
    (begin : 'index Node.t)
    (size : 'index Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input; P begin; P size ]
    ; attributes = []
    }

let softmax
    ?(name = "Softmax")
    (logits : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = logits.output_type
    ; inputs = [ P logits ]
    ; attributes = []
    }

let softplus
    ?(name = "Softplus")
    (features : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = features.output_type
    ; inputs = [ P features ]
    ; attributes = []
    }

let softplusGrad
    ?(name = "SoftplusGrad")
    (gradients : ([< `float | `double ] as 't) Node.t)
    (features : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = gradients.output_type
    ; inputs = [ P gradients; P features ]
    ; attributes = []
    }

let softsign
    ?(name = "Softsign")
    (features : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = features.output_type
    ; inputs = [ P features ]
    ; attributes = []
    }

let softsignGrad
    ?(name = "SoftsignGrad")
    (gradients : ([< `float | `double ] as 't) Node.t)
    (features : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = gradients.output_type
    ; inputs = [ P gradients; P features ]
    ; attributes = []
    }

let spaceToDepth
    ?(name = "SpaceToDepth")
    (input : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let sparseApplyAdagrad
    ?(name = "SparseApplyAdagrad")
    (var : ([< `float | `double ] as 't) Node.t)
    (accum : ([< `float | `double ] as 't) Node.t)
    (lr : ([< `float | `double ] as 't) Node.t)
    (grad : ([< `float | `double ] as 't) Node.t)
    (indices : 'tindices Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = var.output_type
    ; inputs = [ P var; P accum; P lr; P grad; P indices ]
    ; attributes = []
    }

let sparseApplyFtrl
    ?(name = "SparseApplyFtrl")
    (var : ([< `float | `double ] as 't) Node.t)
    (accum : ([< `float | `double ] as 't) Node.t)
    (linear : ([< `float | `double ] as 't) Node.t)
    (grad : ([< `float | `double ] as 't) Node.t)
    (indices : 'tindices Node.t)
    (lr : ([< `float | `double ] as 't) Node.t)
    (l1 : ([< `float | `double ] as 't) Node.t)
    (l2 : ([< `float | `double ] as 't) Node.t)
    (lr_power : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = var.output_type
    ; inputs = [ P var; P accum; P linear; P grad; P indices; P lr; P l1; P l2; P lr_power ]
    ; attributes = []
    }

let sparseApplyMomentum
    ?(name = "SparseApplyMomentum")
    (var : ([< `float | `double ] as 't) Node.t)
    (accum : ([< `float | `double ] as 't) Node.t)
    (lr : ([< `float | `double ] as 't) Node.t)
    (grad : ([< `float | `double ] as 't) Node.t)
    (indices : 'tindices Node.t)
    (momentum : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = var.output_type
    ; inputs = [ P var; P accum; P lr; P grad; P indices; P momentum ]
    ; attributes = []
    }

let sparseMatMul
    ?(name = "SparseMatMul")
    (a : [ `float ] Node.t)
    (b : [ `float ] Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = Type.(P Float)
    ; inputs = [ P a; P b ]
    ; attributes = []
    }

let sparseToDense
    ?(name = "SparseToDense")
    (sparse_indices : 'tindices Node.t)
    (output_shape : 'tindices Node.t)
    (sparse_values : 't Node.t)
    (default_value : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = sparse_values.output_type
    ; inputs = [ P sparse_indices; P output_shape; P sparse_values; P default_value ]
    ; attributes = []
    }

let sqrt
    ?(name = "Sqrt")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let square
    ?(name = "Square")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let squaredDifference
    ?(name = "SquaredDifference")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let squeeze
    ?(name = "Squeeze")
    (input : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let stopGradient
    ?(name = "StopGradient")
    (input : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = input.output_type
    ; inputs = [ P input ]
    ; attributes = []
    }

let sub
    ?(name = "Sub")
    (x : ([< `float | `double ] as 't) Node.t)
    (y : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x; P y ]
    ; attributes = []
    }

let tanh
    ?(name = "Tanh")
    (x : ([< `float | `double ] as 't) Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

let temporaryVariable
    ?(name = "TemporaryVariable")
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = TODO: add a parameter
    ; inputs = [  ]
    ; attributes = []
    }

let truncatedNormal
    ?(name = "TruncatedNormal")
    (shape : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = TODO: add a parameter
    ; inputs = [ P shape ]
    ; attributes = []
    }

let unpack
    ?(name = "Unpack")
    (value : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = value.output_type
    ; inputs = [ P value ]
    ; attributes = []
    }

let variable
    ?(name = "Variable")
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = TODO: add a parameter
    ; inputs = [  ]
    ; attributes = []
    }

let zerosLike
    ?(name = "ZerosLike")
    (x : 't Node.t)
  =
  Node
    { name = Name.make_fresh ~name
    ; output_type = x.output_type
    ; inputs = [ P x ]
    ; attributes = []
    }

