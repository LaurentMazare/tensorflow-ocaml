val abs
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val add
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val addN
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val adjustContrast
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val adjustContrastv2
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val applyAdagrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val applyAdam
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val applyFtrl
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val applyGradientDescent
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val applyMomentum
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val applyRMSProp
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val assign
  :  ?name:string
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

val assignAdd
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val assignSub
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val avgPool
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchCholesky
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

val batchMatMul
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchMatrixDeterminant
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchMatrixInverse
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchMatrixSolve
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchMatrixSolveLs
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> [ `double ] Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchMatrixTriangularSolve
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchNormWithGlobalNormalization
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchSelfAdjointEig
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

val biasAdd
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val biasAddGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val biasAddV1
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val bitcast
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'type__) Node.Type.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 'type__) Node.t

val cast
  :  ?name:string
  -> type_ : 'dstT Node.Type.t
  -> 'srcT Node.t
  -> 'dstT Node.t

val ceil
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val checkNumerics
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val cholesky
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

val const
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> 'dtype Node.t

val controlTrigger
  :  ?name:string
  -> [ `unit ] Node.t

val conv2D
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val cos
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val countUpTo
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val cross
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val depthToSpace
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val depthwiseConv2dNative
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val destroyTemporaryVariable
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val diag
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val diagPart
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val digamma
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val div
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val drawBoundingBoxes
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val elu
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val eluGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val enter
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val erf
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val erfc
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val exit
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val exp
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val floor
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val gather
  :  ?name:string
  -> 'tparams Node.t
  -> 'tindices Node.t
  -> 'tparams Node.t

val hSVToRGB
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val identity
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val inv
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val l2Loss
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val lRN
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val lRNGrad
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val lgamma
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val log
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val matMul
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val matrixDeterminant
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val matrixInverse
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val matrixSolve
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val matrixSolveLs
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> [ `double ] Node.t
  -> ([< `float | `double ] as 't) Node.t

val matrixTriangularSolve
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val maxPool
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val maxPoolGrad
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val maxPoolGradWithArgmax
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> 'targmax Node.t
  -> [ `float ] Node.t

val maximum
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val minimum
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val mod_
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val mul
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val neg
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val nextIteration
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val noOp
  :  ?name:string
  -> [ `unit ] Node.t

val pack
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val placeholder
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> 'dtype Node.t

val pow
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val rGBToHSV
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val randomShuffle
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val randomStandardNormal
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'dtype) Node.Type.t
  -> 't Node.t
  -> ([< `float | `double ] as 'dtype) Node.t

val randomUniform
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'dtype) Node.Type.t
  -> 't Node.t
  -> ([< `float | `double ] as 'dtype) Node.t

val randomUniformInt
  :  ?name:string
  -> 't Node.t
  -> 'tout Node.t
  -> 'tout Node.t
  -> 'tout Node.t

val refEnter
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val refExit
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val refIdentity
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val refNextIteration
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val relu
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val relu6
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val relu6Grad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val reluGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val resizeBilinearGrad
  :  ?name:string
  -> [ `float ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val rsqrt
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val scatterAdd
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val scatterSub
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val scatterUpdate
  :  ?name:string
  -> 't Node.t
  -> 'tindices Node.t
  -> 't Node.t
  -> 't Node.t

val segmentMax
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t

val segmentMean
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t

val segmentMin
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t

val segmentProd
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t

val segmentSum
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t

val selfAdjointEig
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

val sigmoid
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val sign
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val sin
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val slice
  :  ?name:string
  -> 't Node.t
  -> 'index Node.t
  -> 'index Node.t
  -> 't Node.t

val softmax
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val softplus
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val softplusGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val softsign
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val softsignGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val spaceToDepth
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val sparseApplyAdagrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t

val sparseApplyFtrl
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val sparseApplyMomentum
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> 'tindices Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val sparseMatMul
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val sparseToDense
  :  ?name:string
  -> 'tindices Node.t
  -> 'tindices Node.t
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

val sqrt
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val square
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val squaredDifference
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val squeeze
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val stopGradient
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val sub
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val tanh
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val temporaryVariable
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> 'dtype Node.t

val truncatedNormal
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'dtype) Node.Type.t
  -> 't Node.t
  -> ([< `float | `double ] as 'dtype) Node.t

val unpack
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val variable
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> 'dtype Node.t

val zerosLike
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

