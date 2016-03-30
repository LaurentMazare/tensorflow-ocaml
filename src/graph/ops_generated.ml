(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)
open Node

module Op_names = struct
  let abs = Op_name.of_string "Abs"
  let add = Op_name.of_string "Add"
  let addN = Op_name.of_string "AddN"
  let adjustContrast = Op_name.of_string "AdjustContrast"
  let adjustContrastv2 = Op_name.of_string "AdjustContrastv2"
  let all = Op_name.of_string "All"
  let any = Op_name.of_string "Any"
  let applyAdagrad = Op_name.of_string "ApplyAdagrad"
  let applyAdam = Op_name.of_string "ApplyAdam"
  let applyGradientDescent = Op_name.of_string "ApplyGradientDescent"
  let applyMomentum = Op_name.of_string "ApplyMomentum"
  let applyRMSProp = Op_name.of_string "ApplyRMSProp"
  let argMax = Op_name.of_string "ArgMax"
  let argMin = Op_name.of_string "ArgMin"
  let assign = Op_name.of_string "Assign"
  let assignAdd = Op_name.of_string "AssignAdd"
  let assignSub = Op_name.of_string "AssignSub"
  let avgPool = Op_name.of_string "AvgPool"
  let avgPoolGrad = Op_name.of_string "AvgPoolGrad"
  let batchCholesky = Op_name.of_string "BatchCholesky"
  let batchMatMul = Op_name.of_string "BatchMatMul"
  let batchMatrixDeterminant = Op_name.of_string "BatchMatrixDeterminant"
  let batchMatrixInverse = Op_name.of_string "BatchMatrixInverse"
  let batchMatrixSolve = Op_name.of_string "BatchMatrixSolve"
  let batchMatrixSolveLs = Op_name.of_string "BatchMatrixSolveLs"
  let batchMatrixTriangularSolve = Op_name.of_string "BatchMatrixTriangularSolve"
  let batchNormWithGlobalNormalization = Op_name.of_string "BatchNormWithGlobalNormalization"
  let batchSelfAdjointEig = Op_name.of_string "BatchSelfAdjointEig"
  let biasAdd = Op_name.of_string "BiasAdd"
  let cast = Op_name.of_string "Cast"
  let ceil = Op_name.of_string "Ceil"
  let checkNumerics = Op_name.of_string "CheckNumerics"
  let cholesky = Op_name.of_string "Cholesky"
  let complex = Op_name.of_string "Complex"
  let complexAbs = Op_name.of_string "ComplexAbs"
  let concat = Op_name.of_string "Concat"
  let concatOffset = Op_name.of_string "ConcatOffset"
  let conj = Op_name.of_string "Conj"
  let controlTrigger = Op_name.of_string "ControlTrigger"
  let conv2D = Op_name.of_string "Conv2D"
  let conv2DBackpropFilter = Op_name.of_string "Conv2DBackpropFilter"
  let conv2DBackpropInput = Op_name.of_string "Conv2DBackpropInput"
  let cos = Op_name.of_string "Cos"
  let countUpTo = Op_name.of_string "CountUpTo"
  let cross = Op_name.of_string "Cross"
  let decodeJSONExample = Op_name.of_string "DecodeJSONExample"
  let decodePng = Op_name.of_string "DecodePng"
  let decodeRaw = Op_name.of_string "DecodeRaw"
  let depthToSpace = Op_name.of_string "DepthToSpace"
  let destroyTemporaryVariable = Op_name.of_string "DestroyTemporaryVariable"
  let diag = Op_name.of_string "Diag"
  let div = Op_name.of_string "Div"
  let drawBoundingBoxes = Op_name.of_string "DrawBoundingBoxes"
  let dynamicPartition = Op_name.of_string "DynamicPartition"
  let dynamicStitch = Op_name.of_string "DynamicStitch"
  let editDistance = Op_name.of_string "EditDistance"
  let elu = Op_name.of_string "Elu"
  let eluGrad = Op_name.of_string "EluGrad"
  let encodePng = Op_name.of_string "EncodePng"
  let enter = Op_name.of_string "Enter"
  let equal = Op_name.of_string "Equal"
  let erf = Op_name.of_string "Erf"
  let erfc = Op_name.of_string "Erfc"
  let exit = Op_name.of_string "Exit"
  let exp = Op_name.of_string "Exp"
  let expandDims = Op_name.of_string "ExpandDims"
  let extractGlimpse = Op_name.of_string "ExtractGlimpse"
  let fFT2D = Op_name.of_string "FFT2D"
  let fIFOQueue = Op_name.of_string "FIFOQueue"
  let fact = Op_name.of_string "Fact"
  let fill = Op_name.of_string "Fill"
  let fixedLengthRecordReader = Op_name.of_string "FixedLengthRecordReader"
  let floor = Op_name.of_string "Floor"
  let gather = Op_name.of_string "Gather"
  let greater = Op_name.of_string "Greater"
  let greaterEqual = Op_name.of_string "GreaterEqual"
  let hSVToRGB = Op_name.of_string "HSVToRGB"
  let hashTable = Op_name.of_string "HashTable"
  let histogramSummary = Op_name.of_string "HistogramSummary"
  let iFFT2D = Op_name.of_string "IFFT2D"
  let identity = Op_name.of_string "Identity"
  let identityReader = Op_name.of_string "IdentityReader"
  let imag = Op_name.of_string "Imag"
  let imageSummary = Op_name.of_string "ImageSummary"
  let inTopK = Op_name.of_string "InTopK"
  let initializeTable = Op_name.of_string "InitializeTable"
  let inv = Op_name.of_string "Inv"
  let invertPermutation = Op_name.of_string "InvertPermutation"
  let isFinite = Op_name.of_string "IsFinite"
  let isInf = Op_name.of_string "IsInf"
  let isNan = Op_name.of_string "IsNan"
  let l2Loss = Op_name.of_string "L2Loss"
  let lRN = Op_name.of_string "LRN"
  let lRNGrad = Op_name.of_string "LRNGrad"
  let less = Op_name.of_string "Less"
  let lessEqual = Op_name.of_string "LessEqual"
  let lgamma = Op_name.of_string "Lgamma"
  let linSpace = Op_name.of_string "LinSpace"
  let log = Op_name.of_string "Log"
  let logicalAnd = Op_name.of_string "LogicalAnd"
  let logicalNot = Op_name.of_string "LogicalNot"
  let logicalOr = Op_name.of_string "LogicalOr"
  let lookupTableFind = Op_name.of_string "LookupTableFind"
  let lookupTableSize = Op_name.of_string "LookupTableSize"
  let loopCond = Op_name.of_string "LoopCond"
  let matMul = Op_name.of_string "MatMul"
  let matchingFiles = Op_name.of_string "MatchingFiles"
  let matrixDeterminant = Op_name.of_string "MatrixDeterminant"
  let matrixInverse = Op_name.of_string "MatrixInverse"
  let matrixSolve = Op_name.of_string "MatrixSolve"
  let matrixSolveLs = Op_name.of_string "MatrixSolveLs"
  let matrixTriangularSolve = Op_name.of_string "MatrixTriangularSolve"
  let max = Op_name.of_string "Max"
  let maxPool = Op_name.of_string "MaxPool"
  let maxPoolGrad = Op_name.of_string "MaxPoolGrad"
  let maxPoolGradWithArgmax = Op_name.of_string "MaxPoolGradWithArgmax"
  let maximum = Op_name.of_string "Maximum"
  let mean = Op_name.of_string "Mean"
  let mergeSummary = Op_name.of_string "MergeSummary"
  let min = Op_name.of_string "Min"
  let minimum = Op_name.of_string "Minimum"
  let mod_ = Op_name.of_string "Mod"
  let mul = Op_name.of_string "Mul"
  let neg = Op_name.of_string "Neg"
  let negTrain = Op_name.of_string "NegTrain"
  let nextIteration = Op_name.of_string "NextIteration"
  let noOp = Op_name.of_string "NoOp"
  let notEqual = Op_name.of_string "NotEqual"
  let pack = Op_name.of_string "Pack"
  let pad = Op_name.of_string "Pad"
  let paddingFIFOQueue = Op_name.of_string "PaddingFIFOQueue"
  let placeholder = Op_name.of_string "Placeholder"
  let pow = Op_name.of_string "Pow"
  let prod = Op_name.of_string "Prod"
  let queueClose = Op_name.of_string "QueueClose"
  let queueSize = Op_name.of_string "QueueSize"
  let rGBToHSV = Op_name.of_string "RGBToHSV"
  let randomCrop = Op_name.of_string "RandomCrop"
  let randomShuffle = Op_name.of_string "RandomShuffle"
  let randomShuffleQueue = Op_name.of_string "RandomShuffleQueue"
  let randomStandardNormal = Op_name.of_string "RandomStandardNormal"
  let randomUniform = Op_name.of_string "RandomUniform"
  let randomUniformInt = Op_name.of_string "RandomUniformInt"
  let range = Op_name.of_string "Range"
  let rank = Op_name.of_string "Rank"
  let readFile = Op_name.of_string "ReadFile"
  let readerNumRecordsProduced = Op_name.of_string "ReaderNumRecordsProduced"
  let readerNumWorkUnitsCompleted = Op_name.of_string "ReaderNumWorkUnitsCompleted"
  let readerReset = Op_name.of_string "ReaderReset"
  let readerRestoreState = Op_name.of_string "ReaderRestoreState"
  let readerSerializeState = Op_name.of_string "ReaderSerializeState"
  let real = Op_name.of_string "Real"
  let refEnter = Op_name.of_string "RefEnter"
  let refExit = Op_name.of_string "RefExit"
  let refIdentity = Op_name.of_string "RefIdentity"
  let refNextIteration = Op_name.of_string "RefNextIteration"
  let refSelect = Op_name.of_string "RefSelect"
  let relu = Op_name.of_string "Relu"
  let relu6 = Op_name.of_string "Relu6"
  let relu6Grad = Op_name.of_string "Relu6Grad"
  let reluGrad = Op_name.of_string "ReluGrad"
  let reshape = Op_name.of_string "Reshape"
  let resizeArea = Op_name.of_string "ResizeArea"
  let resizeBicubic = Op_name.of_string "ResizeBicubic"
  let resizeBilinear = Op_name.of_string "ResizeBilinear"
  let resizeBilinearGrad = Op_name.of_string "ResizeBilinearGrad"
  let resizeNearestNeighbor = Op_name.of_string "ResizeNearestNeighbor"
  let resizeNearestNeighborGrad = Op_name.of_string "ResizeNearestNeighborGrad"
  let restore = Op_name.of_string "Restore"
  let restoreSlice = Op_name.of_string "RestoreSlice"
  let reverse = Op_name.of_string "Reverse"
  let reverseSequence = Op_name.of_string "ReverseSequence"
  let rsqrt = Op_name.of_string "Rsqrt"
  let scalarSummary = Op_name.of_string "ScalarSummary"
  let scatterAdd = Op_name.of_string "ScatterAdd"
  let scatterSub = Op_name.of_string "ScatterSub"
  let scatterUpdate = Op_name.of_string "ScatterUpdate"
  let segmentMax = Op_name.of_string "SegmentMax"
  let segmentMean = Op_name.of_string "SegmentMean"
  let segmentMin = Op_name.of_string "SegmentMin"
  let segmentProd = Op_name.of_string "SegmentProd"
  let segmentSum = Op_name.of_string "SegmentSum"
  let select = Op_name.of_string "Select"
  let selfAdjointEig = Op_name.of_string "SelfAdjointEig"
  let serializeManySparse = Op_name.of_string "SerializeManySparse"
  let serializeSparse = Op_name.of_string "SerializeSparse"
  let shape = Op_name.of_string "Shape"
  let shapeN = Op_name.of_string "ShapeN"
  let shardedFilename = Op_name.of_string "ShardedFilename"
  let shardedFilespec = Op_name.of_string "ShardedFilespec"
  let sigmoid = Op_name.of_string "Sigmoid"
  let sign = Op_name.of_string "Sign"
  let sin = Op_name.of_string "Sin"
  let size = Op_name.of_string "Size"
  let slice = Op_name.of_string "Slice"
  let softmax = Op_name.of_string "Softmax"
  let softplus = Op_name.of_string "Softplus"
  let softplusGrad = Op_name.of_string "SoftplusGrad"
  let softsign = Op_name.of_string "Softsign"
  let softsignGrad = Op_name.of_string "SoftsignGrad"
  let spaceToDepth = Op_name.of_string "SpaceToDepth"
  let sparseApplyAdagrad = Op_name.of_string "SparseApplyAdagrad"
  let sparseApplyMomentum = Op_name.of_string "SparseApplyMomentum"
  let sparseMatMul = Op_name.of_string "SparseMatMul"
  let sparseSegmentMean = Op_name.of_string "SparseSegmentMean"
  let sparseSegmentMeanGrad = Op_name.of_string "SparseSegmentMeanGrad"
  let sparseSegmentSqrtN = Op_name.of_string "SparseSegmentSqrtN"
  let sparseSegmentSqrtNGrad = Op_name.of_string "SparseSegmentSqrtNGrad"
  let sparseSegmentSum = Op_name.of_string "SparseSegmentSum"
  let sparseToDense = Op_name.of_string "SparseToDense"
  let split = Op_name.of_string "Split"
  let sqrt = Op_name.of_string "Sqrt"
  let square = Op_name.of_string "Square"
  let squeeze = Op_name.of_string "Squeeze"
  let stack = Op_name.of_string "Stack"
  let stackClose = Op_name.of_string "StackClose"
  let stackPop = Op_name.of_string "StackPop"
  let stackPush = Op_name.of_string "StackPush"
  let stopGradient = Op_name.of_string "StopGradient"
  let stringToHashBucket = Op_name.of_string "StringToHashBucket"
  let stringToNumber = Op_name.of_string "StringToNumber"
  let sub = Op_name.of_string "Sub"
  let sum = Op_name.of_string "Sum"
  let tFRecordReader = Op_name.of_string "TFRecordReader"
  let tanh = Op_name.of_string "Tanh"
  let temporaryVariable = Op_name.of_string "TemporaryVariable"
  let tensorArray = Op_name.of_string "TensorArray"
  let tensorArrayClose = Op_name.of_string "TensorArrayClose"
  let tensorArrayGrad = Op_name.of_string "TensorArrayGrad"
  let tensorArrayPack = Op_name.of_string "TensorArrayPack"
  let tensorArrayRead = Op_name.of_string "TensorArrayRead"
  let tensorArraySize = Op_name.of_string "TensorArraySize"
  let tensorArrayUnpack = Op_name.of_string "TensorArrayUnpack"
  let tensorArrayWrite = Op_name.of_string "TensorArrayWrite"
  let textLineReader = Op_name.of_string "TextLineReader"
  let tile = Op_name.of_string "Tile"
  let tileGrad = Op_name.of_string "TileGrad"
  let transpose = Op_name.of_string "Transpose"
  let truncatedNormal = Op_name.of_string "TruncatedNormal"
  let unpack = Op_name.of_string "Unpack"
  let unsortedSegmentSum = Op_name.of_string "UnsortedSegmentSum"
  let variable = Op_name.of_string "Variable"
  let where = Op_name.of_string "Where"
  let wholeFileReader = Op_name.of_string "WholeFileReader"
  let zerosLike = Op_name.of_string "ZerosLike"
end

let abs
    ?(name = "Abs")
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P x.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_names.abs
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
  ; op_name = Op_names.add
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
  ; op_name = Op_names.addN
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
  ; op_name = Op_names.adjustContrast
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
  ; op_name = Op_names.adjustContrastv2
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
  ; op_name = Op_names.all
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
  ; op_name = Op_names.any
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
  ; op_name = Op_names.applyAdagrad
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
  ; op_name = Op_names.applyAdam
  ; output_type = var.output_type
  ; inputs = [ P var; P m; P v; P beta1_power; P beta2_power; P lr; P beta1; P beta2; P epsilon; P grad ]
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
  ; op_name = Op_names.applyGradientDescent
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
  ; op_name = Op_names.applyMomentum
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
  ; op_name = Op_names.applyRMSProp
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
  ; op_name = Op_names.argMax
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
  ; op_name = Op_names.argMin
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
  ; op_name = Op_names.assign
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
  ; op_name = Op_names.assignAdd
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
  ; op_name = Op_names.assignSub
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
  { name = Name.make_fresh ~name
  ; op_name = Op_names.avgPool
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
  { name = Name.make_fresh ~name
  ; op_name = Op_names.avgPoolGrad
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
  ; op_name = Op_names.batchCholesky
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
  ; op_name = Op_names.batchMatMul
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
  ; op_name = Op_names.batchMatrixDeterminant
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
  ; op_name = Op_names.batchMatrixInverse
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
  ; op_name = Op_names.batchMatrixSolve
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
  ; op_name = Op_names.batchMatrixSolveLs
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
  ; op_name = Op_names.batchMatrixTriangularSolve
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
  ; op_name = Op_names.batchNormWithGlobalNormalization
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
  ; op_name = Op_names.batchSelfAdjointEig
  ; output_type = input.output_type
  ; inputs = [ P input ]
  ; attributes
  ; output_idx = None
  }

let biasAdd
    ?(name = "BiasAdd")
    (value : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
    (bias : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P value.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_names.biasAdd
  ; output_type = value.output_type
  ; inputs = [ P value; P bias ]
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
  ; op_name = Op_names.cast
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
  ; op_name = Op_names.ceil
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
  ; op_name = Op_names.checkNumerics
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
  ; op_name = Op_names.cholesky
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
  ; op_name = Op_names.complex
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
  ; op_name = Op_names.complexAbs
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
  ; op_name = Op_names.concat
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
  ; op_name = Op_names.concatOffset
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
  ; op_name = Op_names.conj
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
  ; op_name = Op_names.controlTrigger
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
  { name = Name.make_fresh ~name
  ; op_name = Op_names.conv2D
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
  { name = Name.make_fresh ~name
  ; op_name = Op_names.conv2DBackpropFilter
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
  { name = Name.make_fresh ~name
  ; op_name = Op_names.conv2DBackpropInput
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
  ; op_name = Op_names.cos
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
  ; op_name = Op_names.countUpTo
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
  ; op_name = Op_names.cross
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
  ; op_name = Op_names.decodeJSONExample
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
  ; op_name = Op_names.decodePng
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
  ; op_name = Op_names.decodeRaw
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
  ; op_name = Op_names.depthToSpace
  ; output_type = input.output_type
  ; inputs = [ P input ]
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
  ; op_name = Op_names.destroyTemporaryVariable
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
  ; op_name = Op_names.diag
  ; output_type = diagonal.output_type
  ; inputs = [ P diagonal ]
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
  ; op_name = Op_names.div
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
  ; op_name = Op_names.drawBoundingBoxes
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
  ; op_name = Op_names.dynamicPartition
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
  ; op_name = Op_names.dynamicStitch
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
  ; op_name = Op_names.editDistance
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
  ; op_name = Op_names.elu
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
  ; op_name = Op_names.eluGrad
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
  ; op_name = Op_names.encodePng
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
  ; op_name = Op_names.enter
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
  ; op_name = Op_names.equal
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
  ; op_name = Op_names.erf
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
  ; op_name = Op_names.erfc
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
  ; op_name = Op_names.exit
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
  ; op_name = Op_names.exp
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
  ; op_name = Op_names.expandDims
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
  ; op_name = Op_names.extractGlimpse
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
  ; op_name = Op_names.fFT2D
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
  ; op_name = Op_names.fIFOQueue
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
  ; op_name = Op_names.fact
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
  ; op_name = Op_names.fill
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
  ; op_name = Op_names.fixedLengthRecordReader
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
  ; op_name = Op_names.floor
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
  ; op_name = Op_names.gather
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
  ; op_name = Op_names.greater
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
  ; op_name = Op_names.greaterEqual
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
  ; op_name = Op_names.hSVToRGB
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
  ; op_name = Op_names.hashTable
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
  ; op_name = Op_names.histogramSummary
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
  ; op_name = Op_names.iFFT2D
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
  ; op_name = Op_names.identity
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
  ; op_name = Op_names.identityReader
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
  ; op_name = Op_names.imag
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
  ; op_name = Op_names.imageSummary
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
  ; op_name = Op_names.inTopK
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
  ; op_name = Op_names.initializeTable
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
  ; op_name = Op_names.inv
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
  ; op_name = Op_names.invertPermutation
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
  ; op_name = Op_names.isFinite
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
  ; op_name = Op_names.isInf
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
  ; op_name = Op_names.isNan
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
  ; op_name = Op_names.l2Loss
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
  ; op_name = Op_names.lRN
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
  ; op_name = Op_names.lRNGrad
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
  ; op_name = Op_names.less
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
  ; op_name = Op_names.lessEqual
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
  ; op_name = Op_names.lgamma
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
  ; op_name = Op_names.linSpace
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
  ; op_name = Op_names.log
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
  ; op_name = Op_names.logicalAnd
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
  ; op_name = Op_names.logicalNot
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
  ; op_name = Op_names.logicalOr
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
  ; op_name = Op_names.lookupTableFind
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
  ; op_name = Op_names.lookupTableSize
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
  ; op_name = Op_names.loopCond
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
  ; op_name = Op_names.matMul
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
  ; op_name = Op_names.matchingFiles
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
  ; op_name = Op_names.matrixDeterminant
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
  ; op_name = Op_names.matrixInverse
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
  ; op_name = Op_names.matrixSolve
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
  ; op_name = Op_names.matrixSolveLs
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
  ; op_name = Op_names.matrixTriangularSolve
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
  ; op_name = Op_names.max
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
  { name = Name.make_fresh ~name
  ; op_name = Op_names.maxPool
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
  { name = Name.make_fresh ~name
  ; op_name = Op_names.maxPoolGrad
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
  ; op_name = Op_names.maxPoolGradWithArgmax
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
  ; op_name = Op_names.maximum
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
  ; op_name = Op_names.mean
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
  ; op_name = Op_names.mergeSummary
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
  ; op_name = Op_names.min
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
  ; op_name = Op_names.minimum
  ; output_type = x.output_type
  ; inputs = [ P x; P y ]
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
  ; op_name = Op_names.mod_
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
  ; op_name = Op_names.mul
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
  ; op_name = Op_names.neg
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
  ; op_name = Op_names.negTrain
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
  ; op_name = Op_names.nextIteration
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
  ; op_name = Op_names.noOp
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
  ; op_name = Op_names.notEqual
  ; output_type = Type.Bool
  ; inputs = [ P x; P y ]
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
  ; op_name = Op_names.pack
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
  ; op_name = Op_names.pad
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
  ; op_name = Op_names.paddingFIFOQueue
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
  ; op_name = Op_names.placeholder
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
  ; op_name = Op_names.pow
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
  ; op_name = Op_names.prod
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
  ; op_name = Op_names.queueClose
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
  ; op_name = Op_names.queueSize
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
  ; op_name = Op_names.rGBToHSV
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
  ; op_name = Op_names.randomCrop
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
  ; op_name = Op_names.randomShuffle
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
  ; op_name = Op_names.randomShuffleQueue
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
  ; op_name = Op_names.randomStandardNormal
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
  ; op_name = Op_names.randomUniform
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
  ; op_name = Op_names.randomUniformInt
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
  ; op_name = Op_names.range
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
  ; op_name = Op_names.rank
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
  ; op_name = Op_names.readFile
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
  ; op_name = Op_names.readerNumRecordsProduced
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
  ; op_name = Op_names.readerNumWorkUnitsCompleted
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
  ; op_name = Op_names.readerReset
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
  ; op_name = Op_names.readerRestoreState
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
  ; op_name = Op_names.readerSerializeState
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
  ; op_name = Op_names.real
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
  ; op_name = Op_names.refEnter
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
  ; op_name = Op_names.refExit
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
  ; op_name = Op_names.refIdentity
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
  ; op_name = Op_names.refNextIteration
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
  ; op_name = Op_names.refSelect
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
  ; op_name = Op_names.relu
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
  ; op_name = Op_names.relu6
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
  ; op_name = Op_names.relu6Grad
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
  ; op_name = Op_names.reluGrad
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
  ; op_name = Op_names.reshape
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
  ; op_name = Op_names.resizeArea
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
  ; op_name = Op_names.resizeBicubic
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
  ; op_name = Op_names.resizeBilinear
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
  ; op_name = Op_names.resizeBilinearGrad
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
  ; op_name = Op_names.resizeNearestNeighbor
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
  ; op_name = Op_names.resizeNearestNeighborGrad
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
  ; op_name = Op_names.restore
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
  ; op_name = Op_names.restoreSlice
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
  ; op_name = Op_names.reverse
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
  ; op_name = Op_names.reverseSequence
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
  ; op_name = Op_names.rsqrt
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
  ; op_name = Op_names.scalarSummary
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
  ; op_name = Op_names.scatterAdd
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
  ; op_name = Op_names.scatterSub
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
  ; op_name = Op_names.scatterUpdate
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
  ; op_name = Op_names.segmentMax
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
  ; op_name = Op_names.segmentMean
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
  ; op_name = Op_names.segmentMin
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
  ; op_name = Op_names.segmentProd
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
  ; op_name = Op_names.segmentSum
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
  ; op_name = Op_names.select
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
  ; op_name = Op_names.selfAdjointEig
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
  ; op_name = Op_names.serializeManySparse
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
  ; op_name = Op_names.serializeSparse
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
  ; op_name = Op_names.shape
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
  ; op_name = Op_names.shapeN
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
  ; op_name = Op_names.shardedFilename
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
  ; op_name = Op_names.shardedFilespec
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
  ; op_name = Op_names.sigmoid
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
  ; op_name = Op_names.sign
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
  ; op_name = Op_names.sin
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
  ; op_name = Op_names.size
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
  ; op_name = Op_names.slice
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
  ; op_name = Op_names.softmax
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
  ; op_name = Op_names.softplus
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
  ; op_name = Op_names.softplusGrad
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
  ; op_name = Op_names.softsign
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
  ; op_name = Op_names.softsignGrad
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
  ; op_name = Op_names.spaceToDepth
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
  ; op_name = Op_names.sparseApplyAdagrad
  ; output_type = var.output_type
  ; inputs = [ P var; P accum; P lr; P grad; P indices ]
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
  ; op_name = Op_names.sparseApplyMomentum
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
  ; op_name = Op_names.sparseMatMul
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
  ; op_name = Op_names.sparseSegmentMean
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
  ; op_name = Op_names.sparseSegmentMeanGrad
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
  ; op_name = Op_names.sparseSegmentSqrtN
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
  ; op_name = Op_names.sparseSegmentSqrtNGrad
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
  ; op_name = Op_names.sparseSegmentSum
  ; output_type = data.output_type
  ; inputs = [ P data; P indices; P segment_ids ]
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
  ; op_name = Op_names.sparseToDense
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
  ; op_name = Op_names.split
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
  ; op_name = Op_names.sqrt
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
  ; op_name = Op_names.square
  ; output_type = x.output_type
  ; inputs = [ P x ]
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
  ; op_name = Op_names.squeeze
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
  ; op_name = Op_names.stack
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
  ; op_name = Op_names.stackClose
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
  ; op_name = Op_names.stackPop
  ; output_type = type_
  ; inputs = [ P handle ]
  ; attributes
  ; output_idx = None
  }

let stackPush
    ?(name = "StackPush")
    (handle : [ `string ] t)
    (elem : 't t)
  =
  let attributes = [ "T", Type (P elem.output_type) ] in
  { name = Name.make_fresh ~name
  ; op_name = Op_names.stackPush
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
  ; op_name = Op_names.stopGradient
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
  ; op_name = Op_names.stringToHashBucket
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
  ; op_name = Op_names.stringToNumber
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
  ; op_name = Op_names.sub
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
  ; op_name = Op_names.sum
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
  ; op_name = Op_names.tFRecordReader
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
  ; op_name = Op_names.tanh
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
  ; op_name = Op_names.temporaryVariable
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
  ; op_name = Op_names.tensorArray
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
  ; op_name = Op_names.tensorArrayClose
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
  ; op_name = Op_names.tensorArrayGrad
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
  ; op_name = Op_names.tensorArrayPack
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
  ; op_name = Op_names.tensorArrayRead
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
  ; op_name = Op_names.tensorArraySize
  ; output_type = Type.Int32
  ; inputs = [ P handle; P flow_in ]
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
  ; op_name = Op_names.tensorArrayUnpack
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
  ; op_name = Op_names.tensorArrayWrite
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
  ; op_name = Op_names.textLineReader
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
  ; op_name = Op_names.tile
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
  ; op_name = Op_names.tileGrad
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
  ; op_name = Op_names.transpose
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
  ; op_name = Op_names.truncatedNormal
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
  ; op_name = Op_names.unpack
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
  ; op_name = Op_names.unsortedSegmentSum
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
  ; op_name = Op_names.variable
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
  ; op_name = Op_names.where
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
  ; op_name = Op_names.wholeFileReader
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
  ; op_name = Op_names.zerosLike
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes
  ; output_idx = None
  }

