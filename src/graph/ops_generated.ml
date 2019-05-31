(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)
open Base
open Tensorflow_core.Operation
open Node

module Op_names = struct
  let abort = Op_name.of_string "Abort"
  let abs = Op_name.of_string "Abs"
  let accumulateNV2 = Op_name.of_string "AccumulateNV2"
  let accumulatorApplyGradient = Op_name.of_string "AccumulatorApplyGradient"
  let accumulatorNumAccumulated = Op_name.of_string "AccumulatorNumAccumulated"
  let accumulatorSetGlobalStep = Op_name.of_string "AccumulatorSetGlobalStep"
  let accumulatorTakeGradient = Op_name.of_string "AccumulatorTakeGradient"
  let acos = Op_name.of_string "Acos"
  let acosh = Op_name.of_string "Acosh"
  let add = Op_name.of_string "Add"
  let addManySparseToTensorsMap = Op_name.of_string "AddManySparseToTensorsMap"
  let addN = Op_name.of_string "AddN"
  let addSparseToTensorsMap = Op_name.of_string "AddSparseToTensorsMap"
  let addV2 = Op_name.of_string "AddV2"
  let adjustContrast = Op_name.of_string "AdjustContrast"
  let adjustContrastv2 = Op_name.of_string "AdjustContrastv2"
  let adjustHue = Op_name.of_string "AdjustHue"
  let adjustSaturation = Op_name.of_string "AdjustSaturation"
  let all = Op_name.of_string "All"
  let allCandidateSampler = Op_name.of_string "AllCandidateSampler"
  let allToAll = Op_name.of_string "AllToAll"
  let angle = Op_name.of_string "Angle"
  let any = Op_name.of_string "Any"
  let applyAdaMax = Op_name.of_string "ApplyAdaMax"
  let applyAdadelta = Op_name.of_string "ApplyAdadelta"
  let applyAdagrad = Op_name.of_string "ApplyAdagrad"
  let applyAdagradDA = Op_name.of_string "ApplyAdagradDA"
  let applyAdam = Op_name.of_string "ApplyAdam"
  let applyAddSign = Op_name.of_string "ApplyAddSign"
  let applyCenteredRMSProp = Op_name.of_string "ApplyCenteredRMSProp"
  let applyFtrl = Op_name.of_string "ApplyFtrl"
  let applyFtrlV2 = Op_name.of_string "ApplyFtrlV2"
  let applyGradientDescent = Op_name.of_string "ApplyGradientDescent"
  let applyMomentum = Op_name.of_string "ApplyMomentum"
  let applyPowerSign = Op_name.of_string "ApplyPowerSign"
  let applyProximalAdagrad = Op_name.of_string "ApplyProximalAdagrad"
  let applyProximalGradientDescent = Op_name.of_string "ApplyProximalGradientDescent"
  let applyRMSProp = Op_name.of_string "ApplyRMSProp"
  let approximateEqual = Op_name.of_string "ApproximateEqual"
  let argMax = Op_name.of_string "ArgMax"
  let argMin = Op_name.of_string "ArgMin"
  let asString = Op_name.of_string "AsString"
  let asin = Op_name.of_string "Asin"
  let asinh = Op_name.of_string "Asinh"
  let assign = Op_name.of_string "Assign"
  let assignAdd = Op_name.of_string "AssignAdd"
  let assignSub = Op_name.of_string "AssignSub"
  let atan = Op_name.of_string "Atan"
  let atan2 = Op_name.of_string "Atan2"
  let atanh = Op_name.of_string "Atanh"
  let audioSpectrogram = Op_name.of_string "AudioSpectrogram"
  let audioSummary = Op_name.of_string "AudioSummary"
  let audioSummaryV2 = Op_name.of_string "AudioSummaryV2"
  let avgPool = Op_name.of_string "AvgPool"
  let avgPool3D = Op_name.of_string "AvgPool3D"
  let avgPool3DGrad = Op_name.of_string "AvgPool3DGrad"
  let avgPoolGrad = Op_name.of_string "AvgPoolGrad"
  let barrier = Op_name.of_string "Barrier"
  let barrierClose = Op_name.of_string "BarrierClose"
  let barrierIncompleteSize = Op_name.of_string "BarrierIncompleteSize"
  let barrierInsertMany = Op_name.of_string "BarrierInsertMany"
  let barrierReadySize = Op_name.of_string "BarrierReadySize"
  let batchCholesky = Op_name.of_string "BatchCholesky"
  let batchCholeskyGrad = Op_name.of_string "BatchCholeskyGrad"
  let batchDataset = Op_name.of_string "BatchDataset"
  let batchDatasetV2 = Op_name.of_string "BatchDatasetV2"
  let batchFFT = Op_name.of_string "BatchFFT"
  let batchFFT2D = Op_name.of_string "BatchFFT2D"
  let batchFFT3D = Op_name.of_string "BatchFFT3D"
  let batchIFFT = Op_name.of_string "BatchIFFT"
  let batchIFFT2D = Op_name.of_string "BatchIFFT2D"
  let batchIFFT3D = Op_name.of_string "BatchIFFT3D"
  let batchMatMul = Op_name.of_string "BatchMatMul"
  let batchMatMulV2 = Op_name.of_string "BatchMatMulV2"
  let batchMatrixBandPart = Op_name.of_string "BatchMatrixBandPart"
  let batchMatrixDeterminant = Op_name.of_string "BatchMatrixDeterminant"
  let batchMatrixDiag = Op_name.of_string "BatchMatrixDiag"
  let batchMatrixDiagPart = Op_name.of_string "BatchMatrixDiagPart"
  let batchMatrixInverse = Op_name.of_string "BatchMatrixInverse"
  let batchMatrixSetDiag = Op_name.of_string "BatchMatrixSetDiag"
  let batchMatrixSolve = Op_name.of_string "BatchMatrixSolve"
  let batchMatrixSolveLs = Op_name.of_string "BatchMatrixSolveLs"
  let batchMatrixTriangularSolve = Op_name.of_string "BatchMatrixTriangularSolve"
  let batchNormWithGlobalNormalization = Op_name.of_string "BatchNormWithGlobalNormalization"
  let batchNormWithGlobalNormalizationGrad = Op_name.of_string "BatchNormWithGlobalNormalizationGrad"
  let batchSelfAdjointEig = Op_name.of_string "BatchSelfAdjointEig"
  let batchSelfAdjointEigV2 = Op_name.of_string "BatchSelfAdjointEigV2"
  let batchSvd = Op_name.of_string "BatchSvd"
  let batchToSpace = Op_name.of_string "BatchToSpace"
  let batchToSpaceND = Op_name.of_string "BatchToSpaceND"
  let besselI0e = Op_name.of_string "BesselI0e"
  let besselI1e = Op_name.of_string "BesselI1e"
  let betainc = Op_name.of_string "Betainc"
  let biasAdd = Op_name.of_string "BiasAdd"
  let biasAddGrad = Op_name.of_string "BiasAddGrad"
  let biasAddV1 = Op_name.of_string "BiasAddV1"
  let bincount = Op_name.of_string "Bincount"
  let bitcast = Op_name.of_string "Bitcast"
  let bitwiseAnd = Op_name.of_string "BitwiseAnd"
  let bitwiseOr = Op_name.of_string "BitwiseOr"
  let bitwiseXor = Op_name.of_string "BitwiseXor"
  let boostedTreesAggregateStats = Op_name.of_string "BoostedTreesAggregateStats"
  let boostedTreesBucketize = Op_name.of_string "BoostedTreesBucketize"
  let boostedTreesCalculateBestFeatureSplit = Op_name.of_string "BoostedTreesCalculateBestFeatureSplit"
  let boostedTreesMakeQuantileSummaries = Op_name.of_string "BoostedTreesMakeQuantileSummaries"
  let boostedTreesMakeStatsSummary = Op_name.of_string "BoostedTreesMakeStatsSummary"
  let broadcastArgs = Op_name.of_string "BroadcastArgs"
  let broadcastGradientArgs = Op_name.of_string "BroadcastGradientArgs"
  let broadcastTo = Op_name.of_string "BroadcastTo"
  let bucketize = Op_name.of_string "Bucketize"
  let cTCGreedyDecoder = Op_name.of_string "CTCGreedyDecoder"
  let cTCLoss = Op_name.of_string "CTCLoss"
  let cacheDataset = Op_name.of_string "CacheDataset"
  let cast = Op_name.of_string "Cast"
  let ceil = Op_name.of_string "Ceil"
  let checkNumerics = Op_name.of_string "CheckNumerics"
  let cholesky = Op_name.of_string "Cholesky"
  let choleskyGrad = Op_name.of_string "CholeskyGrad"
  let clipByValue = Op_name.of_string "ClipByValue"
  let collectiveBcastRecv = Op_name.of_string "CollectiveBcastRecv"
  let collectiveBcastSend = Op_name.of_string "CollectiveBcastSend"
  let collectiveGather = Op_name.of_string "CollectiveGather"
  let collectivePermute = Op_name.of_string "CollectivePermute"
  let collectiveReduce = Op_name.of_string "CollectiveReduce"
  let combinedNonMaxSuppression = Op_name.of_string "CombinedNonMaxSuppression"
  let complex = Op_name.of_string "Complex"
  let complexAbs = Op_name.of_string "ComplexAbs"
  let computeAccidentalHits = Op_name.of_string "ComputeAccidentalHits"
  let concat = Op_name.of_string "Concat"
  let concatOffset = Op_name.of_string "ConcatOffset"
  let concatV2 = Op_name.of_string "ConcatV2"
  let concatenateDataset = Op_name.of_string "ConcatenateDataset"
  let conditionalAccumulator = Op_name.of_string "ConditionalAccumulator"
  let configureDistributedTPU = Op_name.of_string "ConfigureDistributedTPU"
  let conj = Op_name.of_string "Conj"
  let conjugateTranspose = Op_name.of_string "ConjugateTranspose"
  let consumeMutexLock = Op_name.of_string "ConsumeMutexLock"
  let controlTrigger = Op_name.of_string "ControlTrigger"
  let conv2D = Op_name.of_string "Conv2D"
  let conv2DBackpropFilter = Op_name.of_string "Conv2DBackpropFilter"
  let conv2DBackpropInput = Op_name.of_string "Conv2DBackpropInput"
  let conv3D = Op_name.of_string "Conv3D"
  let conv3DBackpropFilter = Op_name.of_string "Conv3DBackpropFilter"
  let conv3DBackpropFilterV2 = Op_name.of_string "Conv3DBackpropFilterV2"
  let conv3DBackpropInput = Op_name.of_string "Conv3DBackpropInput"
  let conv3DBackpropInputV2 = Op_name.of_string "Conv3DBackpropInputV2"
  let copy = Op_name.of_string "Copy"
  let copyHost = Op_name.of_string "CopyHost"
  let cos = Op_name.of_string "Cos"
  let cosh = Op_name.of_string "Cosh"
  let countUpTo = Op_name.of_string "CountUpTo"
  let cropAndResize = Op_name.of_string "CropAndResize"
  let cropAndResizeGradBoxes = Op_name.of_string "CropAndResizeGradBoxes"
  let cropAndResizeGradImage = Op_name.of_string "CropAndResizeGradImage"
  let cross = Op_name.of_string "Cross"
  let crossReplicaSum = Op_name.of_string "CrossReplicaSum"
  let cudnnRNN = Op_name.of_string "CudnnRNN"
  let cudnnRNNBackprop = Op_name.of_string "CudnnRNNBackprop"
  let cudnnRNNCanonicalToParams = Op_name.of_string "CudnnRNNCanonicalToParams"
  let cudnnRNNParamsSize = Op_name.of_string "CudnnRNNParamsSize"
  let cumprod = Op_name.of_string "Cumprod"
  let cumsum = Op_name.of_string "Cumsum"
  let dataFormatDimMap = Op_name.of_string "DataFormatDimMap"
  let dataFormatVecPermute = Op_name.of_string "DataFormatVecPermute"
  let datasetToGraph = Op_name.of_string "DatasetToGraph"
  let debugGradientIdentity = Op_name.of_string "DebugGradientIdentity"
  let debugGradientRefIdentity = Op_name.of_string "DebugGradientRefIdentity"
  let debugIdentity = Op_name.of_string "DebugIdentity"
  let debugNanCount = Op_name.of_string "DebugNanCount"
  let debugNumericSummary = Op_name.of_string "DebugNumericSummary"
  let decodeBase64 = Op_name.of_string "DecodeBase64"
  let decodeCompressed = Op_name.of_string "DecodeCompressed"
  let decodeJSONExample = Op_name.of_string "DecodeJSONExample"
  let decodePaddedRaw = Op_name.of_string "DecodePaddedRaw"
  let decodePng = Op_name.of_string "DecodePng"
  let decodeRaw = Op_name.of_string "DecodeRaw"
  let decodeWav = Op_name.of_string "DecodeWav"
  let deepCopy = Op_name.of_string "DeepCopy"
  let deleteSessionTensor = Op_name.of_string "DeleteSessionTensor"
  let denseToDenseSetOperation = Op_name.of_string "DenseToDenseSetOperation"
  let denseToSparseSetOperation = Op_name.of_string "DenseToSparseSetOperation"
  let depthToSpace = Op_name.of_string "DepthToSpace"
  let depthwiseConv2dNative = Op_name.of_string "DepthwiseConv2dNative"
  let depthwiseConv2dNativeBackpropFilter = Op_name.of_string "DepthwiseConv2dNativeBackpropFilter"
  let depthwiseConv2dNativeBackpropInput = Op_name.of_string "DepthwiseConv2dNativeBackpropInput"
  let dequantize = Op_name.of_string "Dequantize"
  let deserializeManySparse = Op_name.of_string "DeserializeManySparse"
  let deserializeSparse = Op_name.of_string "DeserializeSparse"
  let destroyTemporaryVariable = Op_name.of_string "DestroyTemporaryVariable"
  let diag = Op_name.of_string "Diag"
  let diagPart = Op_name.of_string "DiagPart"
  let digamma = Op_name.of_string "Digamma"
  let dilation2D = Op_name.of_string "Dilation2D"
  let dilation2DBackpropFilter = Op_name.of_string "Dilation2DBackpropFilter"
  let dilation2DBackpropInput = Op_name.of_string "Dilation2DBackpropInput"
  let div = Op_name.of_string "Div"
  let divNoNan = Op_name.of_string "DivNoNan"
  let drawBoundingBoxes = Op_name.of_string "DrawBoundingBoxes"
  let drawBoundingBoxesV2 = Op_name.of_string "DrawBoundingBoxesV2"
  let dynamicPartition = Op_name.of_string "DynamicPartition"
  let dynamicStitch = Op_name.of_string "DynamicStitch"
  let editDistance = Op_name.of_string "EditDistance"
  let elu = Op_name.of_string "Elu"
  let eluGrad = Op_name.of_string "EluGrad"
  let empty = Op_name.of_string "Empty"
  let emptyTensorList = Op_name.of_string "EmptyTensorList"
  let encodeBase64 = Op_name.of_string "EncodeBase64"
  let encodePng = Op_name.of_string "EncodePng"
  let encodeWav = Op_name.of_string "EncodeWav"
  let enqueueTPUEmbeddingIntegerBatch = Op_name.of_string "EnqueueTPUEmbeddingIntegerBatch"
  let enqueueTPUEmbeddingSparseBatch = Op_name.of_string "EnqueueTPUEmbeddingSparseBatch"
  let enqueueTPUEmbeddingSparseTensorBatch = Op_name.of_string "EnqueueTPUEmbeddingSparseTensorBatch"
  let ensureShape = Op_name.of_string "EnsureShape"
  let enter = Op_name.of_string "Enter"
  let equal = Op_name.of_string "Equal"
  let erf = Op_name.of_string "Erf"
  let erfc = Op_name.of_string "Erfc"
  let euclideanNorm = Op_name.of_string "EuclideanNorm"
  let exit = Op_name.of_string "Exit"
  let exp = Op_name.of_string "Exp"
  let expandDims = Op_name.of_string "ExpandDims"
  let experimentalAssertNextDataset = Op_name.of_string "ExperimentalAssertNextDataset"
  let experimentalAutoShardDataset = Op_name.of_string "ExperimentalAutoShardDataset"
  let experimentalBytesProducedStatsDataset = Op_name.of_string "ExperimentalBytesProducedStatsDataset"
  let experimentalChooseFastestDataset = Op_name.of_string "ExperimentalChooseFastestDataset"
  let experimentalDatasetCardinality = Op_name.of_string "ExperimentalDatasetCardinality"
  let experimentalDatasetToTFRecord = Op_name.of_string "ExperimentalDatasetToTFRecord"
  let experimentalDenseToSparseBatchDataset = Op_name.of_string "ExperimentalDenseToSparseBatchDataset"
  let experimentalDirectedInterleaveDataset = Op_name.of_string "ExperimentalDirectedInterleaveDataset"
  let experimentalIgnoreErrorsDataset = Op_name.of_string "ExperimentalIgnoreErrorsDataset"
  let experimentalLMDBDataset = Op_name.of_string "ExperimentalLMDBDataset"
  let experimentalLatencyStatsDataset = Op_name.of_string "ExperimentalLatencyStatsDataset"
  let experimentalMatchingFilesDataset = Op_name.of_string "ExperimentalMatchingFilesDataset"
  let experimentalMaxIntraOpParallelismDataset = Op_name.of_string "ExperimentalMaxIntraOpParallelismDataset"
  let experimentalNonSerializableDataset = Op_name.of_string "ExperimentalNonSerializableDataset"
  let experimentalPrivateThreadPoolDataset = Op_name.of_string "ExperimentalPrivateThreadPoolDataset"
  let experimentalRandomDataset = Op_name.of_string "ExperimentalRandomDataset"
  let experimentalRebatchDataset = Op_name.of_string "ExperimentalRebatchDataset"
  let experimentalSleepDataset = Op_name.of_string "ExperimentalSleepDataset"
  let experimentalSlidingWindowDataset = Op_name.of_string "ExperimentalSlidingWindowDataset"
  let experimentalSqlDataset = Op_name.of_string "ExperimentalSqlDataset"
  let experimentalUnbatchDataset = Op_name.of_string "ExperimentalUnbatchDataset"
  let experimentalUniqueDataset = Op_name.of_string "ExperimentalUniqueDataset"
  let expm1 = Op_name.of_string "Expm1"
  let extractGlimpse = Op_name.of_string "ExtractGlimpse"
  let extractImagePatches = Op_name.of_string "ExtractImagePatches"
  let extractJpegShape = Op_name.of_string "ExtractJpegShape"
  let extractVolumePatches = Op_name.of_string "ExtractVolumePatches"
  let fFT = Op_name.of_string "FFT"
  let fFT2D = Op_name.of_string "FFT2D"
  let fFT3D = Op_name.of_string "FFT3D"
  let fIFOQueue = Op_name.of_string "FIFOQueue"
  let fact = Op_name.of_string "Fact"
  let fakeParam = Op_name.of_string "FakeParam"
  let fakeQuantWithMinMaxArgs = Op_name.of_string "FakeQuantWithMinMaxArgs"
  let fakeQuantWithMinMaxArgsGradient = Op_name.of_string "FakeQuantWithMinMaxArgsGradient"
  let fakeQuantWithMinMaxVars = Op_name.of_string "FakeQuantWithMinMaxVars"
  let fakeQuantWithMinMaxVarsGradient = Op_name.of_string "FakeQuantWithMinMaxVarsGradient"
  let fakeQuantWithMinMaxVarsPerChannel = Op_name.of_string "FakeQuantWithMinMaxVarsPerChannel"
  let fakeQuantWithMinMaxVarsPerChannelGradient = Op_name.of_string "FakeQuantWithMinMaxVarsPerChannelGradient"
  let fill = Op_name.of_string "Fill"
  let filterByLastComponentDataset = Op_name.of_string "FilterByLastComponentDataset"
  let fixedLengthRecordDataset = Op_name.of_string "FixedLengthRecordDataset"
  let fixedLengthRecordDatasetV2 = Op_name.of_string "FixedLengthRecordDatasetV2"
  let fixedLengthRecordReader = Op_name.of_string "FixedLengthRecordReader"
  let fixedUnigramCandidateSampler = Op_name.of_string "FixedUnigramCandidateSampler"
  let floor = Op_name.of_string "Floor"
  let floorDiv = Op_name.of_string "FloorDiv"
  let floorMod = Op_name.of_string "FloorMod"
  let fractionalAvgPool = Op_name.of_string "FractionalAvgPool"
  let fractionalAvgPoolGrad = Op_name.of_string "FractionalAvgPoolGrad"
  let fractionalMaxPool = Op_name.of_string "FractionalMaxPool"
  let fractionalMaxPoolGrad = Op_name.of_string "FractionalMaxPoolGrad"
  let fusedBatchNorm = Op_name.of_string "FusedBatchNorm"
  let fusedBatchNormGrad = Op_name.of_string "FusedBatchNormGrad"
  let fusedBatchNormGradV2 = Op_name.of_string "FusedBatchNormGradV2"
  let fusedBatchNormV2 = Op_name.of_string "FusedBatchNormV2"
  let fusedPadConv2D = Op_name.of_string "FusedPadConv2D"
  let fusedResizeAndPadConv2D = Op_name.of_string "FusedResizeAndPadConv2D"
  let gather = Op_name.of_string "Gather"
  let gatherNd = Op_name.of_string "GatherNd"
  let gatherV2 = Op_name.of_string "GatherV2"
  let generateVocabRemapping = Op_name.of_string "GenerateVocabRemapping"
  let getSessionHandle = Op_name.of_string "GetSessionHandle"
  let getSessionTensor = Op_name.of_string "GetSessionTensor"
  let greater = Op_name.of_string "Greater"
  let greaterEqual = Op_name.of_string "GreaterEqual"
  let guaranteeConst = Op_name.of_string "GuaranteeConst"
  let hSVToRGB = Op_name.of_string "HSVToRGB"
  let hashTable = Op_name.of_string "HashTable"
  let histogramFixedWidth = Op_name.of_string "HistogramFixedWidth"
  let histogramSummary = Op_name.of_string "HistogramSummary"
  let hostConst = Op_name.of_string "HostConst"
  let iFFT = Op_name.of_string "IFFT"
  let iFFT2D = Op_name.of_string "IFFT2D"
  let iFFT3D = Op_name.of_string "IFFT3D"
  let iRFFT = Op_name.of_string "IRFFT"
  let iRFFT2D = Op_name.of_string "IRFFT2D"
  let iRFFT3D = Op_name.of_string "IRFFT3D"
  let identity = Op_name.of_string "Identity"
  let identityReader = Op_name.of_string "IdentityReader"
  let igamma = Op_name.of_string "Igamma"
  let igammaGradA = Op_name.of_string "IgammaGradA"
  let igammac = Op_name.of_string "Igammac"
  let imag = Op_name.of_string "Imag"
  let imageSummary = Op_name.of_string "ImageSummary"
  let immutableConst = Op_name.of_string "ImmutableConst"
  let inTopK = Op_name.of_string "InTopK"
  let inTopKV2 = Op_name.of_string "InTopKV2"
  let infeedDequeue = Op_name.of_string "InfeedDequeue"
  let infeedEnqueue = Op_name.of_string "InfeedEnqueue"
  let infeedEnqueuePrelinearizedBuffer = Op_name.of_string "InfeedEnqueuePrelinearizedBuffer"
  let initializeTable = Op_name.of_string "InitializeTable"
  let initializeTableFromTextFile = Op_name.of_string "InitializeTableFromTextFile"
  let inplaceAdd = Op_name.of_string "InplaceAdd"
  let inplaceSub = Op_name.of_string "InplaceSub"
  let inplaceUpdate = Op_name.of_string "InplaceUpdate"
  let inv = Op_name.of_string "Inv"
  let invGrad = Op_name.of_string "InvGrad"
  let invert = Op_name.of_string "Invert"
  let invertPermutation = Op_name.of_string "InvertPermutation"
  let isFinite = Op_name.of_string "IsFinite"
  let isInf = Op_name.of_string "IsInf"
  let isNan = Op_name.of_string "IsNan"
  let isVariableInitialized = Op_name.of_string "IsVariableInitialized"
  let kMC2ChainInitialization = Op_name.of_string "KMC2ChainInitialization"
  let kmeansPlusPlusInitialization = Op_name.of_string "KmeansPlusPlusInitialization"
  let l2Loss = Op_name.of_string "L2Loss"
  let lMDBReader = Op_name.of_string "LMDBReader"
  let lRN = Op_name.of_string "LRN"
  let lRNGrad = Op_name.of_string "LRNGrad"
  let leakyRelu = Op_name.of_string "LeakyRelu"
  let leakyReluGrad = Op_name.of_string "LeakyReluGrad"
  let learnedUnigramCandidateSampler = Op_name.of_string "LearnedUnigramCandidateSampler"
  let leftShift = Op_name.of_string "LeftShift"
  let less = Op_name.of_string "Less"
  let lessEqual = Op_name.of_string "LessEqual"
  let lgamma = Op_name.of_string "Lgamma"
  let linSpace = Op_name.of_string "LinSpace"
  let listDiff = Op_name.of_string "ListDiff"
  let loadAndRemapMatrix = Op_name.of_string "LoadAndRemapMatrix"
  let loadTPUEmbeddingADAMParameters = Op_name.of_string "LoadTPUEmbeddingADAMParameters"
  let loadTPUEmbeddingADAMParametersGradAccumDebug = Op_name.of_string "LoadTPUEmbeddingADAMParametersGradAccumDebug"
  let loadTPUEmbeddingAdadeltaParameters = Op_name.of_string "LoadTPUEmbeddingAdadeltaParameters"
  let loadTPUEmbeddingAdadeltaParametersGradAccumDebug = Op_name.of_string "LoadTPUEmbeddingAdadeltaParametersGradAccumDebug"
  let loadTPUEmbeddingAdagradParameters = Op_name.of_string "LoadTPUEmbeddingAdagradParameters"
  let loadTPUEmbeddingAdagradParametersGradAccumDebug = Op_name.of_string "LoadTPUEmbeddingAdagradParametersGradAccumDebug"
  let loadTPUEmbeddingCenteredRMSPropParameters = Op_name.of_string "LoadTPUEmbeddingCenteredRMSPropParameters"
  let loadTPUEmbeddingFTRLParameters = Op_name.of_string "LoadTPUEmbeddingFTRLParameters"
  let loadTPUEmbeddingFTRLParametersGradAccumDebug = Op_name.of_string "LoadTPUEmbeddingFTRLParametersGradAccumDebug"
  let loadTPUEmbeddingMDLAdagradLightParameters = Op_name.of_string "LoadTPUEmbeddingMDLAdagradLightParameters"
  let loadTPUEmbeddingMomentumParameters = Op_name.of_string "LoadTPUEmbeddingMomentumParameters"
  let loadTPUEmbeddingMomentumParametersGradAccumDebug = Op_name.of_string "LoadTPUEmbeddingMomentumParametersGradAccumDebug"
  let loadTPUEmbeddingProximalAdagradParameters = Op_name.of_string "LoadTPUEmbeddingProximalAdagradParameters"
  let loadTPUEmbeddingProximalAdagradParametersGradAccumDebug = Op_name.of_string "LoadTPUEmbeddingProximalAdagradParametersGradAccumDebug"
  let loadTPUEmbeddingRMSPropParameters = Op_name.of_string "LoadTPUEmbeddingRMSPropParameters"
  let loadTPUEmbeddingRMSPropParametersGradAccumDebug = Op_name.of_string "LoadTPUEmbeddingRMSPropParametersGradAccumDebug"
  let loadTPUEmbeddingStochasticGradientDescentParameters = Op_name.of_string "LoadTPUEmbeddingStochasticGradientDescentParameters"
  let log = Op_name.of_string "Log"
  let log1p = Op_name.of_string "Log1p"
  let logMatrixDeterminant = Op_name.of_string "LogMatrixDeterminant"
  let logSoftmax = Op_name.of_string "LogSoftmax"
  let logUniformCandidateSampler = Op_name.of_string "LogUniformCandidateSampler"
  let logicalAnd = Op_name.of_string "LogicalAnd"
  let logicalNot = Op_name.of_string "LogicalNot"
  let logicalOr = Op_name.of_string "LogicalOr"
  let lookupTableExport = Op_name.of_string "LookupTableExport"
  let lookupTableFind = Op_name.of_string "LookupTableFind"
  let lookupTableImport = Op_name.of_string "LookupTableImport"
  let lookupTableInsert = Op_name.of_string "LookupTableInsert"
  let lookupTableSize = Op_name.of_string "LookupTableSize"
  let loopCond = Op_name.of_string "LoopCond"
  let lowerBound = Op_name.of_string "LowerBound"
  let lu = Op_name.of_string "Lu"
  let mapClear = Op_name.of_string "MapClear"
  let mapIncompleteSize = Op_name.of_string "MapIncompleteSize"
  let mapSize = Op_name.of_string "MapSize"
  let matMul = Op_name.of_string "MatMul"
  let matchingFiles = Op_name.of_string "MatchingFiles"
  let matrixBandPart = Op_name.of_string "MatrixBandPart"
  let matrixDeterminant = Op_name.of_string "MatrixDeterminant"
  let matrixDiag = Op_name.of_string "MatrixDiag"
  let matrixDiagPart = Op_name.of_string "MatrixDiagPart"
  let matrixExponential = Op_name.of_string "MatrixExponential"
  let matrixInverse = Op_name.of_string "MatrixInverse"
  let matrixLogarithm = Op_name.of_string "MatrixLogarithm"
  let matrixSetDiag = Op_name.of_string "MatrixSetDiag"
  let matrixSolve = Op_name.of_string "MatrixSolve"
  let matrixSolveLs = Op_name.of_string "MatrixSolveLs"
  let matrixSquareRoot = Op_name.of_string "MatrixSquareRoot"
  let matrixTriangularSolve = Op_name.of_string "MatrixTriangularSolve"
  let max = Op_name.of_string "Max"
  let maxPool = Op_name.of_string "MaxPool"
  let maxPool3D = Op_name.of_string "MaxPool3D"
  let maxPool3DGrad = Op_name.of_string "MaxPool3DGrad"
  let maxPool3DGradGrad = Op_name.of_string "MaxPool3DGradGrad"
  let maxPoolGrad = Op_name.of_string "MaxPoolGrad"
  let maxPoolGradGrad = Op_name.of_string "MaxPoolGradGrad"
  let maxPoolGradGradV2 = Op_name.of_string "MaxPoolGradGradV2"
  let maxPoolGradGradWithArgmax = Op_name.of_string "MaxPoolGradGradWithArgmax"
  let maxPoolGradV2 = Op_name.of_string "MaxPoolGradV2"
  let maxPoolGradWithArgmax = Op_name.of_string "MaxPoolGradWithArgmax"
  let maxPoolV2 = Op_name.of_string "MaxPoolV2"
  let maxPoolWithArgmax = Op_name.of_string "MaxPoolWithArgmax"
  let maximum = Op_name.of_string "Maximum"
  let mean = Op_name.of_string "Mean"
  let merge = Op_name.of_string "Merge"
  let mergeSummary = Op_name.of_string "MergeSummary"
  let mergeV2Checkpoints = Op_name.of_string "MergeV2Checkpoints"
  let mfcc = Op_name.of_string "Mfcc"
  let min = Op_name.of_string "Min"
  let minimum = Op_name.of_string "Minimum"
  let mirrorPad = Op_name.of_string "MirrorPad"
  let mirrorPadGrad = Op_name.of_string "MirrorPadGrad"
  let mod_ = Op_name.of_string "Mod"
  let modelDataset = Op_name.of_string "ModelDataset"
  let mul = Op_name.of_string "Mul"
  let mulNoNan = Op_name.of_string "MulNoNan"
  let multinomial = Op_name.of_string "Multinomial"
  let mutableDenseHashTable = Op_name.of_string "MutableDenseHashTable"
  let mutableHashTable = Op_name.of_string "MutableHashTable"
  let mutableHashTableOfTensors = Op_name.of_string "MutableHashTableOfTensors"
  let ncclAllReduce = Op_name.of_string "NcclAllReduce"
  let ncclBroadcast = Op_name.of_string "NcclBroadcast"
  let ncclReduce = Op_name.of_string "NcclReduce"
  let nearestNeighbors = Op_name.of_string "NearestNeighbors"
  let neg = Op_name.of_string "Neg"
  let negTrain = Op_name.of_string "NegTrain"
  let nextAfter = Op_name.of_string "NextAfter"
  let nextIteration = Op_name.of_string "NextIteration"
  let noOp = Op_name.of_string "NoOp"
  let nonDeterministicInts = Op_name.of_string "NonDeterministicInts"
  let nonMaxSuppression = Op_name.of_string "NonMaxSuppression"
  let nonMaxSuppressionV2 = Op_name.of_string "NonMaxSuppressionV2"
  let nonMaxSuppressionV3 = Op_name.of_string "NonMaxSuppressionV3"
  let nonMaxSuppressionV4 = Op_name.of_string "NonMaxSuppressionV4"
  let nonMaxSuppressionWithOverlaps = Op_name.of_string "NonMaxSuppressionWithOverlaps"
  let notEqual = Op_name.of_string "NotEqual"
  let nthElement = Op_name.of_string "NthElement"
  let oneHot = Op_name.of_string "OneHot"
  let onesLike = Op_name.of_string "OnesLike"
  let optimizeDataset = Op_name.of_string "OptimizeDataset"
  let optionalHasValue = Op_name.of_string "OptionalHasValue"
  let optionalNone = Op_name.of_string "OptionalNone"
  let orderedMapClear = Op_name.of_string "OrderedMapClear"
  let orderedMapIncompleteSize = Op_name.of_string "OrderedMapIncompleteSize"
  let orderedMapSize = Op_name.of_string "OrderedMapSize"
  let outfeedDequeue = Op_name.of_string "OutfeedDequeue"
  let outfeedEnqueue = Op_name.of_string "OutfeedEnqueue"
  let pack = Op_name.of_string "Pack"
  let pad = Op_name.of_string "Pad"
  let padV2 = Op_name.of_string "PadV2"
  let paddingFIFOQueue = Op_name.of_string "PaddingFIFOQueue"
  let parallelConcat = Op_name.of_string "ParallelConcat"
  let parallelDynamicStitch = Op_name.of_string "ParallelDynamicStitch"
  let parameterizedTruncatedNormal = Op_name.of_string "ParameterizedTruncatedNormal"
  let parseTensor = Op_name.of_string "ParseTensor"
  let placeholder = Op_name.of_string "Placeholder"
  let placeholderV2 = Op_name.of_string "PlaceholderV2"
  let placeholderWithDefault = Op_name.of_string "PlaceholderWithDefault"
  let polygamma = Op_name.of_string "Polygamma"
  let pow = Op_name.of_string "Pow"
  let prefetchDataset = Op_name.of_string "PrefetchDataset"
  let prelinearize = Op_name.of_string "Prelinearize"
  let preventGradient = Op_name.of_string "PreventGradient"
  let printV2 = Op_name.of_string "PrintV2"
  let priorityQueue = Op_name.of_string "PriorityQueue"
  let prod = Op_name.of_string "Prod"
  let qr = Op_name.of_string "Qr"
  let quantizeAndDequantize = Op_name.of_string "QuantizeAndDequantize"
  let quantizeAndDequantizeV2 = Op_name.of_string "QuantizeAndDequantizeV2"
  let quantizeAndDequantizeV3 = Op_name.of_string "QuantizeAndDequantizeV3"
  let quantizeDownAndShrinkRange = Op_name.of_string "QuantizeDownAndShrinkRange"
  let quantizeV2 = Op_name.of_string "QuantizeV2"
  let quantizedAdd = Op_name.of_string "QuantizedAdd"
  let quantizedAvgPool = Op_name.of_string "QuantizedAvgPool"
  let quantizedBatchNormWithGlobalNormalization = Op_name.of_string "QuantizedBatchNormWithGlobalNormalization"
  let quantizedBiasAdd = Op_name.of_string "QuantizedBiasAdd"
  let quantizedConcat = Op_name.of_string "QuantizedConcat"
  let quantizedConv2D = Op_name.of_string "QuantizedConv2D"
  let quantizedConv2DAndRelu = Op_name.of_string "QuantizedConv2DAndRelu"
  let quantizedConv2DAndReluAndRequantize = Op_name.of_string "QuantizedConv2DAndReluAndRequantize"
  let quantizedConv2DAndRequantize = Op_name.of_string "QuantizedConv2DAndRequantize"
  let quantizedConv2DPerChannel = Op_name.of_string "QuantizedConv2DPerChannel"
  let quantizedConv2DWithBias = Op_name.of_string "QuantizedConv2DWithBias"
  let quantizedConv2DWithBiasAndRelu = Op_name.of_string "QuantizedConv2DWithBiasAndRelu"
  let quantizedConv2DWithBiasAndReluAndRequantize = Op_name.of_string "QuantizedConv2DWithBiasAndReluAndRequantize"
  let quantizedConv2DWithBiasAndRequantize = Op_name.of_string "QuantizedConv2DWithBiasAndRequantize"
  let quantizedConv2DWithBiasSignedSumAndReluAndRequantize = Op_name.of_string "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"
  let quantizedConv2DWithBiasSumAndRelu = Op_name.of_string "QuantizedConv2DWithBiasSumAndRelu"
  let quantizedConv2DWithBiasSumAndReluAndRequantize = Op_name.of_string "QuantizedConv2DWithBiasSumAndReluAndRequantize"
  let quantizedDepthwiseConv2D = Op_name.of_string "QuantizedDepthwiseConv2D"
  let quantizedDepthwiseConv2DWithBias = Op_name.of_string "QuantizedDepthwiseConv2DWithBias"
  let quantizedDepthwiseConv2DWithBiasAndRelu = Op_name.of_string "QuantizedDepthwiseConv2DWithBiasAndRelu"
  let quantizedDepthwiseConv2DWithBiasAndReluAndRequantize = Op_name.of_string "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize"
  let quantizedInstanceNorm = Op_name.of_string "QuantizedInstanceNorm"
  let quantizedMatMul = Op_name.of_string "QuantizedMatMul"
  let quantizedMaxPool = Op_name.of_string "QuantizedMaxPool"
  let quantizedMul = Op_name.of_string "QuantizedMul"
  let quantizedRelu = Op_name.of_string "QuantizedRelu"
  let quantizedRelu6 = Op_name.of_string "QuantizedRelu6"
  let quantizedReluX = Op_name.of_string "QuantizedReluX"
  let quantizedReshape = Op_name.of_string "QuantizedReshape"
  let quantizedResizeBilinear = Op_name.of_string "QuantizedResizeBilinear"
  let queueClose = Op_name.of_string "QueueClose"
  let queueIsClosed = Op_name.of_string "QueueIsClosed"
  let queueSize = Op_name.of_string "QueueSize"
  let rFFT = Op_name.of_string "RFFT"
  let rFFT2D = Op_name.of_string "RFFT2D"
  let rFFT3D = Op_name.of_string "RFFT3D"
  let rGBToHSV = Op_name.of_string "RGBToHSV"
  let raggedRange = Op_name.of_string "RaggedRange"
  let raggedTensorToSparse = Op_name.of_string "RaggedTensorToSparse"
  let randomCrop = Op_name.of_string "RandomCrop"
  let randomGamma = Op_name.of_string "RandomGamma"
  let randomGammaGrad = Op_name.of_string "RandomGammaGrad"
  let randomPoisson = Op_name.of_string "RandomPoisson"
  let randomPoissonV2 = Op_name.of_string "RandomPoissonV2"
  let randomShuffle = Op_name.of_string "RandomShuffle"
  let randomShuffleQueue = Op_name.of_string "RandomShuffleQueue"
  let randomStandardNormal = Op_name.of_string "RandomStandardNormal"
  let randomUniform = Op_name.of_string "RandomUniform"
  let randomUniformInt = Op_name.of_string "RandomUniformInt"
  let range = Op_name.of_string "Range"
  let rangeDataset = Op_name.of_string "RangeDataset"
  let rank = Op_name.of_string "Rank"
  let readFile = Op_name.of_string "ReadFile"
  let readerNumRecordsProduced = Op_name.of_string "ReaderNumRecordsProduced"
  let readerNumWorkUnitsCompleted = Op_name.of_string "ReaderNumWorkUnitsCompleted"
  let readerRead = Op_name.of_string "ReaderRead"
  let readerReadUpTo = Op_name.of_string "ReaderReadUpTo"
  let readerReset = Op_name.of_string "ReaderReset"
  let readerRestoreState = Op_name.of_string "ReaderRestoreState"
  let readerSerializeState = Op_name.of_string "ReaderSerializeState"
  let real = Op_name.of_string "Real"
  let realDiv = Op_name.of_string "RealDiv"
  let reciprocal = Op_name.of_string "Reciprocal"
  let reciprocalGrad = Op_name.of_string "ReciprocalGrad"
  let recordInput = Op_name.of_string "RecordInput"
  let recvTPUEmbeddingActivations = Op_name.of_string "RecvTPUEmbeddingActivations"
  let reduceJoin = Op_name.of_string "ReduceJoin"
  let refEnter = Op_name.of_string "RefEnter"
  let refExit = Op_name.of_string "RefExit"
  let refIdentity = Op_name.of_string "RefIdentity"
  let refMerge = Op_name.of_string "RefMerge"
  let refNextIteration = Op_name.of_string "RefNextIteration"
  let refSelect = Op_name.of_string "RefSelect"
  let refSwitch = Op_name.of_string "RefSwitch"
  let regexFullMatch = Op_name.of_string "RegexFullMatch"
  let regexReplace = Op_name.of_string "RegexReplace"
  let relu = Op_name.of_string "Relu"
  let relu6 = Op_name.of_string "Relu6"
  let relu6Grad = Op_name.of_string "Relu6Grad"
  let reluGrad = Op_name.of_string "ReluGrad"
  let repeatDataset = Op_name.of_string "RepeatDataset"
  let requantizationRange = Op_name.of_string "RequantizationRange"
  let requantizationRangePerChannel = Op_name.of_string "RequantizationRangePerChannel"
  let requantize = Op_name.of_string "Requantize"
  let requantizePerChannel = Op_name.of_string "RequantizePerChannel"
  let reshape = Op_name.of_string "Reshape"
  let resizeArea = Op_name.of_string "ResizeArea"
  let resizeBicubic = Op_name.of_string "ResizeBicubic"
  let resizeBicubicGrad = Op_name.of_string "ResizeBicubicGrad"
  let resizeBilinear = Op_name.of_string "ResizeBilinear"
  let resizeBilinearGrad = Op_name.of_string "ResizeBilinearGrad"
  let resizeNearestNeighbor = Op_name.of_string "ResizeNearestNeighbor"
  let resizeNearestNeighborGrad = Op_name.of_string "ResizeNearestNeighborGrad"
  let restore = Op_name.of_string "Restore"
  let restoreSlice = Op_name.of_string "RestoreSlice"
  let retrieveTPUEmbeddingADAMParameters = Op_name.of_string "RetrieveTPUEmbeddingADAMParameters"
  let retrieveTPUEmbeddingADAMParametersGradAccumDebug = Op_name.of_string "RetrieveTPUEmbeddingADAMParametersGradAccumDebug"
  let retrieveTPUEmbeddingAdadeltaParameters = Op_name.of_string "RetrieveTPUEmbeddingAdadeltaParameters"
  let retrieveTPUEmbeddingAdadeltaParametersGradAccumDebug = Op_name.of_string "RetrieveTPUEmbeddingAdadeltaParametersGradAccumDebug"
  let retrieveTPUEmbeddingAdagradParameters = Op_name.of_string "RetrieveTPUEmbeddingAdagradParameters"
  let retrieveTPUEmbeddingAdagradParametersGradAccumDebug = Op_name.of_string "RetrieveTPUEmbeddingAdagradParametersGradAccumDebug"
  let retrieveTPUEmbeddingCenteredRMSPropParameters = Op_name.of_string "RetrieveTPUEmbeddingCenteredRMSPropParameters"
  let retrieveTPUEmbeddingFTRLParameters = Op_name.of_string "RetrieveTPUEmbeddingFTRLParameters"
  let retrieveTPUEmbeddingFTRLParametersGradAccumDebug = Op_name.of_string "RetrieveTPUEmbeddingFTRLParametersGradAccumDebug"
  let retrieveTPUEmbeddingMDLAdagradLightParameters = Op_name.of_string "RetrieveTPUEmbeddingMDLAdagradLightParameters"
  let retrieveTPUEmbeddingMomentumParameters = Op_name.of_string "RetrieveTPUEmbeddingMomentumParameters"
  let retrieveTPUEmbeddingMomentumParametersGradAccumDebug = Op_name.of_string "RetrieveTPUEmbeddingMomentumParametersGradAccumDebug"
  let retrieveTPUEmbeddingProximalAdagradParameters = Op_name.of_string "RetrieveTPUEmbeddingProximalAdagradParameters"
  let retrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug = Op_name.of_string "RetrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug"
  let retrieveTPUEmbeddingRMSPropParameters = Op_name.of_string "RetrieveTPUEmbeddingRMSPropParameters"
  let retrieveTPUEmbeddingRMSPropParametersGradAccumDebug = Op_name.of_string "RetrieveTPUEmbeddingRMSPropParametersGradAccumDebug"
  let retrieveTPUEmbeddingStochasticGradientDescentParameters = Op_name.of_string "RetrieveTPUEmbeddingStochasticGradientDescentParameters"
  let reverse = Op_name.of_string "Reverse"
  let reverseSequence = Op_name.of_string "ReverseSequence"
  let reverseV2 = Op_name.of_string "ReverseV2"
  let rightShift = Op_name.of_string "RightShift"
  let rint = Op_name.of_string "Rint"
  let roll = Op_name.of_string "Roll"
  let round = Op_name.of_string "Round"
  let rpc = Op_name.of_string "Rpc"
  let rsqrt = Op_name.of_string "Rsqrt"
  let rsqrtGrad = Op_name.of_string "RsqrtGrad"
  let sampleDistortedBoundingBox = Op_name.of_string "SampleDistortedBoundingBox"
  let sampleDistortedBoundingBoxV2 = Op_name.of_string "SampleDistortedBoundingBoxV2"
  let samplingDataset = Op_name.of_string "SamplingDataset"
  let scalarSummary = Op_name.of_string "ScalarSummary"
  let scaleAndTranslate = Op_name.of_string "ScaleAndTranslate"
  let scaleAndTranslateGrad = Op_name.of_string "ScaleAndTranslateGrad"
  let scatterAdd = Op_name.of_string "ScatterAdd"
  let scatterDiv = Op_name.of_string "ScatterDiv"
  let scatterMax = Op_name.of_string "ScatterMax"
  let scatterMin = Op_name.of_string "ScatterMin"
  let scatterMul = Op_name.of_string "ScatterMul"
  let scatterNd = Op_name.of_string "ScatterNd"
  let scatterNdAdd = Op_name.of_string "ScatterNdAdd"
  let scatterNdNonAliasingAdd = Op_name.of_string "ScatterNdNonAliasingAdd"
  let scatterNdSub = Op_name.of_string "ScatterNdSub"
  let scatterNdUpdate = Op_name.of_string "ScatterNdUpdate"
  let scatterSub = Op_name.of_string "ScatterSub"
  let scatterUpdate = Op_name.of_string "ScatterUpdate"
  let sdcaFprint = Op_name.of_string "SdcaFprint"
  let sdcaShrinkL1 = Op_name.of_string "SdcaShrinkL1"
  let segmentMax = Op_name.of_string "SegmentMax"
  let segmentMean = Op_name.of_string "SegmentMean"
  let segmentMin = Op_name.of_string "SegmentMin"
  let segmentProd = Op_name.of_string "SegmentProd"
  let segmentSum = Op_name.of_string "SegmentSum"
  let select = Op_name.of_string "Select"
  let selfAdjointEig = Op_name.of_string "SelfAdjointEig"
  let selfAdjointEigV2 = Op_name.of_string "SelfAdjointEigV2"
  let selu = Op_name.of_string "Selu"
  let seluGrad = Op_name.of_string "SeluGrad"
  let sendTPUEmbeddingGradients = Op_name.of_string "SendTPUEmbeddingGradients"
  let serializeManySparse = Op_name.of_string "SerializeManySparse"
  let serializeSparse = Op_name.of_string "SerializeSparse"
  let serializeTensor = Op_name.of_string "SerializeTensor"
  let setSize = Op_name.of_string "SetSize"
  let shape = Op_name.of_string "Shape"
  let shapeN = Op_name.of_string "ShapeN"
  let shardDataset = Op_name.of_string "ShardDataset"
  let shardedFilename = Op_name.of_string "ShardedFilename"
  let shardedFilespec = Op_name.of_string "ShardedFilespec"
  let shuffleAndRepeatDataset = Op_name.of_string "ShuffleAndRepeatDataset"
  let shuffleDataset = Op_name.of_string "ShuffleDataset"
  let shutdownDistributedTPU = Op_name.of_string "ShutdownDistributedTPU"
  let sigmoid = Op_name.of_string "Sigmoid"
  let sigmoidGrad = Op_name.of_string "SigmoidGrad"
  let sign = Op_name.of_string "Sign"
  let sin = Op_name.of_string "Sin"
  let sinh = Op_name.of_string "Sinh"
  let size = Op_name.of_string "Size"
  let skipDataset = Op_name.of_string "SkipDataset"
  let skipgram = Op_name.of_string "Skipgram"
  let slice = Op_name.of_string "Slice"
  let snapshot = Op_name.of_string "Snapshot"
  let softmax = Op_name.of_string "Softmax"
  let softmaxCrossEntropyWithLogits = Op_name.of_string "SoftmaxCrossEntropyWithLogits"
  let softplus = Op_name.of_string "Softplus"
  let softplusGrad = Op_name.of_string "SoftplusGrad"
  let softsign = Op_name.of_string "Softsign"
  let softsignGrad = Op_name.of_string "SoftsignGrad"
  let spaceToBatch = Op_name.of_string "SpaceToBatch"
  let spaceToBatchND = Op_name.of_string "SpaceToBatchND"
  let spaceToDepth = Op_name.of_string "SpaceToDepth"
  let sparseAccumulatorApplyGradient = Op_name.of_string "SparseAccumulatorApplyGradient"
  let sparseAccumulatorTakeGradient = Op_name.of_string "SparseAccumulatorTakeGradient"
  let sparseAdd = Op_name.of_string "SparseAdd"
  let sparseAddGrad = Op_name.of_string "SparseAddGrad"
  let sparseApplyAdadelta = Op_name.of_string "SparseApplyAdadelta"
  let sparseApplyAdagrad = Op_name.of_string "SparseApplyAdagrad"
  let sparseApplyAdagradDA = Op_name.of_string "SparseApplyAdagradDA"
  let sparseApplyCenteredRMSProp = Op_name.of_string "SparseApplyCenteredRMSProp"
  let sparseApplyFtrl = Op_name.of_string "SparseApplyFtrl"
  let sparseApplyFtrlV2 = Op_name.of_string "SparseApplyFtrlV2"
  let sparseApplyMomentum = Op_name.of_string "SparseApplyMomentum"
  let sparseApplyProximalAdagrad = Op_name.of_string "SparseApplyProximalAdagrad"
  let sparseApplyProximalGradientDescent = Op_name.of_string "SparseApplyProximalGradientDescent"
  let sparseApplyRMSProp = Op_name.of_string "SparseApplyRMSProp"
  let sparseConcat = Op_name.of_string "SparseConcat"
  let sparseConditionalAccumulator = Op_name.of_string "SparseConditionalAccumulator"
  let sparseDenseCwiseAdd = Op_name.of_string "SparseDenseCwiseAdd"
  let sparseDenseCwiseDiv = Op_name.of_string "SparseDenseCwiseDiv"
  let sparseDenseCwiseMul = Op_name.of_string "SparseDenseCwiseMul"
  let sparseFillEmptyRows = Op_name.of_string "SparseFillEmptyRows"
  let sparseFillEmptyRowsGrad = Op_name.of_string "SparseFillEmptyRowsGrad"
  let sparseMatMul = Op_name.of_string "SparseMatMul"
  let sparseReduceMax = Op_name.of_string "SparseReduceMax"
  let sparseReduceMaxSparse = Op_name.of_string "SparseReduceMaxSparse"
  let sparseReduceSum = Op_name.of_string "SparseReduceSum"
  let sparseReduceSumSparse = Op_name.of_string "SparseReduceSumSparse"
  let sparseReorder = Op_name.of_string "SparseReorder"
  let sparseReshape = Op_name.of_string "SparseReshape"
  let sparseSegmentMean = Op_name.of_string "SparseSegmentMean"
  let sparseSegmentMeanGrad = Op_name.of_string "SparseSegmentMeanGrad"
  let sparseSegmentMeanWithNumSegments = Op_name.of_string "SparseSegmentMeanWithNumSegments"
  let sparseSegmentSqrtN = Op_name.of_string "SparseSegmentSqrtN"
  let sparseSegmentSqrtNGrad = Op_name.of_string "SparseSegmentSqrtNGrad"
  let sparseSegmentSqrtNWithNumSegments = Op_name.of_string "SparseSegmentSqrtNWithNumSegments"
  let sparseSegmentSum = Op_name.of_string "SparseSegmentSum"
  let sparseSegmentSumWithNumSegments = Op_name.of_string "SparseSegmentSumWithNumSegments"
  let sparseSlice = Op_name.of_string "SparseSlice"
  let sparseSliceGrad = Op_name.of_string "SparseSliceGrad"
  let sparseSoftmax = Op_name.of_string "SparseSoftmax"
  let sparseSoftmaxCrossEntropyWithLogits = Op_name.of_string "SparseSoftmaxCrossEntropyWithLogits"
  let sparseSparseMaximum = Op_name.of_string "SparseSparseMaximum"
  let sparseSparseMinimum = Op_name.of_string "SparseSparseMinimum"
  let sparseTensorDenseAdd = Op_name.of_string "SparseTensorDenseAdd"
  let sparseTensorDenseMatMul = Op_name.of_string "SparseTensorDenseMatMul"
  let sparseTensorSliceDataset = Op_name.of_string "SparseTensorSliceDataset"
  let sparseToDense = Op_name.of_string "SparseToDense"
  let sparseToSparseSetOperation = Op_name.of_string "SparseToSparseSetOperation"
  let split = Op_name.of_string "Split"
  let splitV = Op_name.of_string "SplitV"
  let sqrt = Op_name.of_string "Sqrt"
  let sqrtGrad = Op_name.of_string "SqrtGrad"
  let square = Op_name.of_string "Square"
  let squaredDifference = Op_name.of_string "SquaredDifference"
  let squeeze = Op_name.of_string "Squeeze"
  let stack = Op_name.of_string "Stack"
  let stackClose = Op_name.of_string "StackClose"
  let stackPop = Op_name.of_string "StackPop"
  let stackPush = Op_name.of_string "StackPush"
  let stageClear = Op_name.of_string "StageClear"
  let stageSize = Op_name.of_string "StageSize"
  let statelessMultinomial = Op_name.of_string "StatelessMultinomial"
  let statelessRandomNormal = Op_name.of_string "StatelessRandomNormal"
  let statelessRandomUniform = Op_name.of_string "StatelessRandomUniform"
  let statelessRandomUniformInt = Op_name.of_string "StatelessRandomUniformInt"
  let statelessTruncatedNormal = Op_name.of_string "StatelessTruncatedNormal"
  let staticRegexFullMatch = Op_name.of_string "StaticRegexFullMatch"
  let staticRegexReplace = Op_name.of_string "StaticRegexReplace"
  let stopGradient = Op_name.of_string "StopGradient"
  let stridedSlice = Op_name.of_string "StridedSlice"
  let stridedSliceAssign = Op_name.of_string "StridedSliceAssign"
  let stridedSliceGrad = Op_name.of_string "StridedSliceGrad"
  let stringJoin = Op_name.of_string "StringJoin"
  let stringLength = Op_name.of_string "StringLength"
  let stringSplit = Op_name.of_string "StringSplit"
  let stringSplitV2 = Op_name.of_string "StringSplitV2"
  let stringStrip = Op_name.of_string "StringStrip"
  let stringToHashBucket = Op_name.of_string "StringToHashBucket"
  let stringToHashBucketFast = Op_name.of_string "StringToHashBucketFast"
  let stringToHashBucketStrong = Op_name.of_string "StringToHashBucketStrong"
  let stringToNumber = Op_name.of_string "StringToNumber"
  let sub = Op_name.of_string "Sub"
  let substr = Op_name.of_string "Substr"
  let sum = Op_name.of_string "Sum"
  let svd = Op_name.of_string "Svd"
  let switch = Op_name.of_string "Switch"
  let tFRecordDataset = Op_name.of_string "TFRecordDataset"
  let tFRecordReader = Op_name.of_string "TFRecordReader"
  let tPUCompilationResult = Op_name.of_string "TPUCompilationResult"
  let tPUEmbeddingActivations = Op_name.of_string "TPUEmbeddingActivations"
  let tPUOrdinalSelector = Op_name.of_string "TPUOrdinalSelector"
  let tPUReplicateMetadata = Op_name.of_string "TPUReplicateMetadata"
  let tPUReplicatedInput = Op_name.of_string "TPUReplicatedInput"
  let tPUReplicatedOutput = Op_name.of_string "TPUReplicatedOutput"
  let takeDataset = Op_name.of_string "TakeDataset"
  let takeManySparseFromTensorsMap = Op_name.of_string "TakeManySparseFromTensorsMap"
  let tan = Op_name.of_string "Tan"
  let tanh = Op_name.of_string "Tanh"
  let tanhGrad = Op_name.of_string "TanhGrad"
  let temporaryVariable = Op_name.of_string "TemporaryVariable"
  let tensorArray = Op_name.of_string "TensorArray"
  let tensorArrayClose = Op_name.of_string "TensorArrayClose"
  let tensorArrayCloseV2 = Op_name.of_string "TensorArrayCloseV2"
  let tensorArrayConcat = Op_name.of_string "TensorArrayConcat"
  let tensorArrayConcatV2 = Op_name.of_string "TensorArrayConcatV2"
  let tensorArrayGather = Op_name.of_string "TensorArrayGather"
  let tensorArrayGatherV2 = Op_name.of_string "TensorArrayGatherV2"
  let tensorArrayGrad = Op_name.of_string "TensorArrayGrad"
  let tensorArrayGradV2 = Op_name.of_string "TensorArrayGradV2"
  let tensorArrayPack = Op_name.of_string "TensorArrayPack"
  let tensorArrayRead = Op_name.of_string "TensorArrayRead"
  let tensorArrayReadV2 = Op_name.of_string "TensorArrayReadV2"
  let tensorArrayScatter = Op_name.of_string "TensorArrayScatter"
  let tensorArrayScatterV2 = Op_name.of_string "TensorArrayScatterV2"
  let tensorArraySize = Op_name.of_string "TensorArraySize"
  let tensorArraySizeV2 = Op_name.of_string "TensorArraySizeV2"
  let tensorArraySplit = Op_name.of_string "TensorArraySplit"
  let tensorArraySplitV2 = Op_name.of_string "TensorArraySplitV2"
  let tensorArrayUnpack = Op_name.of_string "TensorArrayUnpack"
  let tensorArrayV2 = Op_name.of_string "TensorArrayV2"
  let tensorArrayWrite = Op_name.of_string "TensorArrayWrite"
  let tensorArrayWriteV2 = Op_name.of_string "TensorArrayWriteV2"
  let tensorListConcat = Op_name.of_string "TensorListConcat"
  let tensorListConcatLists = Op_name.of_string "TensorListConcatLists"
  let tensorListConcatV2 = Op_name.of_string "TensorListConcatV2"
  let tensorListElementShape = Op_name.of_string "TensorListElementShape"
  let tensorListFromTensor = Op_name.of_string "TensorListFromTensor"
  let tensorListGather = Op_name.of_string "TensorListGather"
  let tensorListGetItem = Op_name.of_string "TensorListGetItem"
  let tensorListLength = Op_name.of_string "TensorListLength"
  let tensorListPopBack = Op_name.of_string "TensorListPopBack"
  let tensorListPushBack = Op_name.of_string "TensorListPushBack"
  let tensorListPushBackBatch = Op_name.of_string "TensorListPushBackBatch"
  let tensorListReserve = Op_name.of_string "TensorListReserve"
  let tensorListResize = Op_name.of_string "TensorListResize"
  let tensorListScatter = Op_name.of_string "TensorListScatter"
  let tensorListScatterIntoExistingList = Op_name.of_string "TensorListScatterIntoExistingList"
  let tensorListScatterV2 = Op_name.of_string "TensorListScatterV2"
  let tensorListSetItem = Op_name.of_string "TensorListSetItem"
  let tensorListSplit = Op_name.of_string "TensorListSplit"
  let tensorListStack = Op_name.of_string "TensorListStack"
  let tensorScatterAdd = Op_name.of_string "TensorScatterAdd"
  let tensorScatterSub = Op_name.of_string "TensorScatterSub"
  let tensorScatterUpdate = Op_name.of_string "TensorScatterUpdate"
  let tensorStridedSliceUpdate = Op_name.of_string "TensorStridedSliceUpdate"
  let tensorSummary = Op_name.of_string "TensorSummary"
  let tensorSummaryV2 = Op_name.of_string "TensorSummaryV2"
  let textLineDataset = Op_name.of_string "TextLineDataset"
  let textLineReader = Op_name.of_string "TextLineReader"
  let threadUnsafeUnigramCandidateSampler = Op_name.of_string "ThreadUnsafeUnigramCandidateSampler"
  let tile = Op_name.of_string "Tile"
  let tileGrad = Op_name.of_string "TileGrad"
  let timestamp = Op_name.of_string "Timestamp"
  let topK = Op_name.of_string "TopK"
  let topKV2 = Op_name.of_string "TopKV2"
  let transpose = Op_name.of_string "Transpose"
  let tridiagonalSolve = Op_name.of_string "TridiagonalSolve"
  let truncateDiv = Op_name.of_string "TruncateDiv"
  let truncateMod = Op_name.of_string "TruncateMod"
  let truncatedNormal = Op_name.of_string "TruncatedNormal"
  let tryRpc = Op_name.of_string "TryRpc"
  let unbatch = Op_name.of_string "Unbatch"
  let unbatchGrad = Op_name.of_string "UnbatchGrad"
  let unicodeDecode = Op_name.of_string "UnicodeDecode"
  let unicodeDecodeWithOffsets = Op_name.of_string "UnicodeDecodeWithOffsets"
  let unicodeEncode = Op_name.of_string "UnicodeEncode"
  let unicodeScript = Op_name.of_string "UnicodeScript"
  let unicodeTranscode = Op_name.of_string "UnicodeTranscode"
  let uniformCandidateSampler = Op_name.of_string "UniformCandidateSampler"
  let unique = Op_name.of_string "Unique"
  let uniqueV2 = Op_name.of_string "UniqueV2"
  let uniqueWithCounts = Op_name.of_string "UniqueWithCounts"
  let uniqueWithCountsV2 = Op_name.of_string "UniqueWithCountsV2"
  let unpack = Op_name.of_string "Unpack"
  let unravelIndex = Op_name.of_string "UnravelIndex"
  let unsortedSegmentMax = Op_name.of_string "UnsortedSegmentMax"
  let unsortedSegmentMin = Op_name.of_string "UnsortedSegmentMin"
  let unsortedSegmentProd = Op_name.of_string "UnsortedSegmentProd"
  let unsortedSegmentSum = Op_name.of_string "UnsortedSegmentSum"
  let unwrapDatasetVariant = Op_name.of_string "UnwrapDatasetVariant"
  let upperBound = Op_name.of_string "UpperBound"
  let variable = Op_name.of_string "Variable"
  let variableV2 = Op_name.of_string "VariableV2"
  let where = Op_name.of_string "Where"
  let wholeFileReader = Op_name.of_string "WholeFileReader"
  let windowDataset = Op_name.of_string "WindowDataset"
  let workerHeartbeat = Op_name.of_string "WorkerHeartbeat"
  let wrapDatasetVariant = Op_name.of_string "WrapDatasetVariant"
  let writeFile = Op_name.of_string "WriteFile"
  let xdivy = Op_name.of_string "Xdivy"
  let xlogy = Op_name.of_string "Xlogy"
  let zerosLike = Op_name.of_string "ZerosLike"
  let zeta = Op_name.of_string "Zeta"
  let zipDataset = Op_name.of_string "ZipDataset"
end

let abort
    ?(name = "Abort")
    ?error_msg
    ?exit_without_error
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match error_msg with | None -> attributes | Some error_msg -> ("error_msg", String error_msg) :: attributes
  in
  let attributes =
    match exit_without_error with | None -> attributes | Some exit_without_error -> ("exit_without_error", Bool exit_without_error) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.abort in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let abs
    ?(name = "Abs")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.abs in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let accumulateNV2
    ?(name = "AccumulateNV2")
    ~shape
    ?(control_inputs = [])
    (inputs__ : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn inputs__))) ] in
  let attributes =
    ("N", Int (List.length inputs__)) :: attributes
  in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.accumulateNV2 in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn inputs__))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let accumulatorApplyGradient
    ?(name = "AccumulatorApplyGradient")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (local_step : [ `int64 ] t)
    (gradient : ([< `float | `double | `int32 | `complex64 | `int64 ] as 'dtype) t)
  =
  let attributes = [ "dtype", Type (P (Node.output_type gradient)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.accumulatorApplyGradient in
  let inputs = [ (`single (P handle)); (`single (P local_step)); (`single (P gradient)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let accumulatorNumAccumulated
    ?(name = "AccumulatorNumAccumulated")
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.accumulatorNumAccumulated in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let accumulatorSetGlobalStep
    ?(name = "AccumulatorSetGlobalStep")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (new_global_step : [ `int64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.accumulatorSetGlobalStep in
  let inputs = [ (`single (P handle)); (`single (P new_global_step)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let accumulatorTakeGradient
    ?(name = "AccumulatorTakeGradient")
    ~type_
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (num_required : [ `int32 ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.accumulatorTakeGradient in
  let inputs = [ (`single (P handle)); (`single (P num_required)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let acos
    ?(name = "Acos")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.acos in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let acosh
    ?(name = "Acosh")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.acosh in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let add
    ?(name = "Add")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.add in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let addManySparseToTensorsMap
    ?(name = "AddManySparseToTensorsMap")
    ?container
    ?shared_name
    ?(control_inputs = [])
    (sparse_indices : [ `int64 ] t)
    (sparse_values : 't t)
    (sparse_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type sparse_values)) ] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.addManySparseToTensorsMap in
  let inputs = [ (`single (P sparse_indices)); (`single (P sparse_values)); (`single (P sparse_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let addN
    ?(name = "AddN")
    ?(control_inputs = [])
    (inputs__ : ([< `float | `double | `int32 | `complex64 | `int64 | `variant ] as 't) t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn inputs__))) ] in
  let attributes =
    ("N", Int (List.length inputs__)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.addN in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn inputs__))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let addSparseToTensorsMap
    ?(name = "AddSparseToTensorsMap")
    ?container
    ?shared_name
    ?(control_inputs = [])
    (sparse_indices : [ `int64 ] t)
    (sparse_values : 't t)
    (sparse_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type sparse_values)) ] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.addSparseToTensorsMap in
  let inputs = [ (`single (P sparse_indices)); (`single (P sparse_values)); (`single (P sparse_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let addV2
    ?(name = "AddV2")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.addV2 in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let adjustContrast
    ?(name = "AdjustContrast")
    ?(control_inputs = [])
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (contrast_factor : [ `float ] t)
    (min_value : [ `float ] t)
    (max_value : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.adjustContrast in
  let inputs = [ (`single (P images)); (`single (P contrast_factor)); (`single (P min_value)); (`single (P max_value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let adjustContrastv2
    ?(name = "AdjustContrastv2")
    ?(control_inputs = [])
    (images : ([< `float ] as 't) t)
    (contrast_factor : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.adjustContrastv2 in
  let inputs = [ (`single (P images)); (`single (P contrast_factor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let adjustHue
    ?(name = "AdjustHue")
    ?(control_inputs = [])
    (images : ([< `float ] as 't) t)
    (delta : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.adjustHue in
  let inputs = [ (`single (P images)); (`single (P delta)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let adjustSaturation
    ?(name = "AdjustSaturation")
    ?(control_inputs = [])
    (images : ([< `float ] as 't) t)
    (scale : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.adjustSaturation in
  let inputs = [ (`single (P images)); (`single (P scale)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let all
    ?(name = "All")
    ?keep_dims
    ?(control_inputs = [])
    (input : [ `bool ] t)
    (reduction_indices : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type reduction_indices)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.all in
  let inputs = [ (`single (P input)); (`single (P reduction_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let allCandidateSampler
    ?(name = "AllCandidateSampler")
    ~num_true
    ~num_sampled
    ~unique
    ?seed
    ?seed2
    ?(control_inputs = [])
    (true_classes : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_true", Int num_true) :: attributes
  in
  let attributes =
    ("num_sampled", Int num_sampled) :: attributes
  in
  let attributes =
    ("unique", Bool unique) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.allCandidateSampler in
  let inputs = [ (`single (P true_classes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let allToAll
    ?(name = "AllToAll")
    ~concat_dimension
    ~split_dimension
    ~split_count
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t)
    (group_assignment : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("concat_dimension", Int concat_dimension) :: attributes
  in
  let attributes =
    ("split_dimension", Int split_dimension) :: attributes
  in
  let attributes =
    ("split_count", Int split_count) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.allToAll in
  let inputs = [ (`single (P input)); (`single (P group_assignment)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let angle
    ?(name = "Angle")
    ~type_
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "Tout", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.angle in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let any
    ?(name = "Any")
    ?keep_dims
    ?(control_inputs = [])
    (input : [ `bool ] t)
    (reduction_indices : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type reduction_indices)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.any in
  let inputs = [ (`single (P input)); (`single (P reduction_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyAdaMax
    ?(name = "ApplyAdaMax")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (m : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (v : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta1_power : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyAdaMax in
  let inputs = [ (`single (P var)); (`single (P m)); (`single (P v)); (`single (P beta1_power)); (`single (P lr)); (`single (P beta1)); (`single (P beta2)); (`single (P epsilon)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyAdadelta
    ?(name = "ApplyAdadelta")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum_update : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (rho : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyAdadelta in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P accum_update)); (`single (P lr)); (`single (P rho)); (`single (P epsilon)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyAdagrad
    ?(name = "ApplyAdagrad")
    ?use_locking
    ?update_slots
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let attributes =
    match update_slots with | None -> attributes | Some update_slots -> ("update_slots", Bool update_slots) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyAdagrad in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P lr)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyAdagradDA
    ?(name = "ApplyAdagradDA")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (gradient_accumulator : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (gradient_squared_accumulator : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (global_step : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyAdagradDA in
  let inputs = [ (`single (P var)); (`single (P gradient_accumulator)); (`single (P gradient_squared_accumulator)); (`single (P grad)); (`single (P lr)); (`single (P l1)); (`single (P l2)); (`single (P global_step)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyAdam
    ?(name = "ApplyAdam")
    ?use_locking
    ?use_nesterov
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (m : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (v : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta1_power : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta2_power : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let attributes =
    match use_nesterov with | None -> attributes | Some use_nesterov -> ("use_nesterov", Bool use_nesterov) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyAdam in
  let inputs = [ (`single (P var)); (`single (P m)); (`single (P v)); (`single (P beta1_power)); (`single (P beta2_power)); (`single (P lr)); (`single (P beta1)); (`single (P beta2)); (`single (P epsilon)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyAddSign
    ?(name = "ApplyAddSign")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (m : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (alpha : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (sign_decay : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyAddSign in
  let inputs = [ (`single (P var)); (`single (P m)); (`single (P lr)); (`single (P alpha)); (`single (P sign_decay)); (`single (P beta)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyCenteredRMSProp
    ?(name = "ApplyCenteredRMSProp")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (mg : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (ms : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (mom : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (rho : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (momentum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyCenteredRMSProp in
  let inputs = [ (`single (P var)); (`single (P mg)); (`single (P ms)); (`single (P mom)); (`single (P lr)); (`single (P rho)); (`single (P momentum)); (`single (P epsilon)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyFtrl
    ?(name = "ApplyFtrl")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (linear : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr_power : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyFtrl in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P linear)); (`single (P grad)); (`single (P lr)); (`single (P l1)); (`single (P l2)); (`single (P lr_power)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyFtrlV2
    ?(name = "ApplyFtrlV2")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (linear : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2_shrinkage : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr_power : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyFtrlV2 in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P linear)); (`single (P grad)); (`single (P lr)); (`single (P l1)); (`single (P l2)); (`single (P l2_shrinkage)); (`single (P lr_power)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyGradientDescent
    ?(name = "ApplyGradientDescent")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (alpha : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (delta : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyGradientDescent in
  let inputs = [ (`single (P var)); (`single (P alpha)); (`single (P delta)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyMomentum
    ?(name = "ApplyMomentum")
    ?use_locking
    ?use_nesterov
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (momentum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let attributes =
    match use_nesterov with | None -> attributes | Some use_nesterov -> ("use_nesterov", Bool use_nesterov) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyMomentum in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P lr)); (`single (P grad)); (`single (P momentum)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyPowerSign
    ?(name = "ApplyPowerSign")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (m : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (logbase : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (sign_decay : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyPowerSign in
  let inputs = [ (`single (P var)); (`single (P m)); (`single (P lr)); (`single (P logbase)); (`single (P sign_decay)); (`single (P beta)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyProximalAdagrad
    ?(name = "ApplyProximalAdagrad")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyProximalAdagrad in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P lr)); (`single (P l1)); (`single (P l2)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyProximalGradientDescent
    ?(name = "ApplyProximalGradientDescent")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (alpha : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (delta : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyProximalGradientDescent in
  let inputs = [ (`single (P var)); (`single (P alpha)); (`single (P l1)); (`single (P l2)); (`single (P delta)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let applyRMSProp
    ?(name = "ApplyRMSProp")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (ms : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (mom : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (rho : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (momentum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.applyRMSProp in
  let inputs = [ (`single (P var)); (`single (P ms)); (`single (P mom)); (`single (P lr)); (`single (P rho)); (`single (P momentum)); (`single (P epsilon)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let approximateEqual
    ?(name = "ApproximateEqual")
    ?tolerance
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let attributes =
    match tolerance with | None -> attributes | Some tolerance -> ("tolerance", Float tolerance) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.approximateEqual in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let argMax
    ?(name = "ArgMax")
    ~type_
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (dimension : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type dimension)) ;  "T", Type (P (Node.output_type input)) ;  "output_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.argMax in
  let inputs = [ (`single (P input)); (`single (P dimension)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let argMin
    ?(name = "ArgMin")
    ~type_
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (dimension : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type dimension)) ;  "T", Type (P (Node.output_type input)) ;  "output_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.argMin in
  let inputs = [ (`single (P input)); (`single (P dimension)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let asString
    ?(name = "AsString")
    ?precision
    ?scientific
    ?shortest
    ?width
    ?fill
    ?(control_inputs = [])
    (input : ([< `int32 | `int64 | `complex64 | `float | `double | `bool ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match precision with | None -> attributes | Some precision -> ("precision", Int precision) :: attributes
  in
  let attributes =
    match scientific with | None -> attributes | Some scientific -> ("scientific", Bool scientific) :: attributes
  in
  let attributes =
    match shortest with | None -> attributes | Some shortest -> ("shortest", Bool shortest) :: attributes
  in
  let attributes =
    match width with | None -> attributes | Some width -> ("width", Int width) :: attributes
  in
  let attributes =
    match fill with | None -> attributes | Some fill -> ("fill", String fill) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.asString in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let asin
    ?(name = "Asin")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.asin in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let asinh
    ?(name = "Asinh")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.asinh in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let assign
    ?(name = "Assign")
    ?validate_shape
    ?use_locking
    ?(control_inputs = [])
    (ref : 't t)
    (value : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match validate_shape with | None -> attributes | Some validate_shape -> ("validate_shape", Bool validate_shape) :: attributes
  in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.assign in
  let inputs = [ (`single (P ref)); (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let assignAdd
    ?(name = "AssignAdd")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (value : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.assignAdd in
  let inputs = [ (`single (P ref)); (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let assignSub
    ?(name = "AssignSub")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (value : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.assignSub in
  let inputs = [ (`single (P ref)); (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let atan
    ?(name = "Atan")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.atan in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let atan2
    ?(name = "Atan2")
    ?(control_inputs = [])
    (y : ([< `float | `double ] as 't) t)
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type y)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.atan2 in
  let inputs = [ (`single (P y)); (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let atanh
    ?(name = "Atanh")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.atanh in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let audioSpectrogram
    ?(name = "AudioSpectrogram")
    ~window_size
    ~stride
    ?magnitude_squared
    ?(control_inputs = [])
    (input : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("window_size", Int window_size) :: attributes
  in
  let attributes =
    ("stride", Int stride) :: attributes
  in
  let attributes =
    match magnitude_squared with | None -> attributes | Some magnitude_squared -> ("magnitude_squared", Bool magnitude_squared) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.audioSpectrogram in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let audioSummary
    ?(name = "AudioSummary")
    ~sample_rate
    ?max_outputs
    ?(control_inputs = [])
    (tag : [ `string ] t)
    (tensor : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("sample_rate", Float sample_rate) :: attributes
  in
  let attributes =
    match max_outputs with | None -> attributes | Some max_outputs -> ("max_outputs", Int max_outputs) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.audioSummary in
  let inputs = [ (`single (P tag)); (`single (P tensor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let audioSummaryV2
    ?(name = "AudioSummaryV2")
    ?max_outputs
    ?(control_inputs = [])
    (tag : [ `string ] t)
    (tensor : [ `float ] t)
    (sample_rate : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match max_outputs with | None -> attributes | Some max_outputs -> ("max_outputs", Int max_outputs) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.audioSummaryV2 in
  let inputs = [ (`single (P tag)); (`single (P tensor)); (`single (P sample_rate)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let avgPool
    ?(name = "AvgPool")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (value : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.avgPool in
  let inputs = [ (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type value)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let avgPool3D
    ?(name = "AvgPool3D")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.avgPool3D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let avgPool3DGrad
    ?(name = "AvgPool3DGrad")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (orig_input_shape : [ `int32 ] t)
    (grad : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type grad)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.avgPool3DGrad in
  let inputs = [ (`single (P orig_input_shape)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type grad)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let avgPoolGrad
    ?(name = "AvgPoolGrad")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (orig_input_shape : [ `int32 ] t)
    (grad : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type grad)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.avgPoolGrad in
  let inputs = [ (`single (P orig_input_shape)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type grad)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let barrier
    ?(name = "Barrier")
    ~component_types
    ?shapes
    ?capacity
    ?container
    ?shared_name
    ?(control_inputs = [])
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
  let name = Name.of_string name in
  let op_name = Op_names.barrier in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let barrierClose
    ?(name = "BarrierClose")
    ?cancel_pending_enqueues
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match cancel_pending_enqueues with | None -> attributes | Some cancel_pending_enqueues -> ("cancel_pending_enqueues", Bool cancel_pending_enqueues) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.barrierClose in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let barrierIncompleteSize
    ?(name = "BarrierIncompleteSize")
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.barrierIncompleteSize in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let barrierInsertMany
    ?(name = "BarrierInsertMany")
    ~component_index
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (keys : [ `string ] t)
    (values : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type values)) ] in
  let attributes =
    ("component_index", Int component_index) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.barrierInsertMany in
  let inputs = [ (`single (P handle)); (`single (P keys)); (`single (P values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let barrierReadySize
    ?(name = "BarrierReadySize")
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.barrierReadySize in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchCholesky
    ?(name = "BatchCholesky")
    ?(control_inputs = [])
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.batchCholesky in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchCholeskyGrad
    ?(name = "BatchCholeskyGrad")
    ?(control_inputs = [])
    (l : ([< `float | `double ] as 't) t)
    (grad : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type l)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.batchCholeskyGrad in
  let inputs = [ (`single (P l)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type l)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchDataset
    ?(name = "BatchDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (batch_size : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P batch_size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchDatasetV2
    ?(name = "BatchDatasetV2")
    ?parallel_copy
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (batch_size : [ `int64 ] t)
    (drop_remainder : [ `bool ] t)
  =
  let attributes = [] in
  let attributes =
    match parallel_copy with | None -> attributes | Some parallel_copy -> ("parallel_copy", Bool parallel_copy) :: attributes
  in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchDatasetV2 in
  let inputs = [ (`single (P input_dataset)); (`single (P batch_size)); (`single (P drop_remainder)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchFFT
    ?(name = "BatchFFT")
    ?(control_inputs = [])
    (input : [ `complex64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.batchFFT in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Complex64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchFFT2D
    ?(name = "BatchFFT2D")
    ?(control_inputs = [])
    (input : [ `complex64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.batchFFT2D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Complex64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchFFT3D
    ?(name = "BatchFFT3D")
    ?(control_inputs = [])
    (input : [ `complex64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.batchFFT3D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Complex64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchIFFT
    ?(name = "BatchIFFT")
    ?(control_inputs = [])
    (input : [ `complex64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.batchIFFT in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Complex64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchIFFT2D
    ?(name = "BatchIFFT2D")
    ?(control_inputs = [])
    (input : [ `complex64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.batchIFFT2D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Complex64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchIFFT3D
    ?(name = "BatchIFFT3D")
    ?(control_inputs = [])
    (input : [ `complex64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.batchIFFT3D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Complex64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatMul
    ?(name = "BatchMatMul")
    ?adj_x
    ?adj_y
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let attributes =
    match adj_x with | None -> attributes | Some adj_x -> ("adj_x", Bool adj_x) :: attributes
  in
  let attributes =
    match adj_y with | None -> attributes | Some adj_y -> ("adj_y", Bool adj_y) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatMul in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatMulV2
    ?(name = "BatchMatMulV2")
    ?adj_x
    ?adj_y
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let attributes =
    match adj_x with | None -> attributes | Some adj_x -> ("adj_x", Bool adj_x) :: attributes
  in
  let attributes =
    match adj_y with | None -> attributes | Some adj_y -> ("adj_y", Bool adj_y) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatMulV2 in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatrixBandPart
    ?(name = "BatchMatrixBandPart")
    ?(control_inputs = [])
    (input : 't t)
    (num_lower : [ `int64 ] t)
    (num_upper : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatrixBandPart in
  let inputs = [ (`single (P input)); (`single (P num_lower)); (`single (P num_upper)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatrixDeterminant
    ?(name = "BatchMatrixDeterminant")
    ?(control_inputs = [])
    (input : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatrixDeterminant in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatrixDiag
    ?(name = "BatchMatrixDiag")
    ?(control_inputs = [])
    (diagonal : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type diagonal)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatrixDiag in
  let inputs = [ (`single (P diagonal)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type diagonal)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatrixDiagPart
    ?(name = "BatchMatrixDiagPart")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatrixDiagPart in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatrixInverse
    ?(name = "BatchMatrixInverse")
    ?adjoint
    ?(control_inputs = [])
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match adjoint with | None -> attributes | Some adjoint -> ("adjoint", Bool adjoint) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatrixInverse in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatrixSetDiag
    ?(name = "BatchMatrixSetDiag")
    ?(control_inputs = [])
    (input : 't t)
    (diagonal : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatrixSetDiag in
  let inputs = [ (`single (P input)); (`single (P diagonal)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatrixSolve
    ?(name = "BatchMatrixSolve")
    ?adjoint
    ?(control_inputs = [])
    (matrix : ([< `double | `float ] as 't) t)
    (rhs : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type matrix)) ] in
  let attributes =
    match adjoint with | None -> attributes | Some adjoint -> ("adjoint", Bool adjoint) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatrixSolve in
  let inputs = [ (`single (P matrix)); (`single (P rhs)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type matrix)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatrixSolveLs
    ?(name = "BatchMatrixSolveLs")
    ?fast
    ?(control_inputs = [])
    (matrix : ([< `double | `float ] as 't) t)
    (rhs : ([< `double | `float ] as 't) t)
    (l2_regularizer : [ `double ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type matrix)) ] in
  let attributes =
    match fast with | None -> attributes | Some fast -> ("fast", Bool fast) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatrixSolveLs in
  let inputs = [ (`single (P matrix)); (`single (P rhs)); (`single (P l2_regularizer)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type matrix)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchMatrixTriangularSolve
    ?(name = "BatchMatrixTriangularSolve")
    ?lower
    ?adjoint
    ?(control_inputs = [])
    (matrix : ([< `double | `float ] as 't) t)
    (rhs : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type matrix)) ] in
  let attributes =
    match lower with | None -> attributes | Some lower -> ("lower", Bool lower) :: attributes
  in
  let attributes =
    match adjoint with | None -> attributes | Some adjoint -> ("adjoint", Bool adjoint) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchMatrixTriangularSolve in
  let inputs = [ (`single (P matrix)); (`single (P rhs)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type matrix)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchNormWithGlobalNormalization
    ?(name = "BatchNormWithGlobalNormalization")
    ~variance_epsilon
    ~scale_after_normalization
    ?(control_inputs = [])
    (t : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (m : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (v : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (beta : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (gamma : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type t)) ] in
  let attributes =
    ("variance_epsilon", Float variance_epsilon) :: attributes
  in
  let attributes =
    ("scale_after_normalization", Bool scale_after_normalization) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchNormWithGlobalNormalization in
  let inputs = [ (`single (P t)); (`single (P m)); (`single (P v)); (`single (P beta)); (`single (P gamma)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type t)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchNormWithGlobalNormalizationGrad
    ?(name = "BatchNormWithGlobalNormalizationGrad")
    ~variance_epsilon
    ~scale_after_normalization
    ?(control_inputs = [])
    (t : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (m : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (v : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (gamma : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (backprop : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type t)) ;  "T", Type (P (Node.output_type t)) ;  "T", Type (P (Node.output_type t)) ;  "T", Type (P (Node.output_type t)) ;  "T", Type (P (Node.output_type t)) ] in
  let attributes =
    ("variance_epsilon", Float variance_epsilon) :: attributes
  in
  let attributes =
    ("scale_after_normalization", Bool scale_after_normalization) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchNormWithGlobalNormalizationGrad in
  let inputs = [ (`single (P t)); (`single (P m)); (`single (P v)); (`single (P gamma)); (`single (P backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type t)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type t)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type t)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type t)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type t)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 4)

let batchSelfAdjointEig
    ?(name = "BatchSelfAdjointEig")
    ?(control_inputs = [])
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.batchSelfAdjointEig in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchSelfAdjointEigV2
    ?(name = "BatchSelfAdjointEigV2")
    ?compute_v
    ?(control_inputs = [])
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match compute_v with | None -> attributes | Some compute_v -> ("compute_v", Bool compute_v) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchSelfAdjointEigV2 in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let batchSvd
    ?(name = "BatchSvd")
    ?compute_uv
    ?full_matrices
    ?(control_inputs = [])
    (input : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match compute_uv with | None -> attributes | Some compute_uv -> ("compute_uv", Bool compute_uv) :: attributes
  in
  let attributes =
    match full_matrices with | None -> attributes | Some full_matrices -> ("full_matrices", Bool full_matrices) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchSvd in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let batchToSpace
    ?(name = "BatchToSpace")
    ~block_size
    ?(control_inputs = [])
    (input : 't t)
    (crops : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type crops)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("block_size", Int block_size) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.batchToSpace in
  let inputs = [ (`single (P input)); (`single (P crops)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let batchToSpaceND
    ?(name = "BatchToSpaceND")
    ?(control_inputs = [])
    (input : 't t)
    (block_shape : ([< `int32 | `int64 ] as 'tblock_shape) t)
    (crops : ([< `int32 | `int64 ] as 'tcrops) t)
  =
  let attributes = [ "Tcrops", Type (P (Node.output_type crops)) ;  "Tblock_shape", Type (P (Node.output_type block_shape)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.batchToSpaceND in
  let inputs = [ (`single (P input)); (`single (P block_shape)); (`single (P crops)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let besselI0e
    ?(name = "BesselI0e")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.besselI0e in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let besselI1e
    ?(name = "BesselI1e")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.besselI1e in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let betainc
    ?(name = "Betainc")
    ?(control_inputs = [])
    (a : ([< `float | `double ] as 't) t)
    (b : ([< `float | `double ] as 't) t)
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type a)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.betainc in
  let inputs = [ (`single (P a)); (`single (P b)); (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let biasAdd
    ?(name = "BiasAdd")
    ?data_format
    ?(control_inputs = [])
    (value : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (bias : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.biasAdd in
  let inputs = [ (`single (P value)); (`single (P bias)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type value)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let biasAddGrad
    ?(name = "BiasAddGrad")
    ?data_format
    ?(control_inputs = [])
    (out_backprop : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type out_backprop)) ] in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.biasAddGrad in
  let inputs = [ (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type out_backprop)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let biasAddV1
    ?(name = "BiasAddV1")
    ?(control_inputs = [])
    (value : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (bias : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.biasAddV1 in
  let inputs = [ (`single (P value)); (`single (P bias)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type value)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let bincount
    ?(name = "Bincount")
    ?(control_inputs = [])
    (arr : [ `int32 ] t)
    (size : [ `int32 ] t)
    (weights : ([< `int32 | `int64 | `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type weights)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.bincount in
  let inputs = [ (`single (P arr)); (`single (P size)); (`single (P weights)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type weights)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let bitcast
    ?(name = "Bitcast")
    ~type_
    ?(control_inputs = [])
    (input : ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.bitcast in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let bitwiseAnd
    ?(name = "BitwiseAnd")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 ] as 't) t)
    (y : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.bitwiseAnd in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let bitwiseOr
    ?(name = "BitwiseOr")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 ] as 't) t)
    (y : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.bitwiseOr in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let bitwiseXor
    ?(name = "BitwiseXor")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 ] as 't) t)
    (y : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.bitwiseXor in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let boostedTreesAggregateStats
    ?(name = "BoostedTreesAggregateStats")
    ~max_splits
    ~num_buckets
    ?(control_inputs = [])
    (node_ids : [ `int32 ] t)
    (gradients : [ `float ] t)
    (hessians : [ `float ] t)
    (feature : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    ("max_splits", Int max_splits) :: attributes
  in
  let attributes =
    ("num_buckets", Int num_buckets) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.boostedTreesAggregateStats in
  let inputs = [ (`single (P node_ids)); (`single (P gradients)); (`single (P hessians)); (`single (P feature)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let boostedTreesBucketize
    ?(name = "BoostedTreesBucketize")
    ?(control_inputs = [])
    (float_values : [ `float ] t list)
    (bucket_boundaries : [ `float ] t list)
  =
  let attributes = [] in
  let attributes =
    ("num_features", Int (List.length float_values)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.boostedTreesBucketize in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) float_values)); (`multi (List.map ~f:(fun n -> P n) bucket_boundaries)) ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:Type.Int32
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init (List.length float_values) ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let boostedTreesCalculateBestFeatureSplit
    ?(name = "BoostedTreesCalculateBestFeatureSplit")
    ~logits_dimension
    ?split_type
    ?(control_inputs = [])
    (node_id_range : [ `int32 ] t)
    (stats_summary : [ `float ] t)
    (l1 : [ `float ] t)
    (l2 : [ `float ] t)
    (tree_complexity : [ `float ] t)
    (min_node_weight : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("logits_dimension", Int logits_dimension) :: attributes
  in
  let attributes =
    match split_type with | None -> attributes | Some split_type -> ("split_type", String split_type) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.boostedTreesCalculateBestFeatureSplit in
  let inputs = [ (`single (P node_id_range)); (`single (P stats_summary)); (`single (P l1)); (`single (P l2)); (`single (P tree_complexity)); (`single (P min_node_weight)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 4)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 5)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 6)

let boostedTreesMakeQuantileSummaries
    ?(name = "BoostedTreesMakeQuantileSummaries")
    ?(control_inputs = [])
    (float_values : [ `float ] t list)
    (example_weights : [ `float ] t)
    (epsilon : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_features", Int (List.length float_values)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.boostedTreesMakeQuantileSummaries in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) float_values)); (`single (P example_weights)); (`single (P epsilon)) ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:Type.Float
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init (List.length float_values) ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let boostedTreesMakeStatsSummary
    ?(name = "BoostedTreesMakeStatsSummary")
    ~max_splits
    ~num_buckets
    ?(control_inputs = [])
    (node_ids : [ `int32 ] t)
    (gradients : [ `float ] t)
    (hessians : [ `float ] t)
    (bucketized_features_list : [ `int32 ] t list)
  =
  let attributes = [] in
  let attributes =
    ("max_splits", Int max_splits) :: attributes
  in
  let attributes =
    ("num_buckets", Int num_buckets) :: attributes
  in
  let attributes =
    ("num_features", Int (List.length bucketized_features_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.boostedTreesMakeStatsSummary in
  let inputs = [ (`single (P node_ids)); (`single (P gradients)); (`single (P hessians)); (`multi (List.map ~f:(fun n -> P n) bucketized_features_list)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let broadcastArgs
    ?(name = "BroadcastArgs")
    ?(control_inputs = [])
    (s0 : ([< `int32 | `int64 ] as 't) t)
    (s1 : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type s0)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.broadcastArgs in
  let inputs = [ (`single (P s0)); (`single (P s1)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type s0)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let broadcastGradientArgs
    ?(name = "BroadcastGradientArgs")
    ?(control_inputs = [])
    (s0 : ([< `int32 | `int64 ] as 't) t)
    (s1 : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type s0)) ;  "T", Type (P (Node.output_type s0)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.broadcastGradientArgs in
  let inputs = [ (`single (P s0)); (`single (P s1)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type s0)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type s0)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let broadcastTo
    ?(name = "BroadcastTo")
    ?(control_inputs = [])
    (input : 't t)
    (shape : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type shape)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.broadcastTo in
  let inputs = [ (`single (P input)); (`single (P shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let bucketize
    ?(name = "Bucketize")
    ~boundaries
    ?(control_inputs = [])
    (input : ([< `int32 | `int64 | `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("boundaries", List (Float boundaries)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.bucketize in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cTCGreedyDecoder
    ?(name = "CTCGreedyDecoder")
    ?merge_repeated
    ?(control_inputs = [])
    (inputs__ : [ `float ] t)
    (sequence_length : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match merge_repeated with | None -> attributes | Some merge_repeated -> ("merge_repeated", Bool merge_repeated) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cTCGreedyDecoder in
  let inputs = [ (`single (P inputs__)); (`single (P sequence_length)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let cTCLoss
    ?(name = "CTCLoss")
    ?preprocess_collapse_repeated
    ?ctc_merge_repeated
    ?ignore_longer_outputs_than_inputs
    ?(control_inputs = [])
    (inputs__ : [ `float ] t)
    (labels_indices : [ `int64 ] t)
    (labels_values : [ `int32 ] t)
    (sequence_length : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match preprocess_collapse_repeated with | None -> attributes | Some preprocess_collapse_repeated -> ("preprocess_collapse_repeated", Bool preprocess_collapse_repeated) :: attributes
  in
  let attributes =
    match ctc_merge_repeated with | None -> attributes | Some ctc_merge_repeated -> ("ctc_merge_repeated", Bool ctc_merge_repeated) :: attributes
  in
  let attributes =
    match ignore_longer_outputs_than_inputs with | None -> attributes | Some ignore_longer_outputs_than_inputs -> ("ignore_longer_outputs_than_inputs", Bool ignore_longer_outputs_than_inputs) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cTCLoss in
  let inputs = [ (`single (P inputs__)); (`single (P labels_indices)); (`single (P labels_values)); (`single (P sequence_length)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let cacheDataset
    ?(name = "CacheDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (filename : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cacheDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P filename)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cast
    ?(name = "Cast")
    ~type_
    ?truncate
    ?(control_inputs = [])
    (x : 'srcT t)
  =
  let attributes = [ "SrcT", Type (P (Node.output_type x)) ;  "DstT", Type (P type_) ] in
  let attributes =
    match truncate with | None -> attributes | Some truncate -> ("Truncate", Bool truncate) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cast in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let ceil
    ?(name = "Ceil")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.ceil in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let checkNumerics
    ?(name = "CheckNumerics")
    ~message
    ?(control_inputs = [])
    (tensor : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type tensor)) ] in
  let attributes =
    ("message", String message) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.checkNumerics in
  let inputs = [ (`single (P tensor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type tensor)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cholesky
    ?(name = "Cholesky")
    ?(control_inputs = [])
    (input : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.cholesky in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let choleskyGrad
    ?(name = "CholeskyGrad")
    ?(control_inputs = [])
    (l : ([< `float | `double ] as 't) t)
    (grad : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type l)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.choleskyGrad in
  let inputs = [ (`single (P l)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type l)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let clipByValue
    ?(name = "ClipByValue")
    ?(control_inputs = [])
    (t : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (clip_value_min : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (clip_value_max : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type t)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.clipByValue in
  let inputs = [ (`single (P t)); (`single (P clip_value_min)); (`single (P clip_value_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type t)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let collectiveBcastRecv
    ?(name = "CollectiveBcastRecv")
    ~type_
    ~group_size
    ~group_key
    ~instance_key
    ~shape
    ?(control_inputs = [])
    ()
  =
  let attributes = [ "T", Type (P type_) ] in
  let attributes =
    ("group_size", Int group_size) :: attributes
  in
  let attributes =
    ("group_key", Int group_key) :: attributes
  in
  let attributes =
    ("instance_key", Int instance_key) :: attributes
  in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.collectiveBcastRecv in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let collectiveBcastSend
    ?(name = "CollectiveBcastSend")
    ~group_size
    ~group_key
    ~instance_key
    ~shape
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("group_size", Int group_size) :: attributes
  in
  let attributes =
    ("group_key", Int group_key) :: attributes
  in
  let attributes =
    ("instance_key", Int instance_key) :: attributes
  in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.collectiveBcastSend in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let collectiveGather
    ?(name = "CollectiveGather")
    ~group_size
    ~group_key
    ~instance_key
    ~shape
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("group_size", Int group_size) :: attributes
  in
  let attributes =
    ("group_key", Int group_key) :: attributes
  in
  let attributes =
    ("instance_key", Int instance_key) :: attributes
  in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.collectiveGather in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let collectivePermute
    ?(name = "CollectivePermute")
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (source_target_pairs : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.collectivePermute in
  let inputs = [ (`single (P input)); (`single (P source_target_pairs)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let collectiveReduce
    ?(name = "CollectiveReduce")
    ~group_size
    ~group_key
    ~instance_key
    ~merge_op
    ~final_op
    ~subdiv_offsets
    ?wait_for
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("group_size", Int group_size) :: attributes
  in
  let attributes =
    ("group_key", Int group_key) :: attributes
  in
  let attributes =
    ("instance_key", Int instance_key) :: attributes
  in
  let attributes =
    ("merge_op", String merge_op) :: attributes
  in
  let attributes =
    ("final_op", String final_op) :: attributes
  in
  let attributes =
    ("subdiv_offsets", List (Int subdiv_offsets)) :: attributes
  in
  let attributes =
    match wait_for with | None -> attributes | Some wait_for -> ("wait_for", List (Int wait_for)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.collectiveReduce in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let combinedNonMaxSuppression
    ?(name = "CombinedNonMaxSuppression")
    ?pad_per_class
    ?clip_boxes
    ?(control_inputs = [])
    (boxes : [ `float ] t)
    (scores : [ `float ] t)
    (max_output_size_per_class : [ `int32 ] t)
    (max_total_size : [ `int32 ] t)
    (iou_threshold : [ `float ] t)
    (score_threshold : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match pad_per_class with | None -> attributes | Some pad_per_class -> ("pad_per_class", Bool pad_per_class) :: attributes
  in
  let attributes =
    match clip_boxes with | None -> attributes | Some clip_boxes -> ("clip_boxes", Bool clip_boxes) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.combinedNonMaxSuppression in
  let inputs = [ (`single (P boxes)); (`single (P scores)); (`single (P max_output_size_per_class)); (`single (P max_total_size)); (`single (P iou_threshold)); (`single (P score_threshold)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let complex
    ?(name = "Complex")
    ~type_
    ?(control_inputs = [])
    (real : ([< `float | `double ] as 't) t)
    (imag : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type real)) ;  "Tout", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.complex in
  let inputs = [ (`single (P real)); (`single (P imag)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let complexAbs
    ?(name = "ComplexAbs")
    ~type_
    ?(control_inputs = [])
    (x : ([< `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ;  "Tout", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.complexAbs in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let computeAccidentalHits
    ?(name = "ComputeAccidentalHits")
    ~num_true
    ?seed
    ?seed2
    ?(control_inputs = [])
    (true_classes : [ `int64 ] t)
    (sampled_candidates : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_true", Int num_true) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.computeAccidentalHits in
  let inputs = [ (`single (P true_classes)); (`single (P sampled_candidates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let concat
    ?(name = "Concat")
    ?(control_inputs = [])
    (concat_dim : [ `int32 ] t)
    (values : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn values))) ] in
  let attributes =
    ("N", Int (List.length values)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.concat in
  let inputs = [ (`single (P concat_dim)); (`multi (List.map ~f:(fun n -> P n) values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn values))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let concatOffset
    ?(name = "ConcatOffset")
    ?(control_inputs = [])
    (concat_dim : [ `int32 ] t)
    (shape : [ `int32 ] t list)
  =
  let attributes = [] in
  let attributes =
    ("N", Int (List.length shape)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.concatOffset in
  let inputs = [ (`single (P concat_dim)); (`multi (List.map ~f:(fun n -> P n) shape)) ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:Type.Int32
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init (List.length shape) ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let concatV2
    ?(name = "ConcatV2")
    ?(control_inputs = [])
    (values : 't t list)
    (axis : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type axis)) ;  "T", Type (P (Node.output_type (List.hd_exn values))) ] in
  let attributes =
    ("N", Int (List.length values)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.concatV2 in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) values)); (`single (P axis)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn values))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let concatenateDataset
    ?(name = "ConcatenateDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (another_dataset : [ `variant ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.concatenateDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P another_dataset)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conditionalAccumulator
    ?(name = "ConditionalAccumulator")
    ~shape
    ?container
    ?shared_name
    ?reduction_type
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let attributes =
    match reduction_type with | None -> attributes | Some reduction_type -> ("reduction_type", String reduction_type) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.conditionalAccumulator in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let configureDistributedTPU
    ?(name = "ConfigureDistributedTPU")
    ?embedding_config
    ?tpu_embedding_config
    ?is_global_init
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match embedding_config with | None -> attributes | Some embedding_config -> ("embedding_config", String embedding_config) :: attributes
  in
  let attributes =
    match tpu_embedding_config with | None -> attributes | Some tpu_embedding_config -> ("tpu_embedding_config", String tpu_embedding_config) :: attributes
  in
  let attributes =
    match is_global_init with | None -> attributes | Some is_global_init -> ("is_global_init", Bool is_global_init) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.configureDistributedTPU in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conj
    ?(name = "Conj")
    ?(control_inputs = [])
    (input : ([< `complex64 | `variant ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.conj in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conjugateTranspose
    ?(name = "ConjugateTranspose")
    ?(control_inputs = [])
    (x : 't t)
    (perm : ([< `int32 | `int64 ] as 'tperm) t)
  =
  let attributes = [ "Tperm", Type (P (Node.output_type perm)) ;  "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.conjugateTranspose in
  let inputs = [ (`single (P x)); (`single (P perm)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let consumeMutexLock
    ?(name = "ConsumeMutexLock")
    ?(control_inputs = [])
    (mutex_lock : [ `variant ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.consumeMutexLock in
  let inputs = [ (`single (P mutex_lock)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let controlTrigger
    ?(name = "ControlTrigger")
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.controlTrigger in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conv2D
    ?(name = "Conv2D")
    ~strides
    ?use_cudnn_on_gpu
    ~padding
    ?explicit_paddings
    ?data_format
    ?dilations
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
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
    match explicit_paddings with | None -> attributes | Some explicit_paddings -> ("explicit_paddings", List (Int explicit_paddings)) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.conv2D in
  let inputs = [ (`single (P input)); (`single (P filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conv2DBackpropFilter
    ?(name = "Conv2DBackpropFilter")
    ~strides
    ?use_cudnn_on_gpu
    ~padding
    ?explicit_paddings
    ?data_format
    ?dilations
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (filter_sizes : [ `int32 ] t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
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
    match explicit_paddings with | None -> attributes | Some explicit_paddings -> ("explicit_paddings", List (Int explicit_paddings)) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.conv2DBackpropFilter in
  let inputs = [ (`single (P input)); (`single (P filter_sizes)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conv2DBackpropInput
    ?(name = "Conv2DBackpropInput")
    ~strides
    ?use_cudnn_on_gpu
    ~padding
    ?explicit_paddings
    ?data_format
    ?dilations
    ?(control_inputs = [])
    (input_sizes : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type filter)) ] in
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
    match explicit_paddings with | None -> attributes | Some explicit_paddings -> ("explicit_paddings", List (Int explicit_paddings)) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.conv2DBackpropInput in
  let inputs = [ (`single (P input_sizes)); (`single (P filter)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type filter)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conv3D
    ?(name = "Conv3D")
    ~strides
    ~padding
    ?data_format
    ?dilations
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.conv3D in
  let inputs = [ (`single (P input)); (`single (P filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conv3DBackpropFilter
    ?(name = "Conv3DBackpropFilter")
    ~strides
    ~padding
    ?dilations
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.conv3DBackpropFilter in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conv3DBackpropFilterV2
    ?(name = "Conv3DBackpropFilterV2")
    ~strides
    ~padding
    ?data_format
    ?dilations
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (filter_sizes : [ `int32 ] t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.conv3DBackpropFilterV2 in
  let inputs = [ (`single (P input)); (`single (P filter_sizes)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conv3DBackpropInput
    ?(name = "Conv3DBackpropInput")
    ~strides
    ~padding
    ?dilations
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.conv3DBackpropInput in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let conv3DBackpropInputV2
    ?(name = "Conv3DBackpropInputV2")
    ~strides
    ~padding
    ?data_format
    ?dilations
    ?(control_inputs = [])
    (input_sizes : ([< `int32 | `int64 ] as 'tshape) t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "Tshape", Type (P (Node.output_type input_sizes)) ;  "T", Type (P (Node.output_type filter)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.conv3DBackpropInputV2 in
  let inputs = [ (`single (P input_sizes)); (`single (P filter)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type filter)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let copy
    ?(name = "Copy")
    ?tensor_name
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match tensor_name with | None -> attributes | Some tensor_name -> ("tensor_name", String tensor_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.copy in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let copyHost
    ?(name = "CopyHost")
    ?tensor_name
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match tensor_name with | None -> attributes | Some tensor_name -> ("tensor_name", String tensor_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.copyHost in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cos
    ?(name = "Cos")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.cos in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cosh
    ?(name = "Cosh")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.cosh in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let countUpTo
    ?(name = "CountUpTo")
    ~limit
    ?(control_inputs = [])
    (ref : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    ("limit", Int limit) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.countUpTo in
  let inputs = [ (`single (P ref)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cropAndResize
    ?(name = "CropAndResize")
    ?method_
    ?extrapolation_value
    ?(control_inputs = [])
    (image : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (boxes : [ `float ] t)
    (box_ind : [ `int32 ] t)
    (crop_size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type image)) ] in
  let attributes =
    match method_ with | None -> attributes | Some method_ -> ("method", String method_) :: attributes
  in
  let attributes =
    match extrapolation_value with | None -> attributes | Some extrapolation_value -> ("extrapolation_value", Float extrapolation_value) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cropAndResize in
  let inputs = [ (`single (P image)); (`single (P boxes)); (`single (P box_ind)); (`single (P crop_size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cropAndResizeGradBoxes
    ?(name = "CropAndResizeGradBoxes")
    ?method_
    ?(control_inputs = [])
    (grads : [ `float ] t)
    (image : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (boxes : [ `float ] t)
    (box_ind : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type image)) ] in
  let attributes =
    match method_ with | None -> attributes | Some method_ -> ("method", String method_) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cropAndResizeGradBoxes in
  let inputs = [ (`single (P grads)); (`single (P image)); (`single (P boxes)); (`single (P box_ind)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cropAndResizeGradImage
    ?(name = "CropAndResizeGradImage")
    ~type_
    ?method_
    ?(control_inputs = [])
    (grads : [ `float ] t)
    (boxes : [ `float ] t)
    (box_ind : [ `int32 ] t)
    (image_size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P type_) ] in
  let attributes =
    match method_ with | None -> attributes | Some method_ -> ("method", String method_) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cropAndResizeGradImage in
  let inputs = [ (`single (P grads)); (`single (P boxes)); (`single (P box_ind)); (`single (P image_size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cross
    ?(name = "Cross")
    ?(control_inputs = [])
    (a : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (b : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type a)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.cross in
  let inputs = [ (`single (P a)); (`single (P b)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let crossReplicaSum
    ?(name = "CrossReplicaSum")
    ?(control_inputs = [])
    (input : ([< `float | `int32 ] as 't) t)
    (group_assignment : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.crossReplicaSum in
  let inputs = [ (`single (P input)); (`single (P group_assignment)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cudnnRNN
    ?(name = "CudnnRNN")
    ?rnn_mode
    ?input_mode
    ?direction
    ?dropout
    ?seed
    ?seed2
    ?is_training
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (input_h : ([< `float | `double ] as 't) t)
    (input_c : ([< `float | `double ] as 't) t)
    (params : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match rnn_mode with | None -> attributes | Some rnn_mode -> ("rnn_mode", String rnn_mode) :: attributes
  in
  let attributes =
    match input_mode with | None -> attributes | Some input_mode -> ("input_mode", String input_mode) :: attributes
  in
  let attributes =
    match direction with | None -> attributes | Some direction -> ("direction", String direction) :: attributes
  in
  let attributes =
    match dropout with | None -> attributes | Some dropout -> ("dropout", Float dropout) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let attributes =
    match is_training with | None -> attributes | Some is_training -> ("is_training", Bool is_training) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cudnnRNN in
  let inputs = [ (`single (P input)); (`single (P input_h)); (`single (P input_c)); (`single (P params)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let cudnnRNNBackprop
    ?(name = "CudnnRNNBackprop")
    ?rnn_mode
    ?input_mode
    ?direction
    ?dropout
    ?seed
    ?seed2
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (input_h : ([< `float | `double ] as 't) t)
    (input_c : ([< `float | `double ] as 't) t)
    (params : ([< `float | `double ] as 't) t)
    (output : ([< `float | `double ] as 't) t)
    (output_h : ([< `float | `double ] as 't) t)
    (output_c : ([< `float | `double ] as 't) t)
    (output_backprop : ([< `float | `double ] as 't) t)
    (output_h_backprop : ([< `float | `double ] as 't) t)
    (output_c_backprop : ([< `float | `double ] as 't) t)
    (reserve_space : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match rnn_mode with | None -> attributes | Some rnn_mode -> ("rnn_mode", String rnn_mode) :: attributes
  in
  let attributes =
    match input_mode with | None -> attributes | Some input_mode -> ("input_mode", String input_mode) :: attributes
  in
  let attributes =
    match direction with | None -> attributes | Some direction -> ("direction", String direction) :: attributes
  in
  let attributes =
    match dropout with | None -> attributes | Some dropout -> ("dropout", Float dropout) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cudnnRNNBackprop in
  let inputs = [ (`single (P input)); (`single (P input_h)); (`single (P input_c)); (`single (P params)); (`single (P output)); (`single (P output_h)); (`single (P output_c)); (`single (P output_backprop)); (`single (P output_h_backprop)); (`single (P output_c_backprop)); (`single (P reserve_space)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let cudnnRNNCanonicalToParams
    ?(name = "CudnnRNNCanonicalToParams")
    ?rnn_mode
    ?input_mode
    ?direction
    ?dropout
    ?seed
    ?seed2
    ?(control_inputs = [])
    (num_layers : [ `int32 ] t)
    (num_units : [ `int32 ] t)
    (input_size : [ `int32 ] t)
    (weights : ([< `float | `double ] as 't) t list)
    (biases : ([< `float | `double ] as 't) t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn weights))) ] in
  let attributes =
    ("num_params", Int (List.length weights)) :: attributes
  in
  let attributes =
    match rnn_mode with | None -> attributes | Some rnn_mode -> ("rnn_mode", String rnn_mode) :: attributes
  in
  let attributes =
    match input_mode with | None -> attributes | Some input_mode -> ("input_mode", String input_mode) :: attributes
  in
  let attributes =
    match direction with | None -> attributes | Some direction -> ("direction", String direction) :: attributes
  in
  let attributes =
    match dropout with | None -> attributes | Some dropout -> ("dropout", Float dropout) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cudnnRNNCanonicalToParams in
  let inputs = [ (`single (P num_layers)); (`single (P num_units)); (`single (P input_size)); (`multi (List.map ~f:(fun n -> P n) weights)); (`multi (List.map ~f:(fun n -> P n) biases)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn weights))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cudnnRNNParamsSize
    ?(name = "CudnnRNNParamsSize")
    ~type_
    ?rnn_mode
    ?input_mode
    ?direction
    ?dropout
    ?seed
    ?seed2
    ?(control_inputs = [])
    (num_layers : [ `int32 ] t)
    (num_units : [ `int32 ] t)
    (input_size : [ `int32 ] t)
  =
  let attributes = [ "S", Type (P type_) ] in
  let attributes =
    match rnn_mode with | None -> attributes | Some rnn_mode -> ("rnn_mode", String rnn_mode) :: attributes
  in
  let attributes =
    match input_mode with | None -> attributes | Some input_mode -> ("input_mode", String input_mode) :: attributes
  in
  let attributes =
    match direction with | None -> attributes | Some direction -> ("direction", String direction) :: attributes
  in
  let attributes =
    match dropout with | None -> attributes | Some dropout -> ("dropout", Float dropout) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cudnnRNNParamsSize in
  let inputs = [ (`single (P num_layers)); (`single (P num_units)); (`single (P input_size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cumprod
    ?(name = "Cumprod")
    ?exclusive
    ?reverse
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (axis : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type axis)) ;  "T", Type (P (Node.output_type x)) ] in
  let attributes =
    match exclusive with | None -> attributes | Some exclusive -> ("exclusive", Bool exclusive) :: attributes
  in
  let attributes =
    match reverse with | None -> attributes | Some reverse -> ("reverse", Bool reverse) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cumprod in
  let inputs = [ (`single (P x)); (`single (P axis)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let cumsum
    ?(name = "Cumsum")
    ?exclusive
    ?reverse
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (axis : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type axis)) ;  "T", Type (P (Node.output_type x)) ] in
  let attributes =
    match exclusive with | None -> attributes | Some exclusive -> ("exclusive", Bool exclusive) :: attributes
  in
  let attributes =
    match reverse with | None -> attributes | Some reverse -> ("reverse", Bool reverse) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.cumsum in
  let inputs = [ (`single (P x)); (`single (P axis)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let dataFormatDimMap
    ?(name = "DataFormatDimMap")
    ?src_format
    ?dst_format
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let attributes =
    match src_format with | None -> attributes | Some src_format -> ("src_format", String src_format) :: attributes
  in
  let attributes =
    match dst_format with | None -> attributes | Some dst_format -> ("dst_format", String dst_format) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.dataFormatDimMap in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let dataFormatVecPermute
    ?(name = "DataFormatVecPermute")
    ?src_format
    ?dst_format
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let attributes =
    match src_format with | None -> attributes | Some src_format -> ("src_format", String src_format) :: attributes
  in
  let attributes =
    match dst_format with | None -> attributes | Some dst_format -> ("dst_format", String dst_format) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.dataFormatVecPermute in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let datasetToGraph
    ?(name = "DatasetToGraph")
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.datasetToGraph in
  let inputs = [ (`single (P input_dataset)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let debugGradientIdentity
    ?(name = "DebugGradientIdentity")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.debugGradientIdentity in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let debugGradientRefIdentity
    ?(name = "DebugGradientRefIdentity")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.debugGradientRefIdentity in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let debugIdentity
    ?(name = "DebugIdentity")
    ?device_name
    ?tensor_name
    ?gated_grpc
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match device_name with | None -> attributes | Some device_name -> ("device_name", String device_name) :: attributes
  in
  let attributes =
    match tensor_name with | None -> attributes | Some tensor_name -> ("tensor_name", String tensor_name) :: attributes
  in
  let attributes =
    match gated_grpc with | None -> attributes | Some gated_grpc -> ("gated_grpc", Bool gated_grpc) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.debugIdentity in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let debugNanCount
    ?(name = "DebugNanCount")
    ?device_name
    ?tensor_name
    ?gated_grpc
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match device_name with | None -> attributes | Some device_name -> ("device_name", String device_name) :: attributes
  in
  let attributes =
    match tensor_name with | None -> attributes | Some tensor_name -> ("tensor_name", String tensor_name) :: attributes
  in
  let attributes =
    match gated_grpc with | None -> attributes | Some gated_grpc -> ("gated_grpc", Bool gated_grpc) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.debugNanCount in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let debugNumericSummary
    ?(name = "DebugNumericSummary")
    ?device_name
    ?tensor_name
    ?lower_bound
    ?upper_bound
    ?mute_if_healthy
    ?gated_grpc
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match device_name with | None -> attributes | Some device_name -> ("device_name", String device_name) :: attributes
  in
  let attributes =
    match tensor_name with | None -> attributes | Some tensor_name -> ("tensor_name", String tensor_name) :: attributes
  in
  let attributes =
    match lower_bound with | None -> attributes | Some lower_bound -> ("lower_bound", Float lower_bound) :: attributes
  in
  let attributes =
    match upper_bound with | None -> attributes | Some upper_bound -> ("upper_bound", Float upper_bound) :: attributes
  in
  let attributes =
    match mute_if_healthy with | None -> attributes | Some mute_if_healthy -> ("mute_if_healthy", Bool mute_if_healthy) :: attributes
  in
  let attributes =
    match gated_grpc with | None -> attributes | Some gated_grpc -> ("gated_grpc", Bool gated_grpc) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.debugNumericSummary in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Double
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let decodeBase64
    ?(name = "DecodeBase64")
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.decodeBase64 in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let decodeCompressed
    ?(name = "DecodeCompressed")
    ?compression_type
    ?(control_inputs = [])
    (bytes : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match compression_type with | None -> attributes | Some compression_type -> ("compression_type", String compression_type) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.decodeCompressed in
  let inputs = [ (`single (P bytes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let decodeJSONExample
    ?(name = "DecodeJSONExample")
    ?(control_inputs = [])
    (json_examples : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.decodeJSONExample in
  let inputs = [ (`single (P json_examples)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let decodePaddedRaw
    ?(name = "DecodePaddedRaw")
    ~type_
    ?little_endian
    ?(control_inputs = [])
    (input_bytes : [ `string ] t)
    (fixed_length : [ `int32 ] t)
  =
  let attributes = [ "out_type", Type (P type_) ] in
  let attributes =
    match little_endian with | None -> attributes | Some little_endian -> ("little_endian", Bool little_endian) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.decodePaddedRaw in
  let inputs = [ (`single (P input_bytes)); (`single (P fixed_length)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let decodePng
    ?(name = "DecodePng")
    ~type_
    ?channels
    ?(control_inputs = [])
    (contents : [ `string ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match channels with | None -> attributes | Some channels -> ("channels", Int channels) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.decodePng in
  let inputs = [ (`single (P contents)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let decodeRaw
    ?(name = "DecodeRaw")
    ~type_
    ?little_endian
    ?(control_inputs = [])
    (bytes : [ `string ] t)
  =
  let attributes = [ "out_type", Type (P type_) ] in
  let attributes =
    match little_endian with | None -> attributes | Some little_endian -> ("little_endian", Bool little_endian) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.decodeRaw in
  let inputs = [ (`single (P bytes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let decodeWav
    ?(name = "DecodeWav")
    ?desired_channels
    ?desired_samples
    ?(control_inputs = [])
    (contents : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match desired_channels with | None -> attributes | Some desired_channels -> ("desired_channels", Int desired_channels) :: attributes
  in
  let attributes =
    match desired_samples with | None -> attributes | Some desired_samples -> ("desired_samples", Int desired_samples) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.decodeWav in
  let inputs = [ (`single (P contents)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let deepCopy
    ?(name = "DeepCopy")
    ?(control_inputs = [])
    (x : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.deepCopy in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let deleteSessionTensor
    ?(name = "DeleteSessionTensor")
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.deleteSessionTensor in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let denseToDenseSetOperation
    ?(name = "DenseToDenseSetOperation")
    ~set_operation
    ?validate_indices
    ?(control_inputs = [])
    (set1 : ([< `int32 | `int64 | `string ] as 't) t)
    (set2 : ([< `int32 | `int64 | `string ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type set1)) ] in
  let attributes =
    ("set_operation", String set_operation) :: attributes
  in
  let attributes =
    match validate_indices with | None -> attributes | Some validate_indices -> ("validate_indices", Bool validate_indices) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.denseToDenseSetOperation in
  let inputs = [ (`single (P set1)); (`single (P set2)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type set1)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let denseToSparseSetOperation
    ?(name = "DenseToSparseSetOperation")
    ~set_operation
    ?validate_indices
    ?(control_inputs = [])
    (set1 : ([< `int32 | `int64 | `string ] as 't) t)
    (set2_indices : [ `int64 ] t)
    (set2_values : ([< `int32 | `int64 | `string ] as 't) t)
    (set2_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type set1)) ] in
  let attributes =
    ("set_operation", String set_operation) :: attributes
  in
  let attributes =
    match validate_indices with | None -> attributes | Some validate_indices -> ("validate_indices", Bool validate_indices) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.denseToSparseSetOperation in
  let inputs = [ (`single (P set1)); (`single (P set2_indices)); (`single (P set2_values)); (`single (P set2_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type set1)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let depthToSpace
    ?(name = "DepthToSpace")
    ~block_size
    ?data_format
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("block_size", Int block_size) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.depthToSpace in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let depthwiseConv2dNative
    ?(name = "DepthwiseConv2dNative")
    ~strides
    ~padding
    ?data_format
    ?dilations
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (filter : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.depthwiseConv2dNative in
  let inputs = [ (`single (P input)); (`single (P filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let depthwiseConv2dNativeBackpropFilter
    ?(name = "DepthwiseConv2dNativeBackpropFilter")
    ~strides
    ~padding
    ?data_format
    ?dilations
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (filter_sizes : [ `int32 ] t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.depthwiseConv2dNativeBackpropFilter in
  let inputs = [ (`single (P input)); (`single (P filter_sizes)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let depthwiseConv2dNativeBackpropInput
    ?(name = "DepthwiseConv2dNativeBackpropInput")
    ~strides
    ~padding
    ?data_format
    ?dilations
    ?(control_inputs = [])
    (input_sizes : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
    (out_backprop : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type filter)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.depthwiseConv2dNativeBackpropInput in
  let inputs = [ (`single (P input_sizes)); (`single (P filter)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type filter)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let dequantize
    ?(name = "Dequantize")
    ?mode
    ?(control_inputs = [])
    (input : 't t)
    (min_range : [ `float ] t)
    (max_range : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match mode with | None -> attributes | Some mode -> ("mode", String mode) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.dequantize in
  let inputs = [ (`single (P input)); (`single (P min_range)); (`single (P max_range)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let deserializeManySparse
    ?(name = "DeserializeManySparse")
    ~type_1
    ?(control_inputs = [])
    (serialized_sparse : [ `string ] t)
  =
  let attributes = [ "dtype", Type (P type_1) ] in
  let name = Name.of_string name in
  let op_name = Op_names.deserializeManySparse in
  let inputs = [ (`single (P serialized_sparse)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let deserializeSparse
    ?(name = "DeserializeSparse")
    ~type_1
    ?(control_inputs = [])
    (serialized_sparse : ([< `string | `variant ] as 'tserialized) t)
  =
  let attributes = [ "Tserialized", Type (P (Node.output_type serialized_sparse)) ;  "dtype", Type (P type_1) ] in
  let name = Name.of_string name in
  let op_name = Op_names.deserializeSparse in
  let inputs = [ (`single (P serialized_sparse)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let destroyTemporaryVariable
    ?(name = "DestroyTemporaryVariable")
    ~var_name
    ?(control_inputs = [])
    (ref : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    ("var_name", String var_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.destroyTemporaryVariable in
  let inputs = [ (`single (P ref)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let diag
    ?(name = "Diag")
    ?(control_inputs = [])
    (diagonal : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type diagonal)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.diag in
  let inputs = [ (`single (P diagonal)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type diagonal)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let diagPart
    ?(name = "DiagPart")
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.diagPart in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let digamma
    ?(name = "Digamma")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.digamma in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let dilation2D
    ?(name = "Dilation2D")
    ~strides
    ~rates
    ~padding
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (filter : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("rates", List (Int rates)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.dilation2D in
  let inputs = [ (`single (P input)); (`single (P filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let dilation2DBackpropFilter
    ?(name = "Dilation2DBackpropFilter")
    ~strides
    ~rates
    ~padding
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (filter : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (out_backprop : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("rates", List (Int rates)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.dilation2DBackpropFilter in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let dilation2DBackpropInput
    ?(name = "Dilation2DBackpropInput")
    ~strides
    ~rates
    ~padding
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (filter : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (out_backprop : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("rates", List (Int rates)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.dilation2DBackpropInput in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P out_backprop)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let div
    ?(name = "Div")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.div in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let divNoNan
    ?(name = "DivNoNan")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
    (y : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.divNoNan in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let drawBoundingBoxes
    ?(name = "DrawBoundingBoxes")
    ?(control_inputs = [])
    (images : ([< `float ] as 't) t)
    (boxes : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.drawBoundingBoxes in
  let inputs = [ (`single (P images)); (`single (P boxes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let drawBoundingBoxesV2
    ?(name = "DrawBoundingBoxesV2")
    ?(control_inputs = [])
    (images : ([< `float ] as 't) t)
    (boxes : [ `float ] t)
    (colors : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.drawBoundingBoxesV2 in
  let inputs = [ (`single (P images)); (`single (P boxes)); (`single (P colors)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let dynamicPartition
    ?(name = "DynamicPartition")
    ~num_partitions
    ?(control_inputs = [])
    (data : 't t)
    (partitions : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type data)) ] in
  let attributes =
    ("num_partitions", Int num_partitions) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.dynamicPartition in
  let inputs = [ (`single (P data)); (`single (P partitions)) ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:(Node.output_type data)
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init num_partitions ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let dynamicStitch
    ?(name = "DynamicStitch")
    ?(control_inputs = [])
    (indices : [ `int32 ] t list)
    (data : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn data))) ] in
  let attributes =
    ("N", Int (List.length indices)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.dynamicStitch in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) indices)); (`multi (List.map ~f:(fun n -> P n) data)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn data))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let editDistance
    ?(name = "EditDistance")
    ?normalize
    ?(control_inputs = [])
    (hypothesis_indices : [ `int64 ] t)
    (hypothesis_values : 't t)
    (hypothesis_shape : [ `int64 ] t)
    (truth_indices : [ `int64 ] t)
    (truth_values : 't t)
    (truth_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type hypothesis_values)) ] in
  let attributes =
    match normalize with | None -> attributes | Some normalize -> ("normalize", Bool normalize) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.editDistance in
  let inputs = [ (`single (P hypothesis_indices)); (`single (P hypothesis_values)); (`single (P hypothesis_shape)); (`single (P truth_indices)); (`single (P truth_values)); (`single (P truth_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let elu
    ?(name = "Elu")
    ?(control_inputs = [])
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type features)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.elu in
  let inputs = [ (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let eluGrad
    ?(name = "EluGrad")
    ?(control_inputs = [])
    (gradients : ([< `float | `double ] as 't) t)
    (outputs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type gradients)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.eluGrad in
  let inputs = [ (`single (P gradients)); (`single (P outputs)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type gradients)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let empty
    ?(name = "Empty")
    ~type_
    ?init
    ?(control_inputs = [])
    (shape : [ `int32 ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match init with | None -> attributes | Some init -> ("init", Bool init) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.empty in
  let inputs = [ (`single (P shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let emptyTensorList
    ?(name = "EmptyTensorList")
    ?(control_inputs = [])
    (element_shape : ([< `int32 | `int64 ] as 'shape_type) t)
    (max_num_elements : [ `int32 ] t)
  =
  let attributes = [ "shape_type", Type (P (Node.output_type element_shape)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.emptyTensorList in
  let inputs = [ (`single (P element_shape)); (`single (P max_num_elements)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let encodeBase64
    ?(name = "EncodeBase64")
    ?pad
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match pad with | None -> attributes | Some pad -> ("pad", Bool pad) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.encodeBase64 in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let encodePng
    ?(name = "EncodePng")
    ?compression
    ?(control_inputs = [])
    (image : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type image)) ] in
  let attributes =
    match compression with | None -> attributes | Some compression -> ("compression", Int compression) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.encodePng in
  let inputs = [ (`single (P image)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let encodeWav
    ?(name = "EncodeWav")
    ?(control_inputs = [])
    (audio : [ `float ] t)
    (sample_rate : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.encodeWav in
  let inputs = [ (`single (P audio)); (`single (P sample_rate)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let enqueueTPUEmbeddingIntegerBatch
    ?(name = "EnqueueTPUEmbeddingIntegerBatch")
    ?device_ordinal
    ?(control_inputs = [])
    (batch : [ `int32 ] t list)
    (mode_override : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("N", Int (List.length batch)) :: attributes
  in
  let attributes =
    match device_ordinal with | None -> attributes | Some device_ordinal -> ("device_ordinal", Int device_ordinal) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.enqueueTPUEmbeddingIntegerBatch in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) batch)); (`single (P mode_override)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let enqueueTPUEmbeddingSparseBatch
    ?(name = "EnqueueTPUEmbeddingSparseBatch")
    ?device_ordinal
    ?(control_inputs = [])
    (sample_indices : ([< `int32 | `int64 ] as 't1) t list)
    (embedding_indices : ([< `int32 | `int64 ] as 't2) t list)
    (aggregation_weights : ([< `float | `double ] as 't3) t list)
    (mode_override : [ `string ] t)
  =
  let attributes = [ "T3", Type (P (Node.output_type (List.hd_exn aggregation_weights))) ;  "T2", Type (P (Node.output_type (List.hd_exn embedding_indices))) ;  "T1", Type (P (Node.output_type (List.hd_exn sample_indices))) ] in
  let attributes =
    ("N", Int (List.length sample_indices)) :: attributes
  in
  let attributes =
    match device_ordinal with | None -> attributes | Some device_ordinal -> ("device_ordinal", Int device_ordinal) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.enqueueTPUEmbeddingSparseBatch in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) sample_indices)); (`multi (List.map ~f:(fun n -> P n) embedding_indices)); (`multi (List.map ~f:(fun n -> P n) aggregation_weights)); (`single (P mode_override)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let enqueueTPUEmbeddingSparseTensorBatch
    ?(name = "EnqueueTPUEmbeddingSparseTensorBatch")
    ?device_ordinal
    ~table_ids
    ?max_sequence_lengths
    ?(control_inputs = [])
    (sample_indices : ([< `int32 | `int64 ] as 't1) t list)
    (embedding_indices : ([< `int32 | `int64 ] as 't2) t list)
    (aggregation_weights : ([< `float | `double ] as 't3) t list)
    (mode_override : [ `string ] t)
  =
  let attributes = [ "T3", Type (P (Node.output_type (List.hd_exn aggregation_weights))) ;  "T2", Type (P (Node.output_type (List.hd_exn embedding_indices))) ;  "T1", Type (P (Node.output_type (List.hd_exn sample_indices))) ] in
  let attributes =
    ("N", Int (List.length sample_indices)) :: attributes
  in
  let attributes =
    match device_ordinal with | None -> attributes | Some device_ordinal -> ("device_ordinal", Int device_ordinal) :: attributes
  in
  let attributes =
    ("table_ids", List (Int table_ids)) :: attributes
  in
  let attributes =
    match max_sequence_lengths with | None -> attributes | Some max_sequence_lengths -> ("max_sequence_lengths", List (Int max_sequence_lengths)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.enqueueTPUEmbeddingSparseTensorBatch in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) sample_indices)); (`multi (List.map ~f:(fun n -> P n) embedding_indices)); (`multi (List.map ~f:(fun n -> P n) aggregation_weights)); (`single (P mode_override)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let ensureShape
    ?(name = "EnsureShape")
    ~shape
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.ensureShape in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let enter
    ?(name = "Enter")
    ~frame_name
    ?is_constant
    ?parallel_iterations
    ?(control_inputs = [])
    (data : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type data)) ] in
  let attributes =
    ("frame_name", String frame_name) :: attributes
  in
  let attributes =
    match is_constant with | None -> attributes | Some is_constant -> ("is_constant", Bool is_constant) :: attributes
  in
  let attributes =
    match parallel_iterations with | None -> attributes | Some parallel_iterations -> ("parallel_iterations", Int parallel_iterations) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.enter in
  let inputs = [ (`single (P data)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let equal
    ?(name = "Equal")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.equal in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let erf
    ?(name = "Erf")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.erf in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let erfc
    ?(name = "Erfc")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.erfc in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let euclideanNorm
    ?(name = "EuclideanNorm")
    ?keep_dims
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (reduction_indices : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type reduction_indices)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.euclideanNorm in
  let inputs = [ (`single (P input)); (`single (P reduction_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let exit
    ?(name = "Exit")
    ?(control_inputs = [])
    (data : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.exit in
  let inputs = [ (`single (P data)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let exp
    ?(name = "Exp")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.exp in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let expandDims
    ?(name = "ExpandDims")
    ?(control_inputs = [])
    (input : 't t)
    (dim : ([< `int32 | `int64 ] as 'tdim) t)
  =
  let attributes = [ "Tdim", Type (P (Node.output_type dim)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.expandDims in
  let inputs = [ (`single (P input)); (`single (P dim)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalAssertNextDataset
    ?(name = "ExperimentalAssertNextDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (transformations : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalAssertNextDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P transformations)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalAutoShardDataset
    ?(name = "ExperimentalAutoShardDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (num_workers : [ `int64 ] t)
    (index : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalAutoShardDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P num_workers)); (`single (P index)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalBytesProducedStatsDataset
    ?(name = "ExperimentalBytesProducedStatsDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (tag : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalBytesProducedStatsDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P tag)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalChooseFastestDataset
    ?(name = "ExperimentalChooseFastestDataset")
    ~num_experiments
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_datasets : [ `variant ] t list)
  =
  let attributes = [] in
  let attributes =
    ("N", Int (List.length input_datasets)) :: attributes
  in
  let attributes =
    ("num_experiments", Int num_experiments) :: attributes
  in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalChooseFastestDataset in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) input_datasets)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalDatasetCardinality
    ?(name = "ExperimentalDatasetCardinality")
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalDatasetCardinality in
  let inputs = [ (`single (P input_dataset)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalDatasetToTFRecord
    ?(name = "ExperimentalDatasetToTFRecord")
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (filename : [ `string ] t)
    (compression_type : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalDatasetToTFRecord in
  let inputs = [ (`single (P input_dataset)); (`single (P filename)); (`single (P compression_type)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalDenseToSparseBatchDataset
    ?(name = "ExperimentalDenseToSparseBatchDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (batch_size : [ `int64 ] t)
    (row_shape : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalDenseToSparseBatchDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P batch_size)); (`single (P row_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalDirectedInterleaveDataset
    ?(name = "ExperimentalDirectedInterleaveDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (selector_input_dataset : [ `variant ] t)
    (data_input_datasets : [ `variant ] t list)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let attributes =
    ("N", Int (List.length data_input_datasets)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalDirectedInterleaveDataset in
  let inputs = [ (`single (P selector_input_dataset)); (`multi (List.map ~f:(fun n -> P n) data_input_datasets)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalIgnoreErrorsDataset
    ?(name = "ExperimentalIgnoreErrorsDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalIgnoreErrorsDataset in
  let inputs = [ (`single (P input_dataset)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalLMDBDataset
    ?(name = "ExperimentalLMDBDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (filenames : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalLMDBDataset in
  let inputs = [ (`single (P filenames)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalLatencyStatsDataset
    ?(name = "ExperimentalLatencyStatsDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (tag : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalLatencyStatsDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P tag)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalMatchingFilesDataset
    ?(name = "ExperimentalMatchingFilesDataset")
    ?(control_inputs = [])
    (patterns : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalMatchingFilesDataset in
  let inputs = [ (`single (P patterns)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalMaxIntraOpParallelismDataset
    ?(name = "ExperimentalMaxIntraOpParallelismDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (max_intra_op_parallelism : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalMaxIntraOpParallelismDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P max_intra_op_parallelism)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalNonSerializableDataset
    ?(name = "ExperimentalNonSerializableDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalNonSerializableDataset in
  let inputs = [ (`single (P input_dataset)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalPrivateThreadPoolDataset
    ?(name = "ExperimentalPrivateThreadPoolDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (num_threads : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalPrivateThreadPoolDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P num_threads)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalRandomDataset
    ?(name = "ExperimentalRandomDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (seed : [ `int64 ] t)
    (seed2 : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalRandomDataset in
  let inputs = [ (`single (P seed)); (`single (P seed2)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalRebatchDataset
    ?(name = "ExperimentalRebatchDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (num_workers : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalRebatchDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P num_workers)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalSleepDataset
    ?(name = "ExperimentalSleepDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (sleep_microseconds : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalSleepDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P sleep_microseconds)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalSlidingWindowDataset
    ?(name = "ExperimentalSlidingWindowDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (window_size : [ `int64 ] t)
    (window_shift : [ `int64 ] t)
    (window_stride : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalSlidingWindowDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P window_size)); (`single (P window_shift)); (`single (P window_stride)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalSqlDataset
    ?(name = "ExperimentalSqlDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (driver_name : [ `string ] t)
    (data_source_name : [ `string ] t)
    (query : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalSqlDataset in
  let inputs = [ (`single (P driver_name)); (`single (P data_source_name)); (`single (P query)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalUnbatchDataset
    ?(name = "ExperimentalUnbatchDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalUnbatchDataset in
  let inputs = [ (`single (P input_dataset)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let experimentalUniqueDataset
    ?(name = "ExperimentalUniqueDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.experimentalUniqueDataset in
  let inputs = [ (`single (P input_dataset)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let expm1
    ?(name = "Expm1")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.expm1 in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let extractGlimpse
    ?(name = "ExtractGlimpse")
    ?centered
    ?normalized
    ?uniform_noise
    ?noise
    ?(control_inputs = [])
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
  let attributes =
    match noise with | None -> attributes | Some noise -> ("noise", String noise) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.extractGlimpse in
  let inputs = [ (`single (P input)); (`single (P size)); (`single (P offsets)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let extractImagePatches
    ?(name = "ExtractImagePatches")
    ~ksizes
    ~strides
    ~rates
    ~padding
    ?(control_inputs = [])
    (images : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let attributes =
    ("ksizes", List (Int ksizes)) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("rates", List (Int rates)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.extractImagePatches in
  let inputs = [ (`single (P images)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let extractJpegShape
    ?(name = "ExtractJpegShape")
    ~type_
    ?(control_inputs = [])
    (contents : [ `string ] t)
  =
  let attributes = [ "output_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.extractJpegShape in
  let inputs = [ (`single (P contents)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let extractVolumePatches
    ?(name = "ExtractVolumePatches")
    ~ksizes
    ~strides
    ~padding
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("ksizes", List (Int ksizes)) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.extractVolumePatches in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fFT
    ?(name = "FFT")
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 'tcomplex) t)
  =
  let attributes = [ "Tcomplex", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.fFT in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fFT2D
    ?(name = "FFT2D")
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 'tcomplex) t)
  =
  let attributes = [ "Tcomplex", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.fFT2D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fFT3D
    ?(name = "FFT3D")
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 'tcomplex) t)
  =
  let attributes = [ "Tcomplex", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.fFT3D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fIFOQueue
    ?(name = "FIFOQueue")
    ~component_types
    ?shapes
    ?capacity
    ?container
    ?shared_name
    ?(control_inputs = [])
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
  let name = Name.of_string name in
  let op_name = Op_names.fIFOQueue in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fact
    ?(name = "Fact")
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.fact in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fakeParam
    ?(name = "FakeParam")
    ~type_
    ~shape
    ?(control_inputs = [])
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fakeParam in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fakeQuantWithMinMaxArgs
    ?(name = "FakeQuantWithMinMaxArgs")
    ?min
    ?max
    ?num_bits
    ?narrow_range
    ?(control_inputs = [])
    (inputs__ : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match min with | None -> attributes | Some min -> ("min", Float min) :: attributes
  in
  let attributes =
    match max with | None -> attributes | Some max -> ("max", Float max) :: attributes
  in
  let attributes =
    match num_bits with | None -> attributes | Some num_bits -> ("num_bits", Int num_bits) :: attributes
  in
  let attributes =
    match narrow_range with | None -> attributes | Some narrow_range -> ("narrow_range", Bool narrow_range) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fakeQuantWithMinMaxArgs in
  let inputs = [ (`single (P inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fakeQuantWithMinMaxArgsGradient
    ?(name = "FakeQuantWithMinMaxArgsGradient")
    ?min
    ?max
    ?num_bits
    ?narrow_range
    ?(control_inputs = [])
    (gradients : [ `float ] t)
    (inputs__ : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match min with | None -> attributes | Some min -> ("min", Float min) :: attributes
  in
  let attributes =
    match max with | None -> attributes | Some max -> ("max", Float max) :: attributes
  in
  let attributes =
    match num_bits with | None -> attributes | Some num_bits -> ("num_bits", Int num_bits) :: attributes
  in
  let attributes =
    match narrow_range with | None -> attributes | Some narrow_range -> ("narrow_range", Bool narrow_range) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fakeQuantWithMinMaxArgsGradient in
  let inputs = [ (`single (P gradients)); (`single (P inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fakeQuantWithMinMaxVars
    ?(name = "FakeQuantWithMinMaxVars")
    ?num_bits
    ?narrow_range
    ?(control_inputs = [])
    (inputs__ : [ `float ] t)
    (min : [ `float ] t)
    (max : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match num_bits with | None -> attributes | Some num_bits -> ("num_bits", Int num_bits) :: attributes
  in
  let attributes =
    match narrow_range with | None -> attributes | Some narrow_range -> ("narrow_range", Bool narrow_range) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fakeQuantWithMinMaxVars in
  let inputs = [ (`single (P inputs__)); (`single (P min)); (`single (P max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fakeQuantWithMinMaxVarsGradient
    ?(name = "FakeQuantWithMinMaxVarsGradient")
    ?num_bits
    ?narrow_range
    ?(control_inputs = [])
    (gradients : [ `float ] t)
    (inputs__ : [ `float ] t)
    (min : [ `float ] t)
    (max : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match num_bits with | None -> attributes | Some num_bits -> ("num_bits", Int num_bits) :: attributes
  in
  let attributes =
    match narrow_range with | None -> attributes | Some narrow_range -> ("narrow_range", Bool narrow_range) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fakeQuantWithMinMaxVarsGradient in
  let inputs = [ (`single (P gradients)); (`single (P inputs__)); (`single (P min)); (`single (P max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let fakeQuantWithMinMaxVarsPerChannel
    ?(name = "FakeQuantWithMinMaxVarsPerChannel")
    ?num_bits
    ?narrow_range
    ?(control_inputs = [])
    (inputs__ : [ `float ] t)
    (min : [ `float ] t)
    (max : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match num_bits with | None -> attributes | Some num_bits -> ("num_bits", Int num_bits) :: attributes
  in
  let attributes =
    match narrow_range with | None -> attributes | Some narrow_range -> ("narrow_range", Bool narrow_range) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fakeQuantWithMinMaxVarsPerChannel in
  let inputs = [ (`single (P inputs__)); (`single (P min)); (`single (P max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fakeQuantWithMinMaxVarsPerChannelGradient
    ?(name = "FakeQuantWithMinMaxVarsPerChannelGradient")
    ?num_bits
    ?narrow_range
    ?(control_inputs = [])
    (gradients : [ `float ] t)
    (inputs__ : [ `float ] t)
    (min : [ `float ] t)
    (max : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match num_bits with | None -> attributes | Some num_bits -> ("num_bits", Int num_bits) :: attributes
  in
  let attributes =
    match narrow_range with | None -> attributes | Some narrow_range -> ("narrow_range", Bool narrow_range) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fakeQuantWithMinMaxVarsPerChannelGradient in
  let inputs = [ (`single (P gradients)); (`single (P inputs__)); (`single (P min)); (`single (P max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let fill
    ?(name = "Fill")
    ?(control_inputs = [])
    (dims : ([< `int32 | `int64 ] as 'index_type) t)
    (value : 't t)
  =
  let attributes = [ "index_type", Type (P (Node.output_type dims)) ;  "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.fill in
  let inputs = [ (`single (P dims)); (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type value)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let filterByLastComponentDataset
    ?(name = "FilterByLastComponentDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.filterByLastComponentDataset in
  let inputs = [ (`single (P input_dataset)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fixedLengthRecordDataset
    ?(name = "FixedLengthRecordDataset")
    ?(control_inputs = [])
    (filenames : [ `string ] t)
    (header_bytes : [ `int64 ] t)
    (record_bytes : [ `int64 ] t)
    (footer_bytes : [ `int64 ] t)
    (buffer_size : [ `int64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.fixedLengthRecordDataset in
  let inputs = [ (`single (P filenames)); (`single (P header_bytes)); (`single (P record_bytes)); (`single (P footer_bytes)); (`single (P buffer_size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fixedLengthRecordDatasetV2
    ?(name = "FixedLengthRecordDatasetV2")
    ?(control_inputs = [])
    (filenames : [ `string ] t)
    (header_bytes : [ `int64 ] t)
    (record_bytes : [ `int64 ] t)
    (footer_bytes : [ `int64 ] t)
    (buffer_size : [ `int64 ] t)
    (compression_type : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.fixedLengthRecordDatasetV2 in
  let inputs = [ (`single (P filenames)); (`single (P header_bytes)); (`single (P record_bytes)); (`single (P footer_bytes)); (`single (P buffer_size)); (`single (P compression_type)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fixedLengthRecordReader
    ?(name = "FixedLengthRecordReader")
    ?header_bytes
    ~record_bytes
    ?footer_bytes
    ?hop_bytes
    ?container
    ?shared_name
    ?(control_inputs = [])
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
    match hop_bytes with | None -> attributes | Some hop_bytes -> ("hop_bytes", Int hop_bytes) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fixedLengthRecordReader in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fixedUnigramCandidateSampler
    ?(name = "FixedUnigramCandidateSampler")
    ~num_true
    ~num_sampled
    ~unique
    ~range_max
    ?vocab_file
    ?distortion
    ?num_reserved_ids
    ?num_shards
    ?shard
    ?unigrams
    ?seed
    ?seed2
    ?(control_inputs = [])
    (true_classes : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_true", Int num_true) :: attributes
  in
  let attributes =
    ("num_sampled", Int num_sampled) :: attributes
  in
  let attributes =
    ("unique", Bool unique) :: attributes
  in
  let attributes =
    ("range_max", Int range_max) :: attributes
  in
  let attributes =
    match vocab_file with | None -> attributes | Some vocab_file -> ("vocab_file", String vocab_file) :: attributes
  in
  let attributes =
    match distortion with | None -> attributes | Some distortion -> ("distortion", Float distortion) :: attributes
  in
  let attributes =
    match num_reserved_ids with | None -> attributes | Some num_reserved_ids -> ("num_reserved_ids", Int num_reserved_ids) :: attributes
  in
  let attributes =
    match num_shards with | None -> attributes | Some num_shards -> ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    match shard with | None -> attributes | Some shard -> ("shard", Int shard) :: attributes
  in
  let attributes =
    match unigrams with | None -> attributes | Some unigrams -> ("unigrams", List (Float unigrams)) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fixedUnigramCandidateSampler in
  let inputs = [ (`single (P true_classes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let floor
    ?(name = "Floor")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.floor in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let floorDiv
    ?(name = "FloorDiv")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.floorDiv in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let floorMod
    ?(name = "FloorMod")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (y : ([< `int32 | `int64 | `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.floorMod in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fractionalAvgPool
    ?(name = "FractionalAvgPool")
    ~pooling_ratio
    ?pseudo_random
    ?overlapping
    ?deterministic
    ?seed
    ?seed2
    ?(control_inputs = [])
    (value : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let attributes =
    ("pooling_ratio", List (Float pooling_ratio)) :: attributes
  in
  let attributes =
    match pseudo_random with | None -> attributes | Some pseudo_random -> ("pseudo_random", Bool pseudo_random) :: attributes
  in
  let attributes =
    match overlapping with | None -> attributes | Some overlapping -> ("overlapping", Bool overlapping) :: attributes
  in
  let attributes =
    match deterministic with | None -> attributes | Some deterministic -> ("deterministic", Bool deterministic) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fractionalAvgPool in
  let inputs = [ (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type value)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let fractionalAvgPoolGrad
    ?(name = "FractionalAvgPoolGrad")
    ?overlapping
    ?(control_inputs = [])
    (orig_input_tensor_shape : [ `int64 ] t)
    (out_backprop : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (row_pooling_sequence : [ `int64 ] t)
    (col_pooling_sequence : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type out_backprop)) ] in
  let attributes =
    match overlapping with | None -> attributes | Some overlapping -> ("overlapping", Bool overlapping) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fractionalAvgPoolGrad in
  let inputs = [ (`single (P orig_input_tensor_shape)); (`single (P out_backprop)); (`single (P row_pooling_sequence)); (`single (P col_pooling_sequence)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type out_backprop)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fractionalMaxPool
    ?(name = "FractionalMaxPool")
    ~pooling_ratio
    ?pseudo_random
    ?overlapping
    ?deterministic
    ?seed
    ?seed2
    ?(control_inputs = [])
    (value : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let attributes =
    ("pooling_ratio", List (Float pooling_ratio)) :: attributes
  in
  let attributes =
    match pseudo_random with | None -> attributes | Some pseudo_random -> ("pseudo_random", Bool pseudo_random) :: attributes
  in
  let attributes =
    match overlapping with | None -> attributes | Some overlapping -> ("overlapping", Bool overlapping) :: attributes
  in
  let attributes =
    match deterministic with | None -> attributes | Some deterministic -> ("deterministic", Bool deterministic) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fractionalMaxPool in
  let inputs = [ (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type value)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let fractionalMaxPoolGrad
    ?(name = "FractionalMaxPoolGrad")
    ?overlapping
    ?(control_inputs = [])
    (orig_input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (orig_output : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (out_backprop : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (row_pooling_sequence : [ `int64 ] t)
    (col_pooling_sequence : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type orig_input)) ] in
  let attributes =
    match overlapping with | None -> attributes | Some overlapping -> ("overlapping", Bool overlapping) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fractionalMaxPoolGrad in
  let inputs = [ (`single (P orig_input)); (`single (P orig_output)); (`single (P out_backprop)); (`single (P row_pooling_sequence)); (`single (P col_pooling_sequence)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type orig_input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fusedBatchNorm
    ?(name = "FusedBatchNorm")
    ?epsilon
    ?data_format
    ?is_training
    ?(control_inputs = [])
    (x : ([< `float ] as 't) t)
    (scale : ([< `float ] as 't) t)
    (offset : ([< `float ] as 't) t)
    (mean : ([< `float ] as 't) t)
    (variance : ([< `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ;  "T", Type (P (Node.output_type x)) ;  "T", Type (P (Node.output_type x)) ;  "T", Type (P (Node.output_type x)) ;  "T", Type (P (Node.output_type x)) ] in
  let attributes =
    match epsilon with | None -> attributes | Some epsilon -> ("epsilon", Float epsilon) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match is_training with | None -> attributes | Some is_training -> ("is_training", Bool is_training) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fusedBatchNorm in
  let inputs = [ (`single (P x)); (`single (P scale)); (`single (P offset)); (`single (P mean)); (`single (P variance)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 4)

let fusedBatchNormGrad
    ?(name = "FusedBatchNormGrad")
    ?epsilon
    ?data_format
    ?is_training
    ?(control_inputs = [])
    (y_backprop : ([< `float ] as 't) t)
    (x : ([< `float ] as 't) t)
    (scale : ([< `float ] as 't) t)
    (reserve_space_1 : ([< `float ] as 't) t)
    (reserve_space_2 : ([< `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type y_backprop)) ;  "T", Type (P (Node.output_type y_backprop)) ;  "T", Type (P (Node.output_type y_backprop)) ;  "T", Type (P (Node.output_type y_backprop)) ;  "T", Type (P (Node.output_type y_backprop)) ] in
  let attributes =
    match epsilon with | None -> attributes | Some epsilon -> ("epsilon", Float epsilon) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match is_training with | None -> attributes | Some is_training -> ("is_training", Bool is_training) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fusedBatchNormGrad in
  let inputs = [ (`single (P y_backprop)); (`single (P x)); (`single (P scale)); (`single (P reserve_space_1)); (`single (P reserve_space_2)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y_backprop)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y_backprop)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y_backprop)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y_backprop)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y_backprop)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 4)

let fusedBatchNormGradV2
    ?(name = "FusedBatchNormGradV2")
    ?epsilon
    ?data_format
    ?is_training
    ?(control_inputs = [])
    (y_backprop : ([< `float ] as 't) t)
    (x : ([< `float ] as 't) t)
    (scale : [ `float ] t)
    (reserve_space_1 : ([< `float ] as 'u) t)
    (reserve_space_2 : ([< `float ] as 'u) t)
  =
  let attributes = [ "T", Type (P (Node.output_type y_backprop)) ;  "U", Type (P (Node.output_type reserve_space_1)) ;  "U", Type (P (Node.output_type reserve_space_1)) ;  "U", Type (P (Node.output_type reserve_space_1)) ;  "U", Type (P (Node.output_type reserve_space_1)) ] in
  let attributes =
    match epsilon with | None -> attributes | Some epsilon -> ("epsilon", Float epsilon) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match is_training with | None -> attributes | Some is_training -> ("is_training", Bool is_training) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fusedBatchNormGradV2 in
  let inputs = [ (`single (P y_backprop)); (`single (P x)); (`single (P scale)); (`single (P reserve_space_1)); (`single (P reserve_space_2)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y_backprop)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type reserve_space_1)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type reserve_space_1)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type reserve_space_1)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type reserve_space_1)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 4)

let fusedBatchNormV2
    ?(name = "FusedBatchNormV2")
    ?epsilon
    ?data_format
    ?is_training
    ?(control_inputs = [])
    (x : ([< `float ] as 't) t)
    (scale : ([< `float ] as 'u) t)
    (offset : ([< `float ] as 'u) t)
    (mean : ([< `float ] as 'u) t)
    (variance : ([< `float ] as 'u) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ;  "U", Type (P (Node.output_type scale)) ;  "U", Type (P (Node.output_type scale)) ;  "U", Type (P (Node.output_type scale)) ;  "U", Type (P (Node.output_type scale)) ] in
  let attributes =
    match epsilon with | None -> attributes | Some epsilon -> ("epsilon", Float epsilon) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let attributes =
    match is_training with | None -> attributes | Some is_training -> ("is_training", Bool is_training) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fusedBatchNormV2 in
  let inputs = [ (`single (P x)); (`single (P scale)); (`single (P offset)); (`single (P mean)); (`single (P variance)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type scale)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type scale)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type scale)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type scale)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 4)

let fusedPadConv2D
    ?(name = "FusedPadConv2D")
    ~mode
    ~strides
    ~padding
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (paddings : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("mode", String mode) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fusedPadConv2D in
  let inputs = [ (`single (P input)); (`single (P paddings)); (`single (P filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let fusedResizeAndPadConv2D
    ?(name = "FusedResizeAndPadConv2D")
    ?resize_align_corners
    ~mode
    ~strides
    ~padding
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (size : [ `int32 ] t)
    (paddings : [ `int32 ] t)
    (filter : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match resize_align_corners with | None -> attributes | Some resize_align_corners -> ("resize_align_corners", Bool resize_align_corners) :: attributes
  in
  let attributes =
    ("mode", String mode) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.fusedResizeAndPadConv2D in
  let inputs = [ (`single (P input)); (`single (P size)); (`single (P paddings)); (`single (P filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let gather
    ?(name = "Gather")
    ?validate_indices
    ?(control_inputs = [])
    (params : 'tparams t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "Tparams", Type (P (Node.output_type params)) ] in
  let attributes =
    match validate_indices with | None -> attributes | Some validate_indices -> ("validate_indices", Bool validate_indices) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.gather in
  let inputs = [ (`single (P params)); (`single (P indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type params)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let gatherNd
    ?(name = "GatherNd")
    ?(control_inputs = [])
    (params : 'tparams t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "Tparams", Type (P (Node.output_type params)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.gatherNd in
  let inputs = [ (`single (P params)); (`single (P indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type params)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let gatherV2
    ?(name = "GatherV2")
    ?(control_inputs = [])
    (params : 'tparams t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (axis : ([< `int32 | `int64 ] as 'taxis) t)
  =
  let attributes = [ "Taxis", Type (P (Node.output_type axis)) ;  "Tindices", Type (P (Node.output_type indices)) ;  "Tparams", Type (P (Node.output_type params)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.gatherV2 in
  let inputs = [ (`single (P params)); (`single (P indices)); (`single (P axis)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type params)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let generateVocabRemapping
    ?(name = "GenerateVocabRemapping")
    ~new_vocab_offset
    ~num_new_vocab
    ?old_vocab_size
    ?(control_inputs = [])
    (new_vocab_file : [ `string ] t)
    (old_vocab_file : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("new_vocab_offset", Int new_vocab_offset) :: attributes
  in
  let attributes =
    ("num_new_vocab", Int num_new_vocab) :: attributes
  in
  let attributes =
    match old_vocab_size with | None -> attributes | Some old_vocab_size -> ("old_vocab_size", Int old_vocab_size) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.generateVocabRemapping in
  let inputs = [ (`single (P new_vocab_file)); (`single (P old_vocab_file)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let getSessionHandle
    ?(name = "GetSessionHandle")
    ?(control_inputs = [])
    (value : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.getSessionHandle in
  let inputs = [ (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let getSessionTensor
    ?(name = "GetSessionTensor")
    ~type_
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.getSessionTensor in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let greater
    ?(name = "Greater")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.greater in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let greaterEqual
    ?(name = "GreaterEqual")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.greaterEqual in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let guaranteeConst
    ?(name = "GuaranteeConst")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.guaranteeConst in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let hSVToRGB
    ?(name = "HSVToRGB")
    ?(control_inputs = [])
    (images : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.hSVToRGB in
  let inputs = [ (`single (P images)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let hashTable
    ?(name = "HashTable")
    ?container
    ?shared_name
    ?use_node_name_sharing
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let attributes =
    match use_node_name_sharing with | None -> attributes | Some use_node_name_sharing -> ("use_node_name_sharing", Bool use_node_name_sharing) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.hashTable in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let histogramFixedWidth
    ?(name = "HistogramFixedWidth")
    ~type_
    ?(control_inputs = [])
    (values : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (value_range : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (nbins : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type values)) ;  "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.histogramFixedWidth in
  let inputs = [ (`single (P values)); (`single (P value_range)); (`single (P nbins)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let histogramSummary
    ?(name = "HistogramSummary")
    ?(control_inputs = [])
    (tag : [ `string ] t)
    (values : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.histogramSummary in
  let inputs = [ (`single (P tag)); (`single (P values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let hostConst
    ?(name = "HostConst")
    ~type_
    ?(control_inputs = [])
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.hostConst in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let iFFT
    ?(name = "IFFT")
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 'tcomplex) t)
  =
  let attributes = [ "Tcomplex", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.iFFT in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let iFFT2D
    ?(name = "IFFT2D")
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 'tcomplex) t)
  =
  let attributes = [ "Tcomplex", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.iFFT2D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let iFFT3D
    ?(name = "IFFT3D")
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 'tcomplex) t)
  =
  let attributes = [ "Tcomplex", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.iFFT3D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let iRFFT
    ?(name = "IRFFT")
    ?(control_inputs = [])
    (input : [ `complex64 ] t)
    (fft_length : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.iRFFT in
  let inputs = [ (`single (P input)); (`single (P fft_length)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let iRFFT2D
    ?(name = "IRFFT2D")
    ?(control_inputs = [])
    (input : [ `complex64 ] t)
    (fft_length : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.iRFFT2D in
  let inputs = [ (`single (P input)); (`single (P fft_length)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let iRFFT3D
    ?(name = "IRFFT3D")
    ?(control_inputs = [])
    (input : [ `complex64 ] t)
    (fft_length : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.iRFFT3D in
  let inputs = [ (`single (P input)); (`single (P fft_length)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let identity
    ?(name = "Identity")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.identity in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let identityReader
    ?(name = "IdentityReader")
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.identityReader in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let igamma
    ?(name = "Igamma")
    ?(control_inputs = [])
    (a : ([< `float | `double ] as 't) t)
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type a)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.igamma in
  let inputs = [ (`single (P a)); (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let igammaGradA
    ?(name = "IgammaGradA")
    ?(control_inputs = [])
    (a : ([< `float | `double ] as 't) t)
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type a)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.igammaGradA in
  let inputs = [ (`single (P a)); (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let igammac
    ?(name = "Igammac")
    ?(control_inputs = [])
    (a : ([< `float | `double ] as 't) t)
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type a)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.igammac in
  let inputs = [ (`single (P a)); (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let imag
    ?(name = "Imag")
    ~type_
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "Tout", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.imag in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let imageSummary
    ?(name = "ImageSummary")
    ?max_images
    ?(control_inputs = [])
    (tag : [ `string ] t)
    (tensor : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type tensor)) ] in
  let attributes =
    match max_images with | None -> attributes | Some max_images -> ("max_images", Int max_images) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.imageSummary in
  let inputs = [ (`single (P tag)); (`single (P tensor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let immutableConst
    ?(name = "ImmutableConst")
    ~type_
    ~shape
    ~memory_region_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let attributes =
    ("memory_region_name", String memory_region_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.immutableConst in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let inTopK
    ?(name = "InTopK")
    ~k
    ?(control_inputs = [])
    (predictions : [ `float ] t)
    (targets : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type targets)) ] in
  let attributes =
    ("k", Int k) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.inTopK in
  let inputs = [ (`single (P predictions)); (`single (P targets)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let inTopKV2
    ?(name = "InTopKV2")
    ?(control_inputs = [])
    (predictions : [ `float ] t)
    (targets : ([< `int32 | `int64 ] as 't) t)
    (k : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type targets)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.inTopKV2 in
  let inputs = [ (`single (P predictions)); (`single (P targets)); (`single (P k)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let infeedDequeue
    ?(name = "InfeedDequeue")
    ~type_
    ~shape
    ?(control_inputs = [])
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.infeedDequeue in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let infeedEnqueue
    ?(name = "InfeedEnqueue")
    ?shape
    ?layout
    ?device_ordinal
    ?(control_inputs = [])
    (input : 'dtype t)
  =
  let attributes = [ "dtype", Type (P (Node.output_type input)) ] in
  let attributes =
    match shape with | None -> attributes | Some shape -> ("shape", Shape shape) :: attributes
  in
  let attributes =
    match layout with | None -> attributes | Some layout -> ("layout", List (Int layout)) :: attributes
  in
  let attributes =
    match device_ordinal with | None -> attributes | Some device_ordinal -> ("device_ordinal", Int device_ordinal) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.infeedEnqueue in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let infeedEnqueuePrelinearizedBuffer
    ?(name = "InfeedEnqueuePrelinearizedBuffer")
    ?device_ordinal
    ?(control_inputs = [])
    (input : [ `variant ] t)
  =
  let attributes = [] in
  let attributes =
    match device_ordinal with | None -> attributes | Some device_ordinal -> ("device_ordinal", Int device_ordinal) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.infeedEnqueuePrelinearizedBuffer in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let initializeTable
    ?(name = "InitializeTable")
    ?(control_inputs = [])
    (table_handle : [ `string ] t)
    (keys : 'tkey t)
    (values : 'tval t)
  =
  let attributes = [ "Tval", Type (P (Node.output_type values)) ;  "Tkey", Type (P (Node.output_type keys)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.initializeTable in
  let inputs = [ (`single (P table_handle)); (`single (P keys)); (`single (P values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let initializeTableFromTextFile
    ?(name = "InitializeTableFromTextFile")
    ~key_index
    ~value_index
    ?vocab_size
    ?delimiter
    ?(control_inputs = [])
    (table_handle : [ `string ] t)
    (filename : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("key_index", Int key_index) :: attributes
  in
  let attributes =
    ("value_index", Int value_index) :: attributes
  in
  let attributes =
    match vocab_size with | None -> attributes | Some vocab_size -> ("vocab_size", Int vocab_size) :: attributes
  in
  let attributes =
    match delimiter with | None -> attributes | Some delimiter -> ("delimiter", String delimiter) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.initializeTableFromTextFile in
  let inputs = [ (`single (P table_handle)); (`single (P filename)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let inplaceAdd
    ?(name = "InplaceAdd")
    ?(control_inputs = [])
    (x : 't t)
    (i : [ `int32 ] t)
    (v : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.inplaceAdd in
  let inputs = [ (`single (P x)); (`single (P i)); (`single (P v)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let inplaceSub
    ?(name = "InplaceSub")
    ?(control_inputs = [])
    (x : 't t)
    (i : [ `int32 ] t)
    (v : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.inplaceSub in
  let inputs = [ (`single (P x)); (`single (P i)); (`single (P v)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let inplaceUpdate
    ?(name = "InplaceUpdate")
    ?(control_inputs = [])
    (x : 't t)
    (i : [ `int32 ] t)
    (v : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.inplaceUpdate in
  let inputs = [ (`single (P x)); (`single (P i)); (`single (P v)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let inv
    ?(name = "Inv")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.inv in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let invGrad
    ?(name = "InvGrad")
    ?(control_inputs = [])
    (y : ([< `float | `double | `complex64 ] as 't) t)
    (dy : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type y)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.invGrad in
  let inputs = [ (`single (P y)); (`single (P dy)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let invert
    ?(name = "Invert")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.invert in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let invertPermutation
    ?(name = "InvertPermutation")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.invertPermutation in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let isFinite
    ?(name = "IsFinite")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.isFinite in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let isInf
    ?(name = "IsInf")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.isInf in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let isNan
    ?(name = "IsNan")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.isNan in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let isVariableInitialized
    ?(name = "IsVariableInitialized")
    ?(control_inputs = [])
    (ref : 'dtype t)
  =
  let attributes = [ "dtype", Type (P (Node.output_type ref)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.isVariableInitialized in
  let inputs = [ (`single (P ref)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let kMC2ChainInitialization
    ?(name = "KMC2ChainInitialization")
    ?(control_inputs = [])
    (distances : [ `float ] t)
    (seed : [ `int64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.kMC2ChainInitialization in
  let inputs = [ (`single (P distances)); (`single (P seed)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let kmeansPlusPlusInitialization
    ?(name = "KmeansPlusPlusInitialization")
    ?(control_inputs = [])
    (points : [ `float ] t)
    (num_to_sample : [ `int64 ] t)
    (seed : [ `int64 ] t)
    (num_retries_per_sample : [ `int64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.kmeansPlusPlusInitialization in
  let inputs = [ (`single (P points)); (`single (P num_to_sample)); (`single (P seed)); (`single (P num_retries_per_sample)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let l2Loss
    ?(name = "L2Loss")
    ?(control_inputs = [])
    (t : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type t)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.l2Loss in
  let inputs = [ (`single (P t)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type t)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lMDBReader
    ?(name = "LMDBReader")
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.lMDBReader in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lRN
    ?(name = "LRN")
    ?depth_radius
    ?bias
    ?alpha
    ?beta
    ?(control_inputs = [])
    (input : ([< `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.lRN in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lRNGrad
    ?(name = "LRNGrad")
    ?depth_radius
    ?bias
    ?alpha
    ?beta
    ?(control_inputs = [])
    (input_grads : ([< `float ] as 't) t)
    (input_image : ([< `float ] as 't) t)
    (output_image : ([< `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input_grads)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.lRNGrad in
  let inputs = [ (`single (P input_grads)); (`single (P input_image)); (`single (P output_image)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input_grads)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let leakyRelu
    ?(name = "LeakyRelu")
    ?alpha
    ?(control_inputs = [])
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type features)) ] in
  let attributes =
    match alpha with | None -> attributes | Some alpha -> ("alpha", Float alpha) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.leakyRelu in
  let inputs = [ (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let leakyReluGrad
    ?(name = "LeakyReluGrad")
    ?alpha
    ?(control_inputs = [])
    (gradients : ([< `float | `double ] as 't) t)
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type gradients)) ] in
  let attributes =
    match alpha with | None -> attributes | Some alpha -> ("alpha", Float alpha) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.leakyReluGrad in
  let inputs = [ (`single (P gradients)); (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type gradients)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let learnedUnigramCandidateSampler
    ?(name = "LearnedUnigramCandidateSampler")
    ~num_true
    ~num_sampled
    ~unique
    ~range_max
    ?seed
    ?seed2
    ?(control_inputs = [])
    (true_classes : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_true", Int num_true) :: attributes
  in
  let attributes =
    ("num_sampled", Int num_sampled) :: attributes
  in
  let attributes =
    ("unique", Bool unique) :: attributes
  in
  let attributes =
    ("range_max", Int range_max) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.learnedUnigramCandidateSampler in
  let inputs = [ (`single (P true_classes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let leftShift
    ?(name = "LeftShift")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 ] as 't) t)
    (y : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.leftShift in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let less
    ?(name = "Less")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.less in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lessEqual
    ?(name = "LessEqual")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.lessEqual in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lgamma
    ?(name = "Lgamma")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.lgamma in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let linSpace
    ?(name = "LinSpace")
    ?(control_inputs = [])
    (start : ([< `float | `double ] as 't) t)
    (stop : ([< `float | `double ] as 't) t)
    (num : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type num)) ;  "T", Type (P (Node.output_type start)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.linSpace in
  let inputs = [ (`single (P start)); (`single (P stop)); (`single (P num)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type start)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let listDiff
    ?(name = "ListDiff")
    ~type_1
    ?(control_inputs = [])
    (x : 't t)
    (y : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ;  "out_idx", Type (P type_1) ] in
  let name = Name.of_string name in
  let op_name = Op_names.listDiff in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let loadAndRemapMatrix
    ?(name = "LoadAndRemapMatrix")
    ~num_rows
    ~num_cols
    ?max_rows_in_memory
    ?(control_inputs = [])
    (ckpt_path : [ `string ] t)
    (old_tensor_name : [ `string ] t)
    (row_remapping : [ `int64 ] t)
    (col_remapping : [ `int64 ] t)
    (initializing_values : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_rows", Int num_rows) :: attributes
  in
  let attributes =
    ("num_cols", Int num_cols) :: attributes
  in
  let attributes =
    match max_rows_in_memory with | None -> attributes | Some max_rows_in_memory -> ("max_rows_in_memory", Int max_rows_in_memory) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadAndRemapMatrix in
  let inputs = [ (`single (P ckpt_path)); (`single (P old_tensor_name)); (`single (P row_remapping)); (`single (P col_remapping)); (`single (P initializing_values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingADAMParameters
    ?(name = "LoadTPUEmbeddingADAMParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (momenta : [ `float ] t)
    (velocities : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingADAMParameters in
  let inputs = [ (`single (P parameters)); (`single (P momenta)); (`single (P velocities)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingADAMParametersGradAccumDebug
    ?(name = "LoadTPUEmbeddingADAMParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (momenta : [ `float ] t)
    (velocities : [ `float ] t)
    (gradient_accumulators : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingADAMParametersGradAccumDebug in
  let inputs = [ (`single (P parameters)); (`single (P momenta)); (`single (P velocities)); (`single (P gradient_accumulators)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingAdadeltaParameters
    ?(name = "LoadTPUEmbeddingAdadeltaParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (accumulators : [ `float ] t)
    (updates : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingAdadeltaParameters in
  let inputs = [ (`single (P parameters)); (`single (P accumulators)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingAdadeltaParametersGradAccumDebug
    ?(name = "LoadTPUEmbeddingAdadeltaParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (accumulators : [ `float ] t)
    (updates : [ `float ] t)
    (gradient_accumulators : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingAdadeltaParametersGradAccumDebug in
  let inputs = [ (`single (P parameters)); (`single (P accumulators)); (`single (P updates)); (`single (P gradient_accumulators)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingAdagradParameters
    ?(name = "LoadTPUEmbeddingAdagradParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (accumulators : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingAdagradParameters in
  let inputs = [ (`single (P parameters)); (`single (P accumulators)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingAdagradParametersGradAccumDebug
    ?(name = "LoadTPUEmbeddingAdagradParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (accumulators : [ `float ] t)
    (gradient_accumulators : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingAdagradParametersGradAccumDebug in
  let inputs = [ (`single (P parameters)); (`single (P accumulators)); (`single (P gradient_accumulators)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingCenteredRMSPropParameters
    ?(name = "LoadTPUEmbeddingCenteredRMSPropParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (ms : [ `float ] t)
    (mom : [ `float ] t)
    (mg : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingCenteredRMSPropParameters in
  let inputs = [ (`single (P parameters)); (`single (P ms)); (`single (P mom)); (`single (P mg)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingFTRLParameters
    ?(name = "LoadTPUEmbeddingFTRLParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (accumulators : [ `float ] t)
    (linears : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingFTRLParameters in
  let inputs = [ (`single (P parameters)); (`single (P accumulators)); (`single (P linears)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingFTRLParametersGradAccumDebug
    ?(name = "LoadTPUEmbeddingFTRLParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (accumulators : [ `float ] t)
    (linears : [ `float ] t)
    (gradient_accumulators : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingFTRLParametersGradAccumDebug in
  let inputs = [ (`single (P parameters)); (`single (P accumulators)); (`single (P linears)); (`single (P gradient_accumulators)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingMDLAdagradLightParameters
    ?(name = "LoadTPUEmbeddingMDLAdagradLightParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (accumulators : [ `float ] t)
    (weights : [ `float ] t)
    (benefits : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingMDLAdagradLightParameters in
  let inputs = [ (`single (P parameters)); (`single (P accumulators)); (`single (P weights)); (`single (P benefits)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingMomentumParameters
    ?(name = "LoadTPUEmbeddingMomentumParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (momenta : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingMomentumParameters in
  let inputs = [ (`single (P parameters)); (`single (P momenta)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingMomentumParametersGradAccumDebug
    ?(name = "LoadTPUEmbeddingMomentumParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (momenta : [ `float ] t)
    (gradient_accumulators : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingMomentumParametersGradAccumDebug in
  let inputs = [ (`single (P parameters)); (`single (P momenta)); (`single (P gradient_accumulators)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingProximalAdagradParameters
    ?(name = "LoadTPUEmbeddingProximalAdagradParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (accumulators : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingProximalAdagradParameters in
  let inputs = [ (`single (P parameters)); (`single (P accumulators)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingProximalAdagradParametersGradAccumDebug
    ?(name = "LoadTPUEmbeddingProximalAdagradParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (accumulators : [ `float ] t)
    (gradient_accumulators : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingProximalAdagradParametersGradAccumDebug in
  let inputs = [ (`single (P parameters)); (`single (P accumulators)); (`single (P gradient_accumulators)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingRMSPropParameters
    ?(name = "LoadTPUEmbeddingRMSPropParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (ms : [ `float ] t)
    (mom : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingRMSPropParameters in
  let inputs = [ (`single (P parameters)); (`single (P ms)); (`single (P mom)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingRMSPropParametersGradAccumDebug
    ?(name = "LoadTPUEmbeddingRMSPropParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
    (ms : [ `float ] t)
    (mom : [ `float ] t)
    (gradient_accumulators : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingRMSPropParametersGradAccumDebug in
  let inputs = [ (`single (P parameters)); (`single (P ms)); (`single (P mom)); (`single (P gradient_accumulators)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loadTPUEmbeddingStochasticGradientDescentParameters
    ?(name = "LoadTPUEmbeddingStochasticGradientDescentParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    (parameters : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.loadTPUEmbeddingStochasticGradientDescentParameters in
  let inputs = [ (`single (P parameters)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let log
    ?(name = "Log")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.log in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let log1p
    ?(name = "Log1p")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.log1p in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let logMatrixDeterminant
    ?(name = "LogMatrixDeterminant")
    ?(control_inputs = [])
    (input : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.logMatrixDeterminant in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let logSoftmax
    ?(name = "LogSoftmax")
    ?(control_inputs = [])
    (logits : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type logits)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.logSoftmax in
  let inputs = [ (`single (P logits)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type logits)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let logUniformCandidateSampler
    ?(name = "LogUniformCandidateSampler")
    ~num_true
    ~num_sampled
    ~unique
    ~range_max
    ?seed
    ?seed2
    ?(control_inputs = [])
    (true_classes : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_true", Int num_true) :: attributes
  in
  let attributes =
    ("num_sampled", Int num_sampled) :: attributes
  in
  let attributes =
    ("unique", Bool unique) :: attributes
  in
  let attributes =
    ("range_max", Int range_max) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.logUniformCandidateSampler in
  let inputs = [ (`single (P true_classes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let logicalAnd
    ?(name = "LogicalAnd")
    ?(control_inputs = [])
    (x : [ `bool ] t)
    (y : [ `bool ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.logicalAnd in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let logicalNot
    ?(name = "LogicalNot")
    ?(control_inputs = [])
    (x : [ `bool ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.logicalNot in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let logicalOr
    ?(name = "LogicalOr")
    ?(control_inputs = [])
    (x : [ `bool ] t)
    (y : [ `bool ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.logicalOr in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lookupTableExport
    ?(name = "LookupTableExport")
    ~type_
    ~type_1
    ?(control_inputs = [])
    (table_handle : [ `string ] t)
  =
  let attributes = [ "Tkeys", Type (P type_) ;  "Tvalues", Type (P type_1) ] in
  let name = Name.of_string name in
  let op_name = Op_names.lookupTableExport in
  let inputs = [ (`single (P table_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let lookupTableFind
    ?(name = "LookupTableFind")
    ?(control_inputs = [])
    (table_handle : [ `string ] t)
    (keys : 'tin t)
    (default_value : 'tout t)
  =
  let attributes = [ "Tin", Type (P (Node.output_type keys)) ;  "Tout", Type (P (Node.output_type default_value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.lookupTableFind in
  let inputs = [ (`single (P table_handle)); (`single (P keys)); (`single (P default_value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type default_value)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lookupTableImport
    ?(name = "LookupTableImport")
    ?(control_inputs = [])
    (table_handle : [ `string ] t)
    (keys : 'tin t)
    (values : 'tout t)
  =
  let attributes = [ "Tout", Type (P (Node.output_type values)) ;  "Tin", Type (P (Node.output_type keys)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.lookupTableImport in
  let inputs = [ (`single (P table_handle)); (`single (P keys)); (`single (P values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lookupTableInsert
    ?(name = "LookupTableInsert")
    ?(control_inputs = [])
    (table_handle : [ `string ] t)
    (keys : 'tin t)
    (values : 'tout t)
  =
  let attributes = [ "Tout", Type (P (Node.output_type values)) ;  "Tin", Type (P (Node.output_type keys)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.lookupTableInsert in
  let inputs = [ (`single (P table_handle)); (`single (P keys)); (`single (P values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lookupTableSize
    ?(name = "LookupTableSize")
    ?(control_inputs = [])
    (table_handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.lookupTableSize in
  let inputs = [ (`single (P table_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let loopCond
    ?(name = "LoopCond")
    ?(control_inputs = [])
    (input : [ `bool ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.loopCond in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lowerBound
    ?(name = "LowerBound")
    ~type_
    ?(control_inputs = [])
    (sorted_inputs : 't t)
    (values : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type sorted_inputs)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.lowerBound in
  let inputs = [ (`single (P sorted_inputs)); (`single (P values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let lu
    ?(name = "Lu")
    ~type_1
    ?(control_inputs = [])
    (input : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "output_idx_type", Type (P type_1) ] in
  let name = Name.of_string name in
  let op_name = Op_names.lu in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let mapClear
    ?(name = "MapClear")
    ?capacity
    ?memory_limit
    ~dtypes
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match memory_limit with | None -> attributes | Some memory_limit -> ("memory_limit", Int memory_limit) :: attributes
  in
  let attributes =
    ("dtypes", List (Type dtypes)) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mapClear in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mapIncompleteSize
    ?(name = "MapIncompleteSize")
    ?capacity
    ?memory_limit
    ~dtypes
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match memory_limit with | None -> attributes | Some memory_limit -> ("memory_limit", Int memory_limit) :: attributes
  in
  let attributes =
    ("dtypes", List (Type dtypes)) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mapIncompleteSize in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mapSize
    ?(name = "MapSize")
    ?capacity
    ?memory_limit
    ~dtypes
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match memory_limit with | None -> attributes | Some memory_limit -> ("memory_limit", Int memory_limit) :: attributes
  in
  let attributes =
    ("dtypes", List (Type dtypes)) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mapSize in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matMul
    ?(name = "MatMul")
    ?transpose_a
    ?transpose_b
    ?(control_inputs = [])
    (a : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (b : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type a)) ] in
  let attributes =
    match transpose_a with | None -> attributes | Some transpose_a -> ("transpose_a", Bool transpose_a) :: attributes
  in
  let attributes =
    match transpose_b with | None -> attributes | Some transpose_b -> ("transpose_b", Bool transpose_b) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.matMul in
  let inputs = [ (`single (P a)); (`single (P b)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matchingFiles
    ?(name = "MatchingFiles")
    ?(control_inputs = [])
    (pattern : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.matchingFiles in
  let inputs = [ (`single (P pattern)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixBandPart
    ?(name = "MatrixBandPart")
    ?(control_inputs = [])
    (input : 't t)
    (num_lower : ([< `int32 | `int64 ] as 'tindex) t)
    (num_upper : ([< `int32 | `int64 ] as 'tindex) t)
  =
  let attributes = [ "Tindex", Type (P (Node.output_type num_lower)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.matrixBandPart in
  let inputs = [ (`single (P input)); (`single (P num_lower)); (`single (P num_upper)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixDeterminant
    ?(name = "MatrixDeterminant")
    ?(control_inputs = [])
    (input : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.matrixDeterminant in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixDiag
    ?(name = "MatrixDiag")
    ?(control_inputs = [])
    (diagonal : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type diagonal)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.matrixDiag in
  let inputs = [ (`single (P diagonal)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type diagonal)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixDiagPart
    ?(name = "MatrixDiagPart")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.matrixDiagPart in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixExponential
    ?(name = "MatrixExponential")
    ?(control_inputs = [])
    (input : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.matrixExponential in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixInverse
    ?(name = "MatrixInverse")
    ?adjoint
    ?(control_inputs = [])
    (input : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match adjoint with | None -> attributes | Some adjoint -> ("adjoint", Bool adjoint) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.matrixInverse in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixLogarithm
    ?(name = "MatrixLogarithm")
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.matrixLogarithm in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixSetDiag
    ?(name = "MatrixSetDiag")
    ?(control_inputs = [])
    (input : 't t)
    (diagonal : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.matrixSetDiag in
  let inputs = [ (`single (P input)); (`single (P diagonal)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixSolve
    ?(name = "MatrixSolve")
    ?adjoint
    ?(control_inputs = [])
    (matrix : ([< `double | `float | `complex64 ] as 't) t)
    (rhs : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type matrix)) ] in
  let attributes =
    match adjoint with | None -> attributes | Some adjoint -> ("adjoint", Bool adjoint) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.matrixSolve in
  let inputs = [ (`single (P matrix)); (`single (P rhs)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type matrix)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixSolveLs
    ?(name = "MatrixSolveLs")
    ?fast
    ?(control_inputs = [])
    (matrix : ([< `double | `float | `complex64 ] as 't) t)
    (rhs : ([< `double | `float | `complex64 ] as 't) t)
    (l2_regularizer : [ `double ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type matrix)) ] in
  let attributes =
    match fast with | None -> attributes | Some fast -> ("fast", Bool fast) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.matrixSolveLs in
  let inputs = [ (`single (P matrix)); (`single (P rhs)); (`single (P l2_regularizer)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type matrix)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixSquareRoot
    ?(name = "MatrixSquareRoot")
    ?(control_inputs = [])
    (input : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.matrixSquareRoot in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let matrixTriangularSolve
    ?(name = "MatrixTriangularSolve")
    ?lower
    ?adjoint
    ?(control_inputs = [])
    (matrix : ([< `double | `float | `complex64 ] as 't) t)
    (rhs : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type matrix)) ] in
  let attributes =
    match lower with | None -> attributes | Some lower -> ("lower", Bool lower) :: attributes
  in
  let attributes =
    match adjoint with | None -> attributes | Some adjoint -> ("adjoint", Bool adjoint) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.matrixTriangularSolve in
  let inputs = [ (`single (P matrix)); (`single (P rhs)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type matrix)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let max
    ?(name = "Max")
    ?keep_dims
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (reduction_indices : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type reduction_indices)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.max in
  let inputs = [ (`single (P input)); (`single (P reduction_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPool
    ?(name = "MaxPool")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.maxPool in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPool3D
    ?(name = "MaxPool3D")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (input : ([< `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.maxPool3D in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPool3DGrad
    ?(name = "MaxPool3DGrad")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (orig_input : ([< `float ] as 'tInput) t)
    (orig_output : ([< `float ] as 'tInput) t)
    (grad : ([< `float ] as 't) t)
  =
  let attributes = [ "TInput", Type (P (Node.output_type orig_input)) ;  "T", Type (P (Node.output_type grad)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.maxPool3DGrad in
  let inputs = [ (`single (P orig_input)); (`single (P orig_output)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type grad)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPool3DGradGrad
    ?(name = "MaxPool3DGradGrad")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (orig_input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (orig_output : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type orig_input)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.maxPool3DGradGrad in
  let inputs = [ (`single (P orig_input)); (`single (P orig_output)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type orig_input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPoolGrad
    ?(name = "MaxPoolGrad")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (orig_input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (orig_output : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type orig_input)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.maxPoolGrad in
  let inputs = [ (`single (P orig_input)); (`single (P orig_output)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type orig_input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPoolGradGrad
    ?(name = "MaxPoolGradGrad")
    ~ksize
    ~strides
    ~padding
    ?data_format
    ?(control_inputs = [])
    (orig_input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (orig_output : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type orig_input)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.maxPoolGradGrad in
  let inputs = [ (`single (P orig_input)); (`single (P orig_output)); (`single (P grad)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type orig_input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPoolGradGradV2
    ?(name = "MaxPoolGradGradV2")
    ~padding
    ?data_format
    ?(control_inputs = [])
    (orig_input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (orig_output : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (ksize : [ `int32 ] t)
    (strides : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type orig_input)) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.maxPoolGradGradV2 in
  let inputs = [ (`single (P orig_input)); (`single (P orig_output)); (`single (P grad)); (`single (P ksize)); (`single (P strides)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type orig_input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPoolGradGradWithArgmax
    ?(name = "MaxPoolGradGradWithArgmax")
    ~ksize
    ~strides
    ~padding
    ?include_batch_in_index
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (argmax : ([< `int32 | `int64 ] as 'targmax) t)
  =
  let attributes = [ "Targmax", Type (P (Node.output_type argmax)) ;  "T", Type (P (Node.output_type input)) ] in
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
    match include_batch_in_index with | None -> attributes | Some include_batch_in_index -> ("include_batch_in_index", Bool include_batch_in_index) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.maxPoolGradGradWithArgmax in
  let inputs = [ (`single (P input)); (`single (P grad)); (`single (P argmax)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPoolGradV2
    ?(name = "MaxPoolGradV2")
    ~padding
    ?data_format
    ?(control_inputs = [])
    (orig_input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (orig_output : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (ksize : [ `int32 ] t)
    (strides : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type orig_input)) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.maxPoolGradV2 in
  let inputs = [ (`single (P orig_input)); (`single (P orig_output)); (`single (P grad)); (`single (P ksize)); (`single (P strides)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type orig_input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPoolGradWithArgmax
    ?(name = "MaxPoolGradWithArgmax")
    ~ksize
    ~strides
    ~padding
    ?include_batch_in_index
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (argmax : ([< `int32 | `int64 ] as 'targmax) t)
  =
  let attributes = [ "Targmax", Type (P (Node.output_type argmax)) ;  "T", Type (P (Node.output_type input)) ] in
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
    match include_batch_in_index with | None -> attributes | Some include_batch_in_index -> ("include_batch_in_index", Bool include_batch_in_index) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.maxPoolGradWithArgmax in
  let inputs = [ (`single (P input)); (`single (P grad)); (`single (P argmax)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPoolV2
    ?(name = "MaxPoolV2")
    ~padding
    ?data_format
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (ksize : [ `int32 ] t)
    (strides : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.maxPoolV2 in
  let inputs = [ (`single (P input)); (`single (P ksize)); (`single (P strides)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let maxPoolWithArgmax
    ?(name = "MaxPoolWithArgmax")
    ~type_1
    ~ksize
    ~strides
    ~padding
    ?include_batch_in_index
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "Targmax", Type (P type_1) ] in
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
    match include_batch_in_index with | None -> attributes | Some include_batch_in_index -> ("include_batch_in_index", Bool include_batch_in_index) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.maxPoolWithArgmax in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let maximum
    ?(name = "Maximum")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.maximum in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mean
    ?(name = "Mean")
    ?keep_dims
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (reduction_indices : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type reduction_indices)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mean in
  let inputs = [ (`single (P input)); (`single (P reduction_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let merge
    ?(name = "Merge")
    ?(control_inputs = [])
    (inputs__ : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn inputs__))) ] in
  let attributes =
    ("N", Int (List.length inputs__)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.merge in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn inputs__))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let mergeSummary
    ?(name = "MergeSummary")
    ?(control_inputs = [])
    (inputs__ : [ `string ] t list)
  =
  let attributes = [] in
  let attributes =
    ("N", Int (List.length inputs__)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mergeSummary in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mergeV2Checkpoints
    ?(name = "MergeV2Checkpoints")
    ?delete_old_dirs
    ?(control_inputs = [])
    (checkpoint_prefixes : [ `string ] t)
    (destination_prefix : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match delete_old_dirs with | None -> attributes | Some delete_old_dirs -> ("delete_old_dirs", Bool delete_old_dirs) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mergeV2Checkpoints in
  let inputs = [ (`single (P checkpoint_prefixes)); (`single (P destination_prefix)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mfcc
    ?(name = "Mfcc")
    ?upper_frequency_limit
    ?lower_frequency_limit
    ?filterbank_channel_count
    ?dct_coefficient_count
    ?(control_inputs = [])
    (spectrogram : [ `float ] t)
    (sample_rate : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match upper_frequency_limit with | None -> attributes | Some upper_frequency_limit -> ("upper_frequency_limit", Float upper_frequency_limit) :: attributes
  in
  let attributes =
    match lower_frequency_limit with | None -> attributes | Some lower_frequency_limit -> ("lower_frequency_limit", Float lower_frequency_limit) :: attributes
  in
  let attributes =
    match filterbank_channel_count with | None -> attributes | Some filterbank_channel_count -> ("filterbank_channel_count", Int filterbank_channel_count) :: attributes
  in
  let attributes =
    match dct_coefficient_count with | None -> attributes | Some dct_coefficient_count -> ("dct_coefficient_count", Int dct_coefficient_count) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mfcc in
  let inputs = [ (`single (P spectrogram)); (`single (P sample_rate)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let min
    ?(name = "Min")
    ?keep_dims
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (reduction_indices : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type reduction_indices)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.min in
  let inputs = [ (`single (P input)); (`single (P reduction_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let minimum
    ?(name = "Minimum")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.minimum in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mirrorPad
    ?(name = "MirrorPad")
    ~mode
    ?(control_inputs = [])
    (input : 't t)
    (paddings : ([< `int32 | `int64 ] as 'tpaddings) t)
  =
  let attributes = [ "Tpaddings", Type (P (Node.output_type paddings)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("mode", String mode) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mirrorPad in
  let inputs = [ (`single (P input)); (`single (P paddings)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mirrorPadGrad
    ?(name = "MirrorPadGrad")
    ~mode
    ?(control_inputs = [])
    (input : 't t)
    (paddings : ([< `int32 | `int64 ] as 'tpaddings) t)
  =
  let attributes = [ "Tpaddings", Type (P (Node.output_type paddings)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("mode", String mode) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mirrorPadGrad in
  let inputs = [ (`single (P input)); (`single (P paddings)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mod_
    ?(name = "Mod")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (y : ([< `int32 | `int64 | `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.mod_ in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let modelDataset
    ?(name = "ModelDataset")
    ?cpu_budget
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
  =
  let attributes = [] in
  let attributes =
    match cpu_budget with | None -> attributes | Some cpu_budget -> ("cpu_budget", Int cpu_budget) :: attributes
  in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.modelDataset in
  let inputs = [ (`single (P input_dataset)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mul
    ?(name = "Mul")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.mul in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mulNoNan
    ?(name = "MulNoNan")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
    (y : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.mulNoNan in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let multinomial
    ?(name = "Multinomial")
    ~type_
    ?seed
    ?seed2
    ?(control_inputs = [])
    (logits : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (num_samples : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type logits)) ;  "output_dtype", Type (P type_) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.multinomial in
  let inputs = [ (`single (P logits)); (`single (P num_samples)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mutableDenseHashTable
    ?(name = "MutableDenseHashTable")
    ?container
    ?shared_name
    ?use_node_name_sharing
    ?value_shape
    ?initial_num_buckets
    ?max_load_factor
    ?(control_inputs = [])
    (empty_key : 'key_dtype t)
  =
  let attributes = [ "key_dtype", Type (P (Node.output_type empty_key)) ] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let attributes =
    match use_node_name_sharing with | None -> attributes | Some use_node_name_sharing -> ("use_node_name_sharing", Bool use_node_name_sharing) :: attributes
  in
  let attributes =
    match value_shape with | None -> attributes | Some value_shape -> ("value_shape", Shape value_shape) :: attributes
  in
  let attributes =
    match initial_num_buckets with | None -> attributes | Some initial_num_buckets -> ("initial_num_buckets", Int initial_num_buckets) :: attributes
  in
  let attributes =
    match max_load_factor with | None -> attributes | Some max_load_factor -> ("max_load_factor", Float max_load_factor) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mutableDenseHashTable in
  let inputs = [ (`single (P empty_key)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mutableHashTable
    ?(name = "MutableHashTable")
    ?container
    ?shared_name
    ?use_node_name_sharing
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let attributes =
    match use_node_name_sharing with | None -> attributes | Some use_node_name_sharing -> ("use_node_name_sharing", Bool use_node_name_sharing) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mutableHashTable in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let mutableHashTableOfTensors
    ?(name = "MutableHashTableOfTensors")
    ?container
    ?shared_name
    ?use_node_name_sharing
    ?value_shape
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let attributes =
    match use_node_name_sharing with | None -> attributes | Some use_node_name_sharing -> ("use_node_name_sharing", Bool use_node_name_sharing) :: attributes
  in
  let attributes =
    match value_shape with | None -> attributes | Some value_shape -> ("value_shape", Shape value_shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.mutableHashTableOfTensors in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let ncclAllReduce
    ?(name = "NcclAllReduce")
    ~reduction
    ~num_devices
    ~shared_name
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("reduction", String reduction) :: attributes
  in
  let attributes =
    ("num_devices", Int num_devices) :: attributes
  in
  let attributes =
    ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.ncclAllReduce in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let ncclBroadcast
    ?(name = "NcclBroadcast")
    ~shape
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.ncclBroadcast in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let ncclReduce
    ?(name = "NcclReduce")
    ~reduction
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn input))) ] in
  let attributes =
    ("reduction", String reduction) :: attributes
  in
  let attributes =
    ("num_devices", Int (List.length input)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.ncclReduce in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn input))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let nearestNeighbors
    ?(name = "NearestNeighbors")
    ?(control_inputs = [])
    (points : [ `float ] t)
    (centers : [ `float ] t)
    (k : [ `int64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.nearestNeighbors in
  let inputs = [ (`single (P points)); (`single (P centers)); (`single (P k)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let neg
    ?(name = "Neg")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.neg in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let negTrain
    ?(name = "NegTrain")
    ~vocab_count
    ~num_negative_samples
    ?(control_inputs = [])
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
  let name = Name.of_string name in
  let op_name = Op_names.negTrain in
  let inputs = [ (`single (P w_in)); (`single (P w_out)); (`single (P examples)); (`single (P labels)); (`single (P lr)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let nextAfter
    ?(name = "NextAfter")
    ?(control_inputs = [])
    (x1 : ([< `double | `float ] as 't) t)
    (x2 : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x1)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.nextAfter in
  let inputs = [ (`single (P x1)); (`single (P x2)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x1)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let nextIteration
    ?(name = "NextIteration")
    ?(control_inputs = [])
    (data : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.nextIteration in
  let inputs = [ (`single (P data)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let noOp
    ?(name = "NoOp")
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.noOp in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let nonDeterministicInts
    ?(name = "NonDeterministicInts")
    ~type_
    ?(control_inputs = [])
    (shape : 'shape_dtype t)
  =
  let attributes = [ "shape_dtype", Type (P (Node.output_type shape)) ;  "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.nonDeterministicInts in
  let inputs = [ (`single (P shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let nonMaxSuppression
    ?(name = "NonMaxSuppression")
    ?iou_threshold
    ?(control_inputs = [])
    (boxes : [ `float ] t)
    (scores : [ `float ] t)
    (max_output_size : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match iou_threshold with | None -> attributes | Some iou_threshold -> ("iou_threshold", Float iou_threshold) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.nonMaxSuppression in
  let inputs = [ (`single (P boxes)); (`single (P scores)); (`single (P max_output_size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let nonMaxSuppressionV2
    ?(name = "NonMaxSuppressionV2")
    ?(control_inputs = [])
    (boxes : ([< `float ] as 't) t)
    (scores : ([< `float ] as 't) t)
    (max_output_size : [ `int32 ] t)
    (iou_threshold : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type boxes)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.nonMaxSuppressionV2 in
  let inputs = [ (`single (P boxes)); (`single (P scores)); (`single (P max_output_size)); (`single (P iou_threshold)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let nonMaxSuppressionV3
    ?(name = "NonMaxSuppressionV3")
    ?(control_inputs = [])
    (boxes : ([< `float ] as 't) t)
    (scores : ([< `float ] as 't) t)
    (max_output_size : [ `int32 ] t)
    (iou_threshold : [ `float ] t)
    (score_threshold : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type boxes)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.nonMaxSuppressionV3 in
  let inputs = [ (`single (P boxes)); (`single (P scores)); (`single (P max_output_size)); (`single (P iou_threshold)); (`single (P score_threshold)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let nonMaxSuppressionV4
    ?(name = "NonMaxSuppressionV4")
    ?pad_to_max_output_size
    ?(control_inputs = [])
    (boxes : ([< `float ] as 't) t)
    (scores : ([< `float ] as 't) t)
    (max_output_size : [ `int32 ] t)
    (iou_threshold : [ `float ] t)
    (score_threshold : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type boxes)) ] in
  let attributes =
    match pad_to_max_output_size with | None -> attributes | Some pad_to_max_output_size -> ("pad_to_max_output_size", Bool pad_to_max_output_size) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.nonMaxSuppressionV4 in
  let inputs = [ (`single (P boxes)); (`single (P scores)); (`single (P max_output_size)); (`single (P iou_threshold)); (`single (P score_threshold)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let nonMaxSuppressionWithOverlaps
    ?(name = "NonMaxSuppressionWithOverlaps")
    ?(control_inputs = [])
    (overlaps : [ `float ] t)
    (scores : [ `float ] t)
    (max_output_size : [ `int32 ] t)
    (overlap_threshold : [ `float ] t)
    (score_threshold : [ `float ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.nonMaxSuppressionWithOverlaps in
  let inputs = [ (`single (P overlaps)); (`single (P scores)); (`single (P max_output_size)); (`single (P overlap_threshold)); (`single (P score_threshold)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let notEqual
    ?(name = "NotEqual")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.notEqual in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let nthElement
    ?(name = "NthElement")
    ?reverse
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (n : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match reverse with | None -> attributes | Some reverse -> ("reverse", Bool reverse) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.nthElement in
  let inputs = [ (`single (P input)); (`single (P n)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let oneHot
    ?(name = "OneHot")
    ?axis
    ?(control_inputs = [])
    (indices : ([< `int32 | `int64 ] as 'tI) t)
    (depth : [ `int32 ] t)
    (on_value : 't t)
    (off_value : 't t)
  =
  let attributes = [ "TI", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type on_value)) ] in
  let attributes =
    match axis with | None -> attributes | Some axis -> ("axis", Int axis) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.oneHot in
  let inputs = [ (`single (P indices)); (`single (P depth)); (`single (P on_value)); (`single (P off_value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type on_value)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let onesLike
    ?(name = "OnesLike")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 | `bool ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.onesLike in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let optimizeDataset
    ?(name = "OptimizeDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (optimizations : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.optimizeDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P optimizations)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let optionalHasValue
    ?(name = "OptionalHasValue")
    ?(control_inputs = [])
    (optional : [ `variant ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.optionalHasValue in
  let inputs = [ (`single (P optional)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let optionalNone
    ?(name = "OptionalNone")
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.optionalNone in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let orderedMapClear
    ?(name = "OrderedMapClear")
    ?capacity
    ?memory_limit
    ~dtypes
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match memory_limit with | None -> attributes | Some memory_limit -> ("memory_limit", Int memory_limit) :: attributes
  in
  let attributes =
    ("dtypes", List (Type dtypes)) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.orderedMapClear in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let orderedMapIncompleteSize
    ?(name = "OrderedMapIncompleteSize")
    ?capacity
    ?memory_limit
    ~dtypes
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match memory_limit with | None -> attributes | Some memory_limit -> ("memory_limit", Int memory_limit) :: attributes
  in
  let attributes =
    ("dtypes", List (Type dtypes)) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.orderedMapIncompleteSize in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let orderedMapSize
    ?(name = "OrderedMapSize")
    ?capacity
    ?memory_limit
    ~dtypes
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match memory_limit with | None -> attributes | Some memory_limit -> ("memory_limit", Int memory_limit) :: attributes
  in
  let attributes =
    ("dtypes", List (Type dtypes)) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.orderedMapSize in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let outfeedDequeue
    ?(name = "OutfeedDequeue")
    ~type_
    ~shape
    ?device_ordinal
    ?(control_inputs = [])
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let attributes =
    match device_ordinal with | None -> attributes | Some device_ordinal -> ("device_ordinal", Int device_ordinal) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.outfeedDequeue in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let outfeedEnqueue
    ?(name = "OutfeedEnqueue")
    ?(control_inputs = [])
    (input : 'dtype t)
  =
  let attributes = [ "dtype", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.outfeedEnqueue in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let pack
    ?(name = "Pack")
    ?axis
    ?(control_inputs = [])
    (values : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn values))) ] in
  let attributes =
    ("N", Int (List.length values)) :: attributes
  in
  let attributes =
    match axis with | None -> attributes | Some axis -> ("axis", Int axis) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.pack in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn values))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let pad
    ?(name = "Pad")
    ?(control_inputs = [])
    (input : 't t)
    (paddings : ([< `int32 | `int64 ] as 'tpaddings) t)
  =
  let attributes = [ "Tpaddings", Type (P (Node.output_type paddings)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.pad in
  let inputs = [ (`single (P input)); (`single (P paddings)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let padV2
    ?(name = "PadV2")
    ?(control_inputs = [])
    (input : 't t)
    (paddings : ([< `int32 | `int64 ] as 'tpaddings) t)
    (constant_values : 't t)
  =
  let attributes = [ "Tpaddings", Type (P (Node.output_type paddings)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.padV2 in
  let inputs = [ (`single (P input)); (`single (P paddings)); (`single (P constant_values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let paddingFIFOQueue
    ?(name = "PaddingFIFOQueue")
    ~component_types
    ?shapes
    ?capacity
    ?container
    ?shared_name
    ?(control_inputs = [])
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
  let name = Name.of_string name in
  let op_name = Op_names.paddingFIFOQueue in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let parallelConcat
    ?(name = "ParallelConcat")
    ~shape
    ?(control_inputs = [])
    (values : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn values))) ] in
  let attributes =
    ("N", Int (List.length values)) :: attributes
  in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.parallelConcat in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn values))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let parallelDynamicStitch
    ?(name = "ParallelDynamicStitch")
    ?(control_inputs = [])
    (indices : [ `int32 ] t list)
    (data : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn data))) ] in
  let attributes =
    ("N", Int (List.length indices)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.parallelDynamicStitch in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) indices)); (`multi (List.map ~f:(fun n -> P n) data)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn data))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let parameterizedTruncatedNormal
    ?(name = "ParameterizedTruncatedNormal")
    ?seed
    ?seed2
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 't) t)
    (means : ([< `float | `double ] as 'dtype) t)
    (stdevs : ([< `float | `double ] as 'dtype) t)
    (minvals : ([< `float | `double ] as 'dtype) t)
    (maxvals : ([< `float | `double ] as 'dtype) t)
  =
  let attributes = [ "T", Type (P (Node.output_type shape)) ;  "dtype", Type (P (Node.output_type means)) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.parameterizedTruncatedNormal in
  let inputs = [ (`single (P shape)); (`single (P means)); (`single (P stdevs)); (`single (P minvals)); (`single (P maxvals)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type means)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let parseTensor
    ?(name = "ParseTensor")
    ~type_
    ?(control_inputs = [])
    (serialized : [ `string ] t)
  =
  let attributes = [ "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.parseTensor in
  let inputs = [ (`single (P serialized)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let placeholder
    ?(name = "Placeholder")
    ~type_
    ?shape
    ?(control_inputs = [])
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match shape with | None -> attributes | Some shape -> ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.placeholder in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let placeholderV2
    ?(name = "PlaceholderV2")
    ~type_
    ~shape
    ?(control_inputs = [])
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.placeholderV2 in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let placeholderWithDefault
    ?(name = "PlaceholderWithDefault")
    ~shape
    ?(control_inputs = [])
    (input : 'dtype t)
  =
  let attributes = [ "dtype", Type (P (Node.output_type input)) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.placeholderWithDefault in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let polygamma
    ?(name = "Polygamma")
    ?(control_inputs = [])
    (a : ([< `float | `double ] as 't) t)
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type a)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.polygamma in
  let inputs = [ (`single (P a)); (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let pow
    ?(name = "Pow")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.pow in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let prefetchDataset
    ?(name = "PrefetchDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (buffer_size : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.prefetchDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P buffer_size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let prelinearize
    ?(name = "Prelinearize")
    ?shape
    ?layout
    ?(control_inputs = [])
    (input : 'dtype t)
  =
  let attributes = [ "dtype", Type (P (Node.output_type input)) ] in
  let attributes =
    match shape with | None -> attributes | Some shape -> ("shape", Shape shape) :: attributes
  in
  let attributes =
    match layout with | None -> attributes | Some layout -> ("layout", List (Int layout)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.prelinearize in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let preventGradient
    ?(name = "PreventGradient")
    ?message
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match message with | None -> attributes | Some message -> ("message", String message) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.preventGradient in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let printV2
    ?(name = "PrintV2")
    ?output_stream
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match output_stream with | None -> attributes | Some output_stream -> ("output_stream", String output_stream) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.printV2 in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let priorityQueue
    ?(name = "PriorityQueue")
    ?component_types
    ~shapes
    ?capacity
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match component_types with | None -> attributes | Some component_types -> ("component_types", List (Type component_types)) :: attributes
  in
  let attributes =
    ("shapes", List (Shape shapes)) :: attributes
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
  let name = Name.of_string name in
  let op_name = Op_names.priorityQueue in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let prod
    ?(name = "Prod")
    ?keep_dims
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (reduction_indices : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type reduction_indices)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.prod in
  let inputs = [ (`single (P input)); (`single (P reduction_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let qr
    ?(name = "Qr")
    ?full_matrices
    ?(control_inputs = [])
    (input : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match full_matrices with | None -> attributes | Some full_matrices -> ("full_matrices", Bool full_matrices) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.qr in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let quantizeAndDequantize
    ?(name = "QuantizeAndDequantize")
    ?signed_input
    ?num_bits
    ?range_given
    ?input_min
    ?input_max
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match signed_input with | None -> attributes | Some signed_input -> ("signed_input", Bool signed_input) :: attributes
  in
  let attributes =
    match num_bits with | None -> attributes | Some num_bits -> ("num_bits", Int num_bits) :: attributes
  in
  let attributes =
    match range_given with | None -> attributes | Some range_given -> ("range_given", Bool range_given) :: attributes
  in
  let attributes =
    match input_min with | None -> attributes | Some input_min -> ("input_min", Float input_min) :: attributes
  in
  let attributes =
    match input_max with | None -> attributes | Some input_max -> ("input_max", Float input_max) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizeAndDequantize in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let quantizeAndDequantizeV2
    ?(name = "QuantizeAndDequantizeV2")
    ?signed_input
    ?num_bits
    ?range_given
    ?round_mode
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (input_min : ([< `float | `double ] as 't) t)
    (input_max : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match signed_input with | None -> attributes | Some signed_input -> ("signed_input", Bool signed_input) :: attributes
  in
  let attributes =
    match num_bits with | None -> attributes | Some num_bits -> ("num_bits", Int num_bits) :: attributes
  in
  let attributes =
    match range_given with | None -> attributes | Some range_given -> ("range_given", Bool range_given) :: attributes
  in
  let attributes =
    match round_mode with | None -> attributes | Some round_mode -> ("round_mode", String round_mode) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizeAndDequantizeV2 in
  let inputs = [ (`single (P input)); (`single (P input_min)); (`single (P input_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let quantizeAndDequantizeV3
    ?(name = "QuantizeAndDequantizeV3")
    ?signed_input
    ?range_given
    ?(control_inputs = [])
    (input : ([< `float | `double ] as 't) t)
    (input_min : ([< `float | `double ] as 't) t)
    (input_max : ([< `float | `double ] as 't) t)
    (num_bits : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match signed_input with | None -> attributes | Some signed_input -> ("signed_input", Bool signed_input) :: attributes
  in
  let attributes =
    match range_given with | None -> attributes | Some range_given -> ("range_given", Bool range_given) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizeAndDequantizeV3 in
  let inputs = [ (`single (P input)); (`single (P input_min)); (`single (P input_max)); (`single (P num_bits)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let quantizeDownAndShrinkRange
    ?(name = "QuantizeDownAndShrinkRange")
    ~type_
    ?(control_inputs = [])
    (input : 'tinput t)
    (input_min : [ `float ] t)
    (input_max : [ `float ] t)
  =
  let attributes = [ "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.quantizeDownAndShrinkRange in
  let inputs = [ (`single (P input)); (`single (P input_min)); (`single (P input_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizeV2
    ?(name = "QuantizeV2")
    ~type_
    ?mode
    ?round_mode
    ?(control_inputs = [])
    (input : [ `float ] t)
    (min_range : [ `float ] t)
    (max_range : [ `float ] t)
  =
  let attributes = [ "T", Type (P type_) ] in
  let attributes =
    match mode with | None -> attributes | Some mode -> ("mode", String mode) :: attributes
  in
  let attributes =
    match round_mode with | None -> attributes | Some round_mode -> ("round_mode", String round_mode) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizeV2 in
  let inputs = [ (`single (P input)); (`single (P min_range)); (`single (P max_range)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedAdd
    ?(name = "QuantizedAdd")
    ~type_
    ?(control_inputs = [])
    (x : 't1 t)
    (y : 't2 t)
    (min_x : [ `float ] t)
    (max_x : [ `float ] t)
    (min_y : [ `float ] t)
    (max_y : [ `float ] t)
  =
  let attributes = [ "T2", Type (P (Node.output_type y)) ;  "T1", Type (P (Node.output_type x)) ;  "Toutput", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedAdd in
  let inputs = [ (`single (P x)); (`single (P y)); (`single (P min_x)); (`single (P max_x)); (`single (P min_y)); (`single (P max_y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedAvgPool
    ?(name = "QuantizedAvgPool")
    ~ksize
    ~strides
    ~padding
    ?(control_inputs = [])
    (input : 't t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("ksize", List (Int ksize)) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedAvgPool in
  let inputs = [ (`single (P input)); (`single (P min_input)); (`single (P max_input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedBatchNormWithGlobalNormalization
    ?(name = "QuantizedBatchNormWithGlobalNormalization")
    ~type_
    ~variance_epsilon
    ~scale_after_normalization
    ?(control_inputs = [])
    (t : 'tinput t)
    (t_min : [ `float ] t)
    (t_max : [ `float ] t)
    (m : 'tinput t)
    (m_min : [ `float ] t)
    (m_max : [ `float ] t)
    (v : 'tinput t)
    (v_min : [ `float ] t)
    (v_max : [ `float ] t)
    (beta : 'tinput t)
    (beta_min : [ `float ] t)
    (beta_max : [ `float ] t)
    (gamma : 'tinput t)
    (gamma_min : [ `float ] t)
    (gamma_max : [ `float ] t)
  =
  let attributes = [ "Tinput", Type (P (Node.output_type t)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("variance_epsilon", Float variance_epsilon) :: attributes
  in
  let attributes =
    ("scale_after_normalization", Bool scale_after_normalization) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedBatchNormWithGlobalNormalization in
  let inputs = [ (`single (P t)); (`single (P t_min)); (`single (P t_max)); (`single (P m)); (`single (P m_min)); (`single (P m_max)); (`single (P v)); (`single (P v_min)); (`single (P v_max)); (`single (P beta)); (`single (P beta_min)); (`single (P beta_max)); (`single (P gamma)); (`single (P gamma_min)); (`single (P gamma_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedBiasAdd
    ?(name = "QuantizedBiasAdd")
    ~type_
    ?(control_inputs = [])
    (input : 't1 t)
    (bias : 't2 t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_bias : [ `float ] t)
    (max_bias : [ `float ] t)
  =
  let attributes = [ "T2", Type (P (Node.output_type bias)) ;  "T1", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedBiasAdd in
  let inputs = [ (`single (P input)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_bias)); (`single (P max_bias)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConcat
    ?(name = "QuantizedConcat")
    ?(control_inputs = [])
    (concat_dim : [ `int32 ] t)
    (values : 't t list)
    (input_mins : [ `float ] t list)
    (input_maxes : [ `float ] t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn values))) ] in
  let attributes =
    ("N", Int (List.length values)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConcat in
  let inputs = [ (`single (P concat_dim)); (`multi (List.map ~f:(fun n -> P n) values)); (`multi (List.map ~f:(fun n -> P n) input_mins)); (`multi (List.map ~f:(fun n -> P n) input_maxes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn values))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2D
    ?(name = "QuantizedConv2D")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2D in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DAndRelu
    ?(name = "QuantizedConv2DAndRelu")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DAndRelu in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DAndReluAndRequantize
    ?(name = "QuantizedConv2DAndReluAndRequantize")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
    (min_freezed_output : [ `float ] t)
    (max_freezed_output : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DAndReluAndRequantize in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)); (`single (P min_freezed_output)); (`single (P max_freezed_output)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DAndRequantize
    ?(name = "QuantizedConv2DAndRequantize")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
    (min_freezed_output : [ `float ] t)
    (max_freezed_output : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DAndRequantize in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)); (`single (P min_freezed_output)); (`single (P max_freezed_output)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DPerChannel
    ?(name = "QuantizedConv2DPerChannel")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DPerChannel in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DWithBias
    ?(name = "QuantizedConv2DWithBias")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : [ `float ] t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DWithBias in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DWithBiasAndRelu
    ?(name = "QuantizedConv2DWithBiasAndRelu")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : [ `float ] t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DWithBiasAndRelu in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DWithBiasAndReluAndRequantize
    ?(name = "QuantizedConv2DWithBiasAndReluAndRequantize")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : ([< `float ] as 'tbias) t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
    (min_freezed_output : [ `float ] t)
    (max_freezed_output : [ `float ] t)
  =
  let attributes = [ "Tbias", Type (P (Node.output_type bias)) ;  "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DWithBiasAndReluAndRequantize in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)); (`single (P min_freezed_output)); (`single (P max_freezed_output)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DWithBiasAndRequantize
    ?(name = "QuantizedConv2DWithBiasAndRequantize")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : ([< `float ] as 'tbias) t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
    (min_freezed_output : [ `float ] t)
    (max_freezed_output : [ `float ] t)
  =
  let attributes = [ "Tbias", Type (P (Node.output_type bias)) ;  "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DWithBiasAndRequantize in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)); (`single (P min_freezed_output)); (`single (P max_freezed_output)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DWithBiasSignedSumAndReluAndRequantize
    ?(name = "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : ([< `float ] as 'tbias) t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
    (min_freezed_output : [ `float ] t)
    (max_freezed_output : [ `float ] t)
    (summand : 'tsummand t)
    (min_summand : [ `float ] t)
    (max_summand : [ `float ] t)
  =
  let attributes = [ "Tsummand", Type (P (Node.output_type summand)) ;  "Tbias", Type (P (Node.output_type bias)) ;  "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DWithBiasSignedSumAndReluAndRequantize in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)); (`single (P min_freezed_output)); (`single (P max_freezed_output)); (`single (P summand)); (`single (P min_summand)); (`single (P max_summand)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DWithBiasSumAndRelu
    ?(name = "QuantizedConv2DWithBiasSumAndRelu")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : [ `float ] t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
    (summand : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DWithBiasSumAndRelu in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)); (`single (P summand)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedConv2DWithBiasSumAndReluAndRequantize
    ?(name = "QuantizedConv2DWithBiasSumAndReluAndRequantize")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?padding_list
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : ([< `float ] as 'tbias) t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
    (min_freezed_output : [ `float ] t)
    (max_freezed_output : [ `float ] t)
    (summand : 'tsummand t)
    (min_summand : [ `float ] t)
    (max_summand : [ `float ] t)
  =
  let attributes = [ "Tsummand", Type (P (Node.output_type summand)) ;  "Tbias", Type (P (Node.output_type bias)) ;  "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let attributes =
    match padding_list with | None -> attributes | Some padding_list -> ("padding_list", List (Int padding_list)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedConv2DWithBiasSumAndReluAndRequantize in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)); (`single (P min_freezed_output)); (`single (P max_freezed_output)); (`single (P summand)); (`single (P min_summand)); (`single (P max_summand)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedDepthwiseConv2D
    ?(name = "QuantizedDepthwiseConv2D")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedDepthwiseConv2D in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedDepthwiseConv2DWithBias
    ?(name = "QuantizedDepthwiseConv2DWithBias")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : [ `float ] t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedDepthwiseConv2DWithBias in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedDepthwiseConv2DWithBiasAndRelu
    ?(name = "QuantizedDepthwiseConv2DWithBiasAndRelu")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : [ `float ] t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
  =
  let attributes = [ "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedDepthwiseConv2DWithBiasAndRelu in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedDepthwiseConv2DWithBiasAndReluAndRequantize
    ?(name = "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")
    ~type_
    ~strides
    ~padding
    ?dilations
    ?(control_inputs = [])
    (input : 'tinput t)
    (filter : 'tfilter t)
    (bias : ([< `float ] as 'tbias) t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
    (min_filter : [ `float ] t)
    (max_filter : [ `float ] t)
    (min_freezed_output : [ `float ] t)
    (max_freezed_output : [ `float ] t)
  =
  let attributes = [ "Tbias", Type (P (Node.output_type bias)) ;  "Tfilter", Type (P (Node.output_type filter)) ;  "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let attributes =
    match dilations with | None -> attributes | Some dilations -> ("dilations", List (Int dilations)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedDepthwiseConv2DWithBiasAndReluAndRequantize in
  let inputs = [ (`single (P input)); (`single (P filter)); (`single (P bias)); (`single (P min_input)); (`single (P max_input)); (`single (P min_filter)); (`single (P max_filter)); (`single (P min_freezed_output)); (`single (P max_freezed_output)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedInstanceNorm
    ?(name = "QuantizedInstanceNorm")
    ?output_range_given
    ?given_y_min
    ?given_y_max
    ?variance_epsilon
    ?min_separation
    ?(control_inputs = [])
    (x : 't t)
    (x_min : [ `float ] t)
    (x_max : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let attributes =
    match output_range_given with | None -> attributes | Some output_range_given -> ("output_range_given", Bool output_range_given) :: attributes
  in
  let attributes =
    match given_y_min with | None -> attributes | Some given_y_min -> ("given_y_min", Float given_y_min) :: attributes
  in
  let attributes =
    match given_y_max with | None -> attributes | Some given_y_max -> ("given_y_max", Float given_y_max) :: attributes
  in
  let attributes =
    match variance_epsilon with | None -> attributes | Some variance_epsilon -> ("variance_epsilon", Float variance_epsilon) :: attributes
  in
  let attributes =
    match min_separation with | None -> attributes | Some min_separation -> ("min_separation", Float min_separation) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedInstanceNorm in
  let inputs = [ (`single (P x)); (`single (P x_min)); (`single (P x_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedMatMul
    ?(name = "QuantizedMatMul")
    ~type_
    ?transpose_a
    ?transpose_b
    ?(control_inputs = [])
    (a : 't1 t)
    (b : 't2 t)
    (min_a : [ `float ] t)
    (max_a : [ `float ] t)
    (min_b : [ `float ] t)
    (max_b : [ `float ] t)
  =
  let attributes = [ "T2", Type (P (Node.output_type b)) ;  "T1", Type (P (Node.output_type a)) ;  "Toutput", Type (P type_) ] in
  let attributes =
    match transpose_a with | None -> attributes | Some transpose_a -> ("transpose_a", Bool transpose_a) :: attributes
  in
  let attributes =
    match transpose_b with | None -> attributes | Some transpose_b -> ("transpose_b", Bool transpose_b) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedMatMul in
  let inputs = [ (`single (P a)); (`single (P b)); (`single (P min_a)); (`single (P max_a)); (`single (P min_b)); (`single (P max_b)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedMaxPool
    ?(name = "QuantizedMaxPool")
    ~ksize
    ~strides
    ~padding
    ?(control_inputs = [])
    (input : 't t)
    (min_input : [ `float ] t)
    (max_input : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("ksize", List (Int ksize)) :: attributes
  in
  let attributes =
    ("strides", List (Int strides)) :: attributes
  in
  let attributes =
    ("padding", String padding) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedMaxPool in
  let inputs = [ (`single (P input)); (`single (P min_input)); (`single (P max_input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedMul
    ?(name = "QuantizedMul")
    ~type_
    ?(control_inputs = [])
    (x : 't1 t)
    (y : 't2 t)
    (min_x : [ `float ] t)
    (max_x : [ `float ] t)
    (min_y : [ `float ] t)
    (max_y : [ `float ] t)
  =
  let attributes = [ "T2", Type (P (Node.output_type y)) ;  "T1", Type (P (Node.output_type x)) ;  "Toutput", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedMul in
  let inputs = [ (`single (P x)); (`single (P y)); (`single (P min_x)); (`single (P max_x)); (`single (P min_y)); (`single (P max_y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedRelu
    ?(name = "QuantizedRelu")
    ~type_
    ?(control_inputs = [])
    (features : 'tinput t)
    (min_features : [ `float ] t)
    (max_features : [ `float ] t)
  =
  let attributes = [ "Tinput", Type (P (Node.output_type features)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedRelu in
  let inputs = [ (`single (P features)); (`single (P min_features)); (`single (P max_features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedRelu6
    ?(name = "QuantizedRelu6")
    ~type_
    ?(control_inputs = [])
    (features : 'tinput t)
    (min_features : [ `float ] t)
    (max_features : [ `float ] t)
  =
  let attributes = [ "Tinput", Type (P (Node.output_type features)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedRelu6 in
  let inputs = [ (`single (P features)); (`single (P min_features)); (`single (P max_features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedReluX
    ?(name = "QuantizedReluX")
    ~type_
    ?(control_inputs = [])
    (features : 'tinput t)
    (max_value : [ `float ] t)
    (min_features : [ `float ] t)
    (max_features : [ `float ] t)
  =
  let attributes = [ "Tinput", Type (P (Node.output_type features)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedReluX in
  let inputs = [ (`single (P features)); (`single (P max_value)); (`single (P min_features)); (`single (P max_features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedReshape
    ?(name = "QuantizedReshape")
    ?(control_inputs = [])
    (tensor : 't t)
    (shape : ([< `int32 | `int64 ] as 'tshape) t)
    (input_min : [ `float ] t)
    (input_max : [ `float ] t)
  =
  let attributes = [ "Tshape", Type (P (Node.output_type shape)) ;  "T", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedReshape in
  let inputs = [ (`single (P tensor)); (`single (P shape)); (`single (P input_min)); (`single (P input_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type tensor)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let quantizedResizeBilinear
    ?(name = "QuantizedResizeBilinear")
    ?align_corners
    ?half_pixel_centers
    ?(control_inputs = [])
    (images : ([< `float ] as 't) t)
    (size : [ `int32 ] t)
    (min : [ `float ] t)
    (max : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  let attributes =
    match half_pixel_centers with | None -> attributes | Some half_pixel_centers -> ("half_pixel_centers", Bool half_pixel_centers) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.quantizedResizeBilinear in
  let inputs = [ (`single (P images)); (`single (P size)); (`single (P min)); (`single (P max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let queueClose
    ?(name = "QueueClose")
    ?cancel_pending_enqueues
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match cancel_pending_enqueues with | None -> attributes | Some cancel_pending_enqueues -> ("cancel_pending_enqueues", Bool cancel_pending_enqueues) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.queueClose in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let queueIsClosed
    ?(name = "QueueIsClosed")
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.queueIsClosed in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let queueSize
    ?(name = "QueueSize")
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.queueSize in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rFFT
    ?(name = "RFFT")
    ?(control_inputs = [])
    (input : [ `float ] t)
    (fft_length : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.rFFT in
  let inputs = [ (`single (P input)); (`single (P fft_length)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Complex64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rFFT2D
    ?(name = "RFFT2D")
    ?(control_inputs = [])
    (input : [ `float ] t)
    (fft_length : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.rFFT2D in
  let inputs = [ (`single (P input)); (`single (P fft_length)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Complex64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rFFT3D
    ?(name = "RFFT3D")
    ?(control_inputs = [])
    (input : [ `float ] t)
    (fft_length : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.rFFT3D in
  let inputs = [ (`single (P input)); (`single (P fft_length)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Complex64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rGBToHSV
    ?(name = "RGBToHSV")
    ?(control_inputs = [])
    (images : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.rGBToHSV in
  let inputs = [ (`single (P images)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let raggedRange
    ?(name = "RaggedRange")
    ?(control_inputs = [])
    (starts : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (limits : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (deltas : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type starts)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.raggedRange in
  let inputs = [ (`single (P starts)); (`single (P limits)); (`single (P deltas)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type starts)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let raggedTensorToSparse
    ?(name = "RaggedTensorToSparse")
    ?(control_inputs = [])
    (rt_nested_splits : [ `int64 ] t list)
    (rt_dense_values : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type rt_dense_values)) ] in
  let attributes =
    ("RAGGED_RANK", Int (List.length rt_nested_splits)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.raggedTensorToSparse in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) rt_nested_splits)); (`single (P rt_dense_values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type rt_dense_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let randomCrop
    ?(name = "RandomCrop")
    ?seed
    ?seed2
    ?(control_inputs = [])
    (image : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type image)) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.randomCrop in
  let inputs = [ (`single (P image)); (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type image)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let randomGamma
    ?(name = "RandomGamma")
    ?seed
    ?seed2
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 's) t)
    (alpha : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "S", Type (P (Node.output_type shape)) ;  "T", Type (P (Node.output_type alpha)) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.randomGamma in
  let inputs = [ (`single (P shape)); (`single (P alpha)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type alpha)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let randomGammaGrad
    ?(name = "RandomGammaGrad")
    ?(control_inputs = [])
    (alpha : ([< `float | `double ] as 't) t)
    (sample : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type alpha)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.randomGammaGrad in
  let inputs = [ (`single (P alpha)); (`single (P sample)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type alpha)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let randomPoisson
    ?(name = "RandomPoisson")
    ?seed
    ?seed2
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 's) t)
    (rate : ([< `float | `double ] as 'dtype) t)
  =
  let attributes = [ "S", Type (P (Node.output_type shape)) ;  "dtype", Type (P (Node.output_type rate)) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.randomPoisson in
  let inputs = [ (`single (P shape)); (`single (P rate)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type rate)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let randomPoissonV2
    ?(name = "RandomPoissonV2")
    ~type_
    ?seed
    ?seed2
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 's) t)
    (rate : ([< `float | `double | `int32 | `int64 ] as 'r) t)
  =
  let attributes = [ "R", Type (P (Node.output_type rate)) ;  "S", Type (P (Node.output_type shape)) ;  "dtype", Type (P type_) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.randomPoissonV2 in
  let inputs = [ (`single (P shape)); (`single (P rate)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let randomShuffle
    ?(name = "RandomShuffle")
    ?seed
    ?seed2
    ?(control_inputs = [])
    (value : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.randomShuffle in
  let inputs = [ (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type value)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

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
    ?(control_inputs = [])
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
  let name = Name.of_string name in
  let op_name = Op_names.randomShuffleQueue in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let randomStandardNormal
    ?(name = "RandomStandardNormal")
    ~type_
    ?seed
    ?seed2
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type shape)) ;  "dtype", Type (P type_) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.randomStandardNormal in
  let inputs = [ (`single (P shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let randomUniform
    ?(name = "RandomUniform")
    ~type_
    ?seed
    ?seed2
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type shape)) ;  "dtype", Type (P type_) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.randomUniform in
  let inputs = [ (`single (P shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let randomUniformInt
    ?(name = "RandomUniformInt")
    ?seed
    ?seed2
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 't) t)
    (minval : ([< `int32 | `int64 ] as 'tout) t)
    (maxval : ([< `int32 | `int64 ] as 'tout) t)
  =
  let attributes = [ "T", Type (P (Node.output_type shape)) ;  "Tout", Type (P (Node.output_type minval)) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.randomUniformInt in
  let inputs = [ (`single (P shape)); (`single (P minval)); (`single (P maxval)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type minval)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let range
    ?(name = "Range")
    ?(control_inputs = [])
    (start : ([< `float | `double | `int32 | `int64 ] as 'tidx) t)
    (limit : ([< `float | `double | `int32 | `int64 ] as 'tidx) t)
    (delta : ([< `float | `double | `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type start)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.range in
  let inputs = [ (`single (P start)); (`single (P limit)); (`single (P delta)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type start)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rangeDataset
    ?(name = "RangeDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (start : [ `int64 ] t)
    (stop : [ `int64 ] t)
    (step : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.rangeDataset in
  let inputs = [ (`single (P start)); (`single (P stop)); (`single (P step)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rank
    ?(name = "Rank")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.rank in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let readFile
    ?(name = "ReadFile")
    ?(control_inputs = [])
    (filename : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.readFile in
  let inputs = [ (`single (P filename)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let readerNumRecordsProduced
    ?(name = "ReaderNumRecordsProduced")
    ?(control_inputs = [])
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.readerNumRecordsProduced in
  let inputs = [ (`single (P reader_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let readerNumWorkUnitsCompleted
    ?(name = "ReaderNumWorkUnitsCompleted")
    ?(control_inputs = [])
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.readerNumWorkUnitsCompleted in
  let inputs = [ (`single (P reader_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let readerRead
    ?(name = "ReaderRead")
    ?(control_inputs = [])
    (reader_handle : [ `string ] t)
    (queue_handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.readerRead in
  let inputs = [ (`single (P reader_handle)); (`single (P queue_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let readerReadUpTo
    ?(name = "ReaderReadUpTo")
    ?(control_inputs = [])
    (reader_handle : [ `string ] t)
    (queue_handle : [ `string ] t)
    (num_records : [ `int64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.readerReadUpTo in
  let inputs = [ (`single (P reader_handle)); (`single (P queue_handle)); (`single (P num_records)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let readerReset
    ?(name = "ReaderReset")
    ?(control_inputs = [])
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.readerReset in
  let inputs = [ (`single (P reader_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let readerRestoreState
    ?(name = "ReaderRestoreState")
    ?(control_inputs = [])
    (reader_handle : [ `string ] t)
    (state : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.readerRestoreState in
  let inputs = [ (`single (P reader_handle)); (`single (P state)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let readerSerializeState
    ?(name = "ReaderSerializeState")
    ?(control_inputs = [])
    (reader_handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.readerSerializeState in
  let inputs = [ (`single (P reader_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let real
    ?(name = "Real")
    ~type_
    ?(control_inputs = [])
    (input : ([< `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "Tout", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.real in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let realDiv
    ?(name = "RealDiv")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.realDiv in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let reciprocal
    ?(name = "Reciprocal")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.reciprocal in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let reciprocalGrad
    ?(name = "ReciprocalGrad")
    ?(control_inputs = [])
    (y : ([< `float | `double | `complex64 ] as 't) t)
    (dy : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type y)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.reciprocalGrad in
  let inputs = [ (`single (P y)); (`single (P dy)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let recordInput
    ?(name = "RecordInput")
    ~file_pattern
    ?file_random_seed
    ?file_shuffle_shift_ratio
    ?file_buffer_size
    ?file_parallelism
    ?batch_size
    ?compression_type
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    ("file_pattern", String file_pattern) :: attributes
  in
  let attributes =
    match file_random_seed with | None -> attributes | Some file_random_seed -> ("file_random_seed", Int file_random_seed) :: attributes
  in
  let attributes =
    match file_shuffle_shift_ratio with | None -> attributes | Some file_shuffle_shift_ratio -> ("file_shuffle_shift_ratio", Float file_shuffle_shift_ratio) :: attributes
  in
  let attributes =
    match file_buffer_size with | None -> attributes | Some file_buffer_size -> ("file_buffer_size", Int file_buffer_size) :: attributes
  in
  let attributes =
    match file_parallelism with | None -> attributes | Some file_parallelism -> ("file_parallelism", Int file_parallelism) :: attributes
  in
  let attributes =
    match batch_size with | None -> attributes | Some batch_size -> ("batch_size", Int batch_size) :: attributes
  in
  let attributes =
    match compression_type with | None -> attributes | Some compression_type -> ("compression_type", String compression_type) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.recordInput in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let recvTPUEmbeddingActivations
    ?(name = "RecvTPUEmbeddingActivations")
    ~num_outputs
    ~config
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    ("num_outputs", Int num_outputs) :: attributes
  in
  let attributes =
    ("config", String config) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.recvTPUEmbeddingActivations in
  let inputs = [  ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:Type.Float
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init num_outputs ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let reduceJoin
    ?(name = "ReduceJoin")
    ?keep_dims
    ?separator
    ?(control_inputs = [])
    (inputs__ : [ `string ] t)
    (reduction_indices : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let attributes =
    match separator with | None -> attributes | Some separator -> ("separator", String separator) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.reduceJoin in
  let inputs = [ (`single (P inputs__)); (`single (P reduction_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let refEnter
    ?(name = "RefEnter")
    ~frame_name
    ?is_constant
    ?parallel_iterations
    ?(control_inputs = [])
    (data : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type data)) ] in
  let attributes =
    ("frame_name", String frame_name) :: attributes
  in
  let attributes =
    match is_constant with | None -> attributes | Some is_constant -> ("is_constant", Bool is_constant) :: attributes
  in
  let attributes =
    match parallel_iterations with | None -> attributes | Some parallel_iterations -> ("parallel_iterations", Int parallel_iterations) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.refEnter in
  let inputs = [ (`single (P data)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let refExit
    ?(name = "RefExit")
    ?(control_inputs = [])
    (data : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.refExit in
  let inputs = [ (`single (P data)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let refIdentity
    ?(name = "RefIdentity")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.refIdentity in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let refMerge
    ?(name = "RefMerge")
    ?(control_inputs = [])
    (inputs__ : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn inputs__))) ] in
  let attributes =
    ("N", Int (List.length inputs__)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.refMerge in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn inputs__))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let refNextIteration
    ?(name = "RefNextIteration")
    ?(control_inputs = [])
    (data : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.refNextIteration in
  let inputs = [ (`single (P data)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let refSelect
    ?(name = "RefSelect")
    ?(control_inputs = [])
    (index : [ `int32 ] t)
    (inputs__ : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn inputs__))) ] in
  let attributes =
    ("N", Int (List.length inputs__)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.refSelect in
  let inputs = [ (`single (P index)); (`multi (List.map ~f:(fun n -> P n) inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn inputs__))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let refSwitch
    ?(name = "RefSwitch")
    ?(control_inputs = [])
    (data : 't t)
    (pred : [ `bool ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type data)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.refSwitch in
  let inputs = [ (`single (P data)); (`single (P pred)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let regexFullMatch
    ?(name = "RegexFullMatch")
    ?(control_inputs = [])
    (input : [ `string ] t)
    (pattern : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.regexFullMatch in
  let inputs = [ (`single (P input)); (`single (P pattern)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let regexReplace
    ?(name = "RegexReplace")
    ?replace_global
    ?(control_inputs = [])
    (input : [ `string ] t)
    (pattern : [ `string ] t)
    (rewrite : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match replace_global with | None -> attributes | Some replace_global -> ("replace_global", Bool replace_global) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.regexReplace in
  let inputs = [ (`single (P input)); (`single (P pattern)); (`single (P rewrite)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let relu
    ?(name = "Relu")
    ?(control_inputs = [])
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type features)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.relu in
  let inputs = [ (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let relu6
    ?(name = "Relu6")
    ?(control_inputs = [])
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type features)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.relu6 in
  let inputs = [ (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let relu6Grad
    ?(name = "Relu6Grad")
    ?(control_inputs = [])
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type gradients)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.relu6Grad in
  let inputs = [ (`single (P gradients)); (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type gradients)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let reluGrad
    ?(name = "ReluGrad")
    ?(control_inputs = [])
    (gradients : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (features : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type gradients)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.reluGrad in
  let inputs = [ (`single (P gradients)); (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type gradients)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let repeatDataset
    ?(name = "RepeatDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (count : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.repeatDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P count)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let requantizationRange
    ?(name = "RequantizationRange")
    ?(control_inputs = [])
    (input : 'tinput t)
    (input_min : [ `float ] t)
    (input_max : [ `float ] t)
  =
  let attributes = [ "Tinput", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.requantizationRange in
  let inputs = [ (`single (P input)); (`single (P input_min)); (`single (P input_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let requantizationRangePerChannel
    ?(name = "RequantizationRangePerChannel")
    ~clip_value_max
    ?(control_inputs = [])
    (input : 't t)
    (input_min : [ `float ] t)
    (input_max : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("clip_value_max", Float clip_value_max) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.requantizationRangePerChannel in
  let inputs = [ (`single (P input)); (`single (P input_min)); (`single (P input_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let requantize
    ?(name = "Requantize")
    ~type_
    ?(control_inputs = [])
    (input : 'tinput t)
    (input_min : [ `float ] t)
    (input_max : [ `float ] t)
    (requested_output_min : [ `float ] t)
    (requested_output_max : [ `float ] t)
  =
  let attributes = [ "Tinput", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.requantize in
  let inputs = [ (`single (P input)); (`single (P input_min)); (`single (P input_max)); (`single (P requested_output_min)); (`single (P requested_output_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let requantizePerChannel
    ?(name = "RequantizePerChannel")
    ~type_
    ?(control_inputs = [])
    (input : 't t)
    (input_min : [ `float ] t)
    (input_max : [ `float ] t)
    (requested_output_min : [ `float ] t)
    (requested_output_max : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.requantizePerChannel in
  let inputs = [ (`single (P input)); (`single (P input_min)); (`single (P input_max)); (`single (P requested_output_min)); (`single (P requested_output_max)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let reshape
    ?(name = "Reshape")
    ?(control_inputs = [])
    (tensor : 't t)
    (shape : ([< `int32 | `int64 ] as 'tshape) t)
  =
  let attributes = [ "Tshape", Type (P (Node.output_type shape)) ;  "T", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.reshape in
  let inputs = [ (`single (P tensor)); (`single (P shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type tensor)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let resizeArea
    ?(name = "ResizeArea")
    ?align_corners
    ?(control_inputs = [])
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.resizeArea in
  let inputs = [ (`single (P images)); (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let resizeBicubic
    ?(name = "ResizeBicubic")
    ?align_corners
    ?half_pixel_centers
    ?(control_inputs = [])
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  let attributes =
    match half_pixel_centers with | None -> attributes | Some half_pixel_centers -> ("half_pixel_centers", Bool half_pixel_centers) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.resizeBicubic in
  let inputs = [ (`single (P images)); (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let resizeBicubicGrad
    ?(name = "ResizeBicubicGrad")
    ?align_corners
    ?half_pixel_centers
    ?(control_inputs = [])
    (grads : [ `float ] t)
    (original_image : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type original_image)) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  let attributes =
    match half_pixel_centers with | None -> attributes | Some half_pixel_centers -> ("half_pixel_centers", Bool half_pixel_centers) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.resizeBicubicGrad in
  let inputs = [ (`single (P grads)); (`single (P original_image)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type original_image)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let resizeBilinear
    ?(name = "ResizeBilinear")
    ?align_corners
    ?half_pixel_centers
    ?(control_inputs = [])
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  let attributes =
    match half_pixel_centers with | None -> attributes | Some half_pixel_centers -> ("half_pixel_centers", Bool half_pixel_centers) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.resizeBilinear in
  let inputs = [ (`single (P images)); (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let resizeBilinearGrad
    ?(name = "ResizeBilinearGrad")
    ?align_corners
    ?half_pixel_centers
    ?(control_inputs = [])
    (grads : [ `float ] t)
    (original_image : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type original_image)) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  let attributes =
    match half_pixel_centers with | None -> attributes | Some half_pixel_centers -> ("half_pixel_centers", Bool half_pixel_centers) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.resizeBilinearGrad in
  let inputs = [ (`single (P grads)); (`single (P original_image)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type original_image)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let resizeNearestNeighbor
    ?(name = "ResizeNearestNeighbor")
    ?align_corners
    ?half_pixel_centers
    ?(control_inputs = [])
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  let attributes =
    match half_pixel_centers with | None -> attributes | Some half_pixel_centers -> ("half_pixel_centers", Bool half_pixel_centers) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.resizeNearestNeighbor in
  let inputs = [ (`single (P images)); (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type images)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let resizeNearestNeighborGrad
    ?(name = "ResizeNearestNeighborGrad")
    ?align_corners
    ?half_pixel_centers
    ?(control_inputs = [])
    (grads : ([< `int32 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type grads)) ] in
  let attributes =
    match align_corners with | None -> attributes | Some align_corners -> ("align_corners", Bool align_corners) :: attributes
  in
  let attributes =
    match half_pixel_centers with | None -> attributes | Some half_pixel_centers -> ("half_pixel_centers", Bool half_pixel_centers) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.resizeNearestNeighborGrad in
  let inputs = [ (`single (P grads)); (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type grads)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let restore
    ?(name = "Restore")
    ~type_
    ?preferred_shard
    ?(control_inputs = [])
    (file_pattern : [ `string ] t)
    (tensor_name : [ `string ] t)
  =
  let attributes = [ "dt", Type (P type_) ] in
  let attributes =
    match preferred_shard with | None -> attributes | Some preferred_shard -> ("preferred_shard", Int preferred_shard) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.restore in
  let inputs = [ (`single (P file_pattern)); (`single (P tensor_name)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let restoreSlice
    ?(name = "RestoreSlice")
    ~type_
    ?preferred_shard
    ?(control_inputs = [])
    (file_pattern : [ `string ] t)
    (tensor_name : [ `string ] t)
    (shape_and_slice : [ `string ] t)
  =
  let attributes = [ "dt", Type (P type_) ] in
  let attributes =
    match preferred_shard with | None -> attributes | Some preferred_shard -> ("preferred_shard", Int preferred_shard) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.restoreSlice in
  let inputs = [ (`single (P file_pattern)); (`single (P tensor_name)); (`single (P shape_and_slice)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let retrieveTPUEmbeddingADAMParameters
    ?(name = "RetrieveTPUEmbeddingADAMParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingADAMParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let retrieveTPUEmbeddingADAMParametersGradAccumDebug
    ?(name = "RetrieveTPUEmbeddingADAMParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingADAMParametersGradAccumDebug in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let retrieveTPUEmbeddingAdadeltaParameters
    ?(name = "RetrieveTPUEmbeddingAdadeltaParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingAdadeltaParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let retrieveTPUEmbeddingAdadeltaParametersGradAccumDebug
    ?(name = "RetrieveTPUEmbeddingAdadeltaParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingAdadeltaParametersGradAccumDebug in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let retrieveTPUEmbeddingAdagradParameters
    ?(name = "RetrieveTPUEmbeddingAdagradParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingAdagradParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let retrieveTPUEmbeddingAdagradParametersGradAccumDebug
    ?(name = "RetrieveTPUEmbeddingAdagradParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingAdagradParametersGradAccumDebug in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let retrieveTPUEmbeddingCenteredRMSPropParameters
    ?(name = "RetrieveTPUEmbeddingCenteredRMSPropParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingCenteredRMSPropParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let retrieveTPUEmbeddingFTRLParameters
    ?(name = "RetrieveTPUEmbeddingFTRLParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingFTRLParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let retrieveTPUEmbeddingFTRLParametersGradAccumDebug
    ?(name = "RetrieveTPUEmbeddingFTRLParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingFTRLParametersGradAccumDebug in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let retrieveTPUEmbeddingMDLAdagradLightParameters
    ?(name = "RetrieveTPUEmbeddingMDLAdagradLightParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingMDLAdagradLightParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let retrieveTPUEmbeddingMomentumParameters
    ?(name = "RetrieveTPUEmbeddingMomentumParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingMomentumParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let retrieveTPUEmbeddingMomentumParametersGradAccumDebug
    ?(name = "RetrieveTPUEmbeddingMomentumParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingMomentumParametersGradAccumDebug in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let retrieveTPUEmbeddingProximalAdagradParameters
    ?(name = "RetrieveTPUEmbeddingProximalAdagradParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingProximalAdagradParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let retrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug
    ?(name = "RetrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let retrieveTPUEmbeddingRMSPropParameters
    ?(name = "RetrieveTPUEmbeddingRMSPropParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingRMSPropParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let retrieveTPUEmbeddingRMSPropParametersGradAccumDebug
    ?(name = "RetrieveTPUEmbeddingRMSPropParametersGradAccumDebug")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingRMSPropParametersGradAccumDebug in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let retrieveTPUEmbeddingStochasticGradientDescentParameters
    ?(name = "RetrieveTPUEmbeddingStochasticGradientDescentParameters")
    ?table_id
    ?table_name
    ~num_shards
    ~shard_id
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match table_id with | None -> attributes | Some table_id -> ("table_id", Int table_id) :: attributes
  in
  let attributes =
    match table_name with | None -> attributes | Some table_name -> ("table_name", String table_name) :: attributes
  in
  let attributes =
    ("num_shards", Int num_shards) :: attributes
  in
  let attributes =
    ("shard_id", Int shard_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.retrieveTPUEmbeddingStochasticGradientDescentParameters in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let reverse
    ?(name = "Reverse")
    ?(control_inputs = [])
    (tensor : ([< `int32 | `int64 | `bool | `float | `double | `complex64 | `string ] as 't) t)
    (dims : [ `bool ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.reverse in
  let inputs = [ (`single (P tensor)); (`single (P dims)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type tensor)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let reverseSequence
    ?(name = "ReverseSequence")
    ~seq_dim
    ?batch_dim
    ?(control_inputs = [])
    (input : 't t)
    (seq_lengths : ([< `int32 | `int64 ] as 'tlen) t)
  =
  let attributes = [ "Tlen", Type (P (Node.output_type seq_lengths)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("seq_dim", Int seq_dim) :: attributes
  in
  let attributes =
    match batch_dim with | None -> attributes | Some batch_dim -> ("batch_dim", Int batch_dim) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.reverseSequence in
  let inputs = [ (`single (P input)); (`single (P seq_lengths)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let reverseV2
    ?(name = "ReverseV2")
    ?(control_inputs = [])
    (tensor : ([< `int32 | `int64 | `bool | `float | `double | `complex64 | `string ] as 't) t)
    (axis : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type axis)) ;  "T", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.reverseV2 in
  let inputs = [ (`single (P tensor)); (`single (P axis)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type tensor)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rightShift
    ?(name = "RightShift")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 ] as 't) t)
    (y : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.rightShift in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rint
    ?(name = "Rint")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.rint in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let roll
    ?(name = "Roll")
    ?(control_inputs = [])
    (input : 't t)
    (shift : ([< `int32 | `int64 ] as 'tshift) t)
    (axis : ([< `int32 | `int64 ] as 'taxis) t)
  =
  let attributes = [ "Taxis", Type (P (Node.output_type axis)) ;  "Tshift", Type (P (Node.output_type shift)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.roll in
  let inputs = [ (`single (P input)); (`single (P shift)); (`single (P axis)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let round
    ?(name = "Round")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.round in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rpc
    ?(name = "Rpc")
    ?protocol
    ?fail_fast
    ?timeout_in_ms
    ?(control_inputs = [])
    (address : [ `string ] t)
    (method__ : [ `string ] t)
    (request : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match protocol with | None -> attributes | Some protocol -> ("protocol", String protocol) :: attributes
  in
  let attributes =
    match fail_fast with | None -> attributes | Some fail_fast -> ("fail_fast", Bool fail_fast) :: attributes
  in
  let attributes =
    match timeout_in_ms with | None -> attributes | Some timeout_in_ms -> ("timeout_in_ms", Int timeout_in_ms) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.rpc in
  let inputs = [ (`single (P address)); (`single (P method__)); (`single (P request)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rsqrt
    ?(name = "Rsqrt")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.rsqrt in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let rsqrtGrad
    ?(name = "RsqrtGrad")
    ?(control_inputs = [])
    (y : ([< `float | `double | `complex64 ] as 't) t)
    (dy : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type y)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.rsqrtGrad in
  let inputs = [ (`single (P y)); (`single (P dy)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sampleDistortedBoundingBox
    ?(name = "SampleDistortedBoundingBox")
    ?seed
    ?seed2
    ?min_object_covered
    ?aspect_ratio_range
    ?area_range
    ?max_attempts
    ?use_image_if_no_bounding_boxes
    ?(control_inputs = [])
    (image_size : ([< `int32 | `int64 ] as 't) t)
    (bounding_boxes : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type image_size)) ;  "T", Type (P (Node.output_type image_size)) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let attributes =
    match min_object_covered with | None -> attributes | Some min_object_covered -> ("min_object_covered", Float min_object_covered) :: attributes
  in
  let attributes =
    match aspect_ratio_range with | None -> attributes | Some aspect_ratio_range -> ("aspect_ratio_range", List (Float aspect_ratio_range)) :: attributes
  in
  let attributes =
    match area_range with | None -> attributes | Some area_range -> ("area_range", List (Float area_range)) :: attributes
  in
  let attributes =
    match max_attempts with | None -> attributes | Some max_attempts -> ("max_attempts", Int max_attempts) :: attributes
  in
  let attributes =
    match use_image_if_no_bounding_boxes with | None -> attributes | Some use_image_if_no_bounding_boxes -> ("use_image_if_no_bounding_boxes", Bool use_image_if_no_bounding_boxes) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sampleDistortedBoundingBox in
  let inputs = [ (`single (P image_size)); (`single (P bounding_boxes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type image_size)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type image_size)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let sampleDistortedBoundingBoxV2
    ?(name = "SampleDistortedBoundingBoxV2")
    ?seed
    ?seed2
    ?aspect_ratio_range
    ?area_range
    ?max_attempts
    ?use_image_if_no_bounding_boxes
    ?(control_inputs = [])
    (image_size : ([< `int32 | `int64 ] as 't) t)
    (bounding_boxes : [ `float ] t)
    (min_object_covered : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type image_size)) ;  "T", Type (P (Node.output_type image_size)) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let attributes =
    match aspect_ratio_range with | None -> attributes | Some aspect_ratio_range -> ("aspect_ratio_range", List (Float aspect_ratio_range)) :: attributes
  in
  let attributes =
    match area_range with | None -> attributes | Some area_range -> ("area_range", List (Float area_range)) :: attributes
  in
  let attributes =
    match max_attempts with | None -> attributes | Some max_attempts -> ("max_attempts", Int max_attempts) :: attributes
  in
  let attributes =
    match use_image_if_no_bounding_boxes with | None -> attributes | Some use_image_if_no_bounding_boxes -> ("use_image_if_no_bounding_boxes", Bool use_image_if_no_bounding_boxes) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sampleDistortedBoundingBoxV2 in
  let inputs = [ (`single (P image_size)); (`single (P bounding_boxes)); (`single (P min_object_covered)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type image_size)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type image_size)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let samplingDataset
    ?(name = "SamplingDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (rate : [ `float ] t)
    (seed : [ `int64 ] t)
    (seed2 : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.samplingDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P rate)); (`single (P seed)); (`single (P seed2)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scalarSummary
    ?(name = "ScalarSummary")
    ?(control_inputs = [])
    (tags : [ `string ] t)
    (values : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.scalarSummary in
  let inputs = [ (`single (P tags)); (`single (P values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scaleAndTranslate
    ?(name = "ScaleAndTranslate")
    ?kernel_type
    ?antialias
    ?(control_inputs = [])
    (images : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (size : [ `int32 ] t)
    (scale : [ `float ] t)
    (translation : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type images)) ] in
  let attributes =
    match kernel_type with | None -> attributes | Some kernel_type -> ("kernel_type", String kernel_type) :: attributes
  in
  let attributes =
    match antialias with | None -> attributes | Some antialias -> ("antialias", Bool antialias) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scaleAndTranslate in
  let inputs = [ (`single (P images)); (`single (P size)); (`single (P scale)); (`single (P translation)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scaleAndTranslateGrad
    ?(name = "ScaleAndTranslateGrad")
    ?kernel_type
    ?antialias
    ?(control_inputs = [])
    (grads : ([< `float ] as 't) t)
    (original_image : ([< `float ] as 't) t)
    (scale : [ `float ] t)
    (translation : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type grads)) ] in
  let attributes =
    match kernel_type with | None -> attributes | Some kernel_type -> ("kernel_type", String kernel_type) :: attributes
  in
  let attributes =
    match antialias with | None -> attributes | Some antialias -> ("antialias", Bool antialias) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scaleAndTranslateGrad in
  let inputs = [ (`single (P grads)); (`single (P original_image)); (`single (P scale)); (`single (P translation)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type grads)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterAdd
    ?(name = "ScatterAdd")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterAdd in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterDiv
    ?(name = "ScatterDiv")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterDiv in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterMax
    ?(name = "ScatterMax")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterMax in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterMin
    ?(name = "ScatterMin")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterMin in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterMul
    ?(name = "ScatterMul")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterMul in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterNd
    ?(name = "ScatterNd")
    ?(control_inputs = [])
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : 't t)
    (shape : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type updates)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.scatterNd in
  let inputs = [ (`single (P indices)); (`single (P updates)); (`single (P shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type updates)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterNdAdd
    ?(name = "ScatterNdAdd")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterNdAdd in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterNdNonAliasingAdd
    ?(name = "ScatterNdNonAliasingAdd")
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.scatterNdNonAliasingAdd in
  let inputs = [ (`single (P input)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterNdSub
    ?(name = "ScatterNdSub")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterNdSub in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterNdUpdate
    ?(name = "ScatterNdUpdate")
    ?use_locking
    ?(control_inputs = [])
    (ref : 't t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : 't t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterNdUpdate in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterSub
    ?(name = "ScatterSub")
    ?use_locking
    ?(control_inputs = [])
    (ref : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterSub in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let scatterUpdate
    ?(name = "ScatterUpdate")
    ?use_locking
    ?(control_inputs = [])
    (ref : 't t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : 't t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.scatterUpdate in
  let inputs = [ (`single (P ref)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sdcaFprint
    ?(name = "SdcaFprint")
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.sdcaFprint in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sdcaShrinkL1
    ?(name = "SdcaShrinkL1")
    ~l1
    ~l2
    ?(control_inputs = [])
    (weights : [ `float ] t list)
  =
  let attributes = [] in
  let attributes =
    ("num_features", Int (List.length weights)) :: attributes
  in
  let attributes =
    ("l1", Float l1) :: attributes
  in
  let attributes =
    ("l2", Float l2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sdcaShrinkL1 in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) weights)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let segmentMax
    ?(name = "SegmentMax")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type segment_ids)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.segmentMax in
  let inputs = [ (`single (P data)); (`single (P segment_ids)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let segmentMean
    ?(name = "SegmentMean")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type segment_ids)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.segmentMean in
  let inputs = [ (`single (P data)); (`single (P segment_ids)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let segmentMin
    ?(name = "SegmentMin")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type segment_ids)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.segmentMin in
  let inputs = [ (`single (P data)); (`single (P segment_ids)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let segmentProd
    ?(name = "SegmentProd")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type segment_ids)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.segmentProd in
  let inputs = [ (`single (P data)); (`single (P segment_ids)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let segmentSum
    ?(name = "SegmentSum")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type segment_ids)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.segmentSum in
  let inputs = [ (`single (P data)); (`single (P segment_ids)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let select
    ?(name = "Select")
    ?(control_inputs = [])
    (condition : [ `bool ] t)
    (t : 't t)
    (e : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type t)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.select in
  let inputs = [ (`single (P condition)); (`single (P t)); (`single (P e)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type t)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let selfAdjointEig
    ?(name = "SelfAdjointEig")
    ?(control_inputs = [])
    (input : ([< `double | `float ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.selfAdjointEig in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let selfAdjointEigV2
    ?(name = "SelfAdjointEigV2")
    ?compute_v
    ?(control_inputs = [])
    (input : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match compute_v with | None -> attributes | Some compute_v -> ("compute_v", Bool compute_v) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.selfAdjointEigV2 in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let selu
    ?(name = "Selu")
    ?(control_inputs = [])
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type features)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.selu in
  let inputs = [ (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let seluGrad
    ?(name = "SeluGrad")
    ?(control_inputs = [])
    (gradients : ([< `float | `double ] as 't) t)
    (outputs : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type gradients)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.seluGrad in
  let inputs = [ (`single (P gradients)); (`single (P outputs)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type gradients)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sendTPUEmbeddingGradients
    ?(name = "SendTPUEmbeddingGradients")
    ~config
    ?(control_inputs = [])
    (inputs__ : [ `float ] t list)
    (learning_rates : [ `float ] t list)
  =
  let attributes = [] in
  let attributes =
    ("N", Int (List.length inputs__)) :: attributes
  in
  let attributes =
    ("NN", Int (List.length learning_rates)) :: attributes
  in
  let attributes =
    ("config", String config) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sendTPUEmbeddingGradients in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) inputs__)); (`multi (List.map ~f:(fun n -> P n) learning_rates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let serializeManySparse
    ?(name = "SerializeManySparse")
    ~type_
    ?(control_inputs = [])
    (sparse_indices : [ `int64 ] t)
    (sparse_values : 't t)
    (sparse_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type sparse_values)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.serializeManySparse in
  let inputs = [ (`single (P sparse_indices)); (`single (P sparse_values)); (`single (P sparse_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let serializeSparse
    ?(name = "SerializeSparse")
    ~type_
    ?(control_inputs = [])
    (sparse_indices : [ `int64 ] t)
    (sparse_values : 't t)
    (sparse_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type sparse_values)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.serializeSparse in
  let inputs = [ (`single (P sparse_indices)); (`single (P sparse_values)); (`single (P sparse_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let serializeTensor
    ?(name = "SerializeTensor")
    ?(control_inputs = [])
    (tensor : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.serializeTensor in
  let inputs = [ (`single (P tensor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let setSize
    ?(name = "SetSize")
    ?validate_indices
    ?(control_inputs = [])
    (set_indices : [ `int64 ] t)
    (set_values : ([< `int32 | `int64 | `string ] as 't) t)
    (set_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type set_values)) ] in
  let attributes =
    match validate_indices with | None -> attributes | Some validate_indices -> ("validate_indices", Bool validate_indices) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.setSize in
  let inputs = [ (`single (P set_indices)); (`single (P set_values)); (`single (P set_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let shape
    ?(name = "Shape")
    ~type_
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.shape in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let shapeN
    ?(name = "ShapeN")
    ~type_
    ?(control_inputs = [])
    (input : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn input))) ;  "out_type", Type (P type_) ] in
  let attributes =
    ("N", Int (List.length input)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.shapeN in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) input)) ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:type_
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init (List.length input) ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let shardDataset
    ?(name = "ShardDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (num_shards : [ `int64 ] t)
    (index : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.shardDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P num_shards)); (`single (P index)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let shardedFilename
    ?(name = "ShardedFilename")
    ?(control_inputs = [])
    (basename : [ `string ] t)
    (shard : [ `int32 ] t)
    (num_shards : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.shardedFilename in
  let inputs = [ (`single (P basename)); (`single (P shard)); (`single (P num_shards)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let shardedFilespec
    ?(name = "ShardedFilespec")
    ?(control_inputs = [])
    (basename : [ `string ] t)
    (num_shards : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.shardedFilespec in
  let inputs = [ (`single (P basename)); (`single (P num_shards)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let shuffleAndRepeatDataset
    ?(name = "ShuffleAndRepeatDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (buffer_size : [ `int64 ] t)
    (seed : [ `int64 ] t)
    (seed2 : [ `int64 ] t)
    (count : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.shuffleAndRepeatDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P buffer_size)); (`single (P seed)); (`single (P seed2)); (`single (P count)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let shuffleDataset
    ?(name = "ShuffleDataset")
    ?reshuffle_each_iteration
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (buffer_size : [ `int64 ] t)
    (seed : [ `int64 ] t)
    (seed2 : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    match reshuffle_each_iteration with | None -> attributes | Some reshuffle_each_iteration -> ("reshuffle_each_iteration", Bool reshuffle_each_iteration) :: attributes
  in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.shuffleDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P buffer_size)); (`single (P seed)); (`single (P seed2)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let shutdownDistributedTPU
    ?(name = "ShutdownDistributedTPU")
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.shutdownDistributedTPU in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sigmoid
    ?(name = "Sigmoid")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sigmoid in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sigmoidGrad
    ?(name = "SigmoidGrad")
    ?(control_inputs = [])
    (y : ([< `float | `double | `complex64 ] as 't) t)
    (dy : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type y)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sigmoidGrad in
  let inputs = [ (`single (P y)); (`single (P dy)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sign
    ?(name = "Sign")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sign in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sin
    ?(name = "Sin")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sin in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sinh
    ?(name = "Sinh")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sinh in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let size
    ?(name = "Size")
    ~type_
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.size in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let skipDataset
    ?(name = "SkipDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (count : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.skipDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P count)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let skipgram
    ?(name = "Skipgram")
    ~filename
    ~batch_size
    ?window_size
    ?min_count
    ?subsample
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    ("filename", String filename) :: attributes
  in
  let attributes =
    ("batch_size", Int batch_size) :: attributes
  in
  let attributes =
    match window_size with | None -> attributes | Some window_size -> ("window_size", Int window_size) :: attributes
  in
  let attributes =
    match min_count with | None -> attributes | Some min_count -> ("min_count", Int min_count) :: attributes
  in
  let attributes =
    match subsample with | None -> attributes | Some subsample -> ("subsample", Float subsample) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.skipgram in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 4)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 5)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 6)

let slice
    ?(name = "Slice")
    ?(control_inputs = [])
    (input : 't t)
    (begin__ : ([< `int32 | `int64 ] as 'index) t)
    (size : ([< `int32 | `int64 ] as 'index) t)
  =
  let attributes = [ "Index", Type (P (Node.output_type begin__)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.slice in
  let inputs = [ (`single (P input)); (`single (P begin__)); (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let snapshot
    ?(name = "Snapshot")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.snapshot in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let softmax
    ?(name = "Softmax")
    ?(control_inputs = [])
    (logits : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type logits)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.softmax in
  let inputs = [ (`single (P logits)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type logits)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let softmaxCrossEntropyWithLogits
    ?(name = "SoftmaxCrossEntropyWithLogits")
    ?(control_inputs = [])
    (features : ([< `float | `double ] as 't) t)
    (labels : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type features)) ;  "T", Type (P (Node.output_type features)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.softmaxCrossEntropyWithLogits in
  let inputs = [ (`single (P features)); (`single (P labels)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let softplus
    ?(name = "Softplus")
    ?(control_inputs = [])
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type features)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.softplus in
  let inputs = [ (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let softplusGrad
    ?(name = "SoftplusGrad")
    ?(control_inputs = [])
    (gradients : ([< `float | `double ] as 't) t)
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type gradients)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.softplusGrad in
  let inputs = [ (`single (P gradients)); (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type gradients)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let softsign
    ?(name = "Softsign")
    ?(control_inputs = [])
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type features)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.softsign in
  let inputs = [ (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let softsignGrad
    ?(name = "SoftsignGrad")
    ?(control_inputs = [])
    (gradients : ([< `float | `double ] as 't) t)
    (features : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type gradients)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.softsignGrad in
  let inputs = [ (`single (P gradients)); (`single (P features)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type gradients)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let spaceToBatch
    ?(name = "SpaceToBatch")
    ~block_size
    ?(control_inputs = [])
    (input : 't t)
    (paddings : ([< `int32 | `int64 ] as 'tpaddings) t)
  =
  let attributes = [ "Tpaddings", Type (P (Node.output_type paddings)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("block_size", Int block_size) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.spaceToBatch in
  let inputs = [ (`single (P input)); (`single (P paddings)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let spaceToBatchND
    ?(name = "SpaceToBatchND")
    ?(control_inputs = [])
    (input : 't t)
    (block_shape : ([< `int32 | `int64 ] as 'tblock_shape) t)
    (paddings : ([< `int32 | `int64 ] as 'tpaddings) t)
  =
  let attributes = [ "Tpaddings", Type (P (Node.output_type paddings)) ;  "Tblock_shape", Type (P (Node.output_type block_shape)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.spaceToBatchND in
  let inputs = [ (`single (P input)); (`single (P block_shape)); (`single (P paddings)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let spaceToDepth
    ?(name = "SpaceToDepth")
    ~block_size
    ?data_format
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("block_size", Int block_size) :: attributes
  in
  let attributes =
    match data_format with | None -> attributes | Some data_format -> ("data_format", String data_format) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.spaceToDepth in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseAccumulatorApplyGradient
    ?(name = "SparseAccumulatorApplyGradient")
    ~has_known_shape
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (local_step : [ `int64 ] t)
    (gradient_indices : [ `int64 ] t)
    (gradient_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 'dtype) t)
    (gradient_shape : [ `int64 ] t)
  =
  let attributes = [ "dtype", Type (P (Node.output_type gradient_values)) ] in
  let attributes =
    ("has_known_shape", Bool has_known_shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseAccumulatorApplyGradient in
  let inputs = [ (`single (P handle)); (`single (P local_step)); (`single (P gradient_indices)); (`single (P gradient_values)); (`single (P gradient_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseAccumulatorTakeGradient
    ?(name = "SparseAccumulatorTakeGradient")
    ~type_1
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (num_required : [ `int32 ] t)
  =
  let attributes = [ "dtype", Type (P type_1) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseAccumulatorTakeGradient in
  let inputs = [ (`single (P handle)); (`single (P num_required)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let sparseAdd
    ?(name = "SparseAdd")
    ?(control_inputs = [])
    (a_indices : [ `int64 ] t)
    (a_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (a_shape : [ `int64 ] t)
    (b_indices : [ `int64 ] t)
    (b_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (b_shape : [ `int64 ] t)
    (thresh : ([< `float | `double | `int32 | `int64 ] as 'treal) t)
  =
  let attributes = [ "Treal", Type (P (Node.output_type thresh)) ;  "T", Type (P (Node.output_type a_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseAdd in
  let inputs = [ (`single (P a_indices)); (`single (P a_values)); (`single (P a_shape)); (`single (P b_indices)); (`single (P b_values)); (`single (P b_shape)); (`single (P thresh)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let sparseAddGrad
    ?(name = "SparseAddGrad")
    ?(control_inputs = [])
    (backprop_val_grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (a_indices : [ `int64 ] t)
    (b_indices : [ `int64 ] t)
    (sum_indices : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type backprop_val_grad)) ;  "T", Type (P (Node.output_type backprop_val_grad)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseAddGrad in
  let inputs = [ (`single (P backprop_val_grad)); (`single (P a_indices)); (`single (P b_indices)); (`single (P sum_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type backprop_val_grad)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type backprop_val_grad)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let sparseApplyAdadelta
    ?(name = "SparseApplyAdadelta")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum_update : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (rho : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyAdadelta in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P accum_update)); (`single (P lr)); (`single (P rho)); (`single (P epsilon)); (`single (P grad)); (`single (P indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseApplyAdagrad
    ?(name = "SparseApplyAdagrad")
    ?use_locking
    ?update_slots
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let attributes =
    match update_slots with | None -> attributes | Some update_slots -> ("update_slots", Bool update_slots) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyAdagrad in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P lr)); (`single (P grad)); (`single (P indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseApplyAdagradDA
    ?(name = "SparseApplyAdagradDA")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (gradient_accumulator : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (gradient_squared_accumulator : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (global_step : [ `int64 ] t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyAdagradDA in
  let inputs = [ (`single (P var)); (`single (P gradient_accumulator)); (`single (P gradient_squared_accumulator)); (`single (P grad)); (`single (P indices)); (`single (P lr)); (`single (P l1)); (`single (P l2)); (`single (P global_step)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseApplyCenteredRMSProp
    ?(name = "SparseApplyCenteredRMSProp")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (mg : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (ms : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (mom : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (rho : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (momentum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyCenteredRMSProp in
  let inputs = [ (`single (P var)); (`single (P mg)); (`single (P ms)); (`single (P mom)); (`single (P lr)); (`single (P rho)); (`single (P momentum)); (`single (P epsilon)); (`single (P grad)); (`single (P indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseApplyFtrl
    ?(name = "SparseApplyFtrl")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (linear : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr_power : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyFtrl in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P linear)); (`single (P grad)); (`single (P indices)); (`single (P lr)); (`single (P l1)); (`single (P l2)); (`single (P lr_power)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseApplyFtrlV2
    ?(name = "SparseApplyFtrlV2")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (linear : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2_shrinkage : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr_power : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyFtrlV2 in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P linear)); (`single (P grad)); (`single (P indices)); (`single (P lr)); (`single (P l1)); (`single (P l2)); (`single (P l2_shrinkage)); (`single (P lr_power)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseApplyMomentum
    ?(name = "SparseApplyMomentum")
    ?use_locking
    ?use_nesterov
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (momentum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let attributes =
    match use_nesterov with | None -> attributes | Some use_nesterov -> ("use_nesterov", Bool use_nesterov) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyMomentum in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P lr)); (`single (P grad)); (`single (P indices)); (`single (P momentum)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseApplyProximalAdagrad
    ?(name = "SparseApplyProximalAdagrad")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (accum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyProximalAdagrad in
  let inputs = [ (`single (P var)); (`single (P accum)); (`single (P lr)); (`single (P l1)); (`single (P l2)); (`single (P grad)); (`single (P indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseApplyProximalGradientDescent
    ?(name = "SparseApplyProximalGradientDescent")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (alpha : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l1 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (l2 : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyProximalGradientDescent in
  let inputs = [ (`single (P var)); (`single (P alpha)); (`single (P l1)); (`single (P l2)); (`single (P grad)); (`single (P indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseApplyRMSProp
    ?(name = "SparseApplyRMSProp")
    ?use_locking
    ?(control_inputs = [])
    (var : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (ms : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (mom : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (lr : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (rho : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (momentum : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (epsilon : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type var)) ] in
  let attributes =
    match use_locking with | None -> attributes | Some use_locking -> ("use_locking", Bool use_locking) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseApplyRMSProp in
  let inputs = [ (`single (P var)); (`single (P ms)); (`single (P mom)); (`single (P lr)); (`single (P rho)); (`single (P momentum)); (`single (P epsilon)); (`single (P grad)); (`single (P indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type var)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseConcat
    ?(name = "SparseConcat")
    ~concat_dim
    ?(control_inputs = [])
    (indices : [ `int64 ] t list)
    (values : 't t list)
    (shapes : [ `int64 ] t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn values))) ] in
  let attributes =
    ("concat_dim", Int concat_dim) :: attributes
  in
  let attributes =
    ("N", Int (List.length indices)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseConcat in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) indices)); (`multi (List.map ~f:(fun n -> P n) values)); (`multi (List.map ~f:(fun n -> P n) shapes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn values))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let sparseConditionalAccumulator
    ?(name = "SparseConditionalAccumulator")
    ~shape
    ?container
    ?shared_name
    ?reduction_type
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let attributes =
    match reduction_type with | None -> attributes | Some reduction_type -> ("reduction_type", String reduction_type) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseConditionalAccumulator in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseDenseCwiseAdd
    ?(name = "SparseDenseCwiseAdd")
    ?(control_inputs = [])
    (sp_indices : [ `int64 ] t)
    (sp_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (sp_shape : [ `int64 ] t)
    (dense : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type sp_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseDenseCwiseAdd in
  let inputs = [ (`single (P sp_indices)); (`single (P sp_values)); (`single (P sp_shape)); (`single (P dense)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type sp_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseDenseCwiseDiv
    ?(name = "SparseDenseCwiseDiv")
    ?(control_inputs = [])
    (sp_indices : [ `int64 ] t)
    (sp_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (sp_shape : [ `int64 ] t)
    (dense : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type sp_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseDenseCwiseDiv in
  let inputs = [ (`single (P sp_indices)); (`single (P sp_values)); (`single (P sp_shape)); (`single (P dense)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type sp_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseDenseCwiseMul
    ?(name = "SparseDenseCwiseMul")
    ?(control_inputs = [])
    (sp_indices : [ `int64 ] t)
    (sp_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (sp_shape : [ `int64 ] t)
    (dense : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type sp_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseDenseCwiseMul in
  let inputs = [ (`single (P sp_indices)); (`single (P sp_values)); (`single (P sp_shape)); (`single (P dense)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type sp_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseFillEmptyRows
    ?(name = "SparseFillEmptyRows")
    ?(control_inputs = [])
    (indices : [ `int64 ] t)
    (values : 't t)
    (dense_shape : [ `int64 ] t)
    (default_value : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseFillEmptyRows in
  let inputs = [ (`single (P indices)); (`single (P values)); (`single (P dense_shape)); (`single (P default_value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 3)

let sparseFillEmptyRowsGrad
    ?(name = "SparseFillEmptyRowsGrad")
    ?(control_inputs = [])
    (reverse_index_map : [ `int64 ] t)
    (grad_values : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type grad_values)) ;  "T", Type (P (Node.output_type grad_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseFillEmptyRowsGrad in
  let inputs = [ (`single (P reverse_index_map)); (`single (P grad_values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type grad_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type grad_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let sparseMatMul
    ?(name = "SparseMatMul")
    ?transpose_a
    ?transpose_b
    ?a_is_sparse
    ?b_is_sparse
    ?(control_inputs = [])
    (a : ([< `float ] as 'ta) t)
    (b : ([< `float ] as 'tb) t)
  =
  let attributes = [ "Tb", Type (P (Node.output_type b)) ;  "Ta", Type (P (Node.output_type a)) ] in
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
  let name = Name.of_string name in
  let op_name = Op_names.sparseMatMul in
  let inputs = [ (`single (P a)); (`single (P b)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseReduceMax
    ?(name = "SparseReduceMax")
    ?keep_dims
    ?(control_inputs = [])
    (input_indices : [ `int64 ] t)
    (input_values : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (input_shape : [ `int64 ] t)
    (reduction_axes : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input_values)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseReduceMax in
  let inputs = [ (`single (P input_indices)); (`single (P input_values)); (`single (P input_shape)); (`single (P reduction_axes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseReduceMaxSparse
    ?(name = "SparseReduceMaxSparse")
    ?keep_dims
    ?(control_inputs = [])
    (input_indices : [ `int64 ] t)
    (input_values : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (input_shape : [ `int64 ] t)
    (reduction_axes : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input_values)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseReduceMaxSparse in
  let inputs = [ (`single (P input_indices)); (`single (P input_values)); (`single (P input_shape)); (`single (P reduction_axes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let sparseReduceSum
    ?(name = "SparseReduceSum")
    ?keep_dims
    ?(control_inputs = [])
    (input_indices : [ `int64 ] t)
    (input_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (input_shape : [ `int64 ] t)
    (reduction_axes : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input_values)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseReduceSum in
  let inputs = [ (`single (P input_indices)); (`single (P input_values)); (`single (P input_shape)); (`single (P reduction_axes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseReduceSumSparse
    ?(name = "SparseReduceSumSparse")
    ?keep_dims
    ?(control_inputs = [])
    (input_indices : [ `int64 ] t)
    (input_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (input_shape : [ `int64 ] t)
    (reduction_axes : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input_values)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseReduceSumSparse in
  let inputs = [ (`single (P input_indices)); (`single (P input_values)); (`single (P input_shape)); (`single (P reduction_axes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let sparseReorder
    ?(name = "SparseReorder")
    ?(control_inputs = [])
    (input_indices : [ `int64 ] t)
    (input_values : 't t)
    (input_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseReorder in
  let inputs = [ (`single (P input_indices)); (`single (P input_values)); (`single (P input_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let sparseReshape
    ?(name = "SparseReshape")
    ?(control_inputs = [])
    (input_indices : [ `int64 ] t)
    (input_shape : [ `int64 ] t)
    (new_shape : [ `int64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseReshape in
  let inputs = [ (`single (P input_indices)); (`single (P input_shape)); (`single (P new_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let sparseSegmentMean
    ?(name = "SparseSegmentMean")
    ?(control_inputs = [])
    (data : ([< `float | `double ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tidx) t)
    (segment_ids : [ `int32 ] t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSegmentMean in
  let inputs = [ (`single (P data)); (`single (P indices)); (`single (P segment_ids)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSegmentMeanGrad
    ?(name = "SparseSegmentMeanGrad")
    ?(control_inputs = [])
    (grad : ([< `float | `double ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tidx) t)
    (segment_ids : [ `int32 ] t)
    (output_dim0 : [ `int32 ] t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type grad)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSegmentMeanGrad in
  let inputs = [ (`single (P grad)); (`single (P indices)); (`single (P segment_ids)); (`single (P output_dim0)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type grad)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSegmentMeanWithNumSegments
    ?(name = "SparseSegmentMeanWithNumSegments")
    ?(control_inputs = [])
    (data : ([< `float | `double ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tidx) t)
    (segment_ids : [ `int32 ] t)
    (num_segments : ([< `int32 | `int64 ] as 'tnumsegments) t)
  =
  let attributes = [ "Tnumsegments", Type (P (Node.output_type num_segments)) ;  "Tidx", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSegmentMeanWithNumSegments in
  let inputs = [ (`single (P data)); (`single (P indices)); (`single (P segment_ids)); (`single (P num_segments)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSegmentSqrtN
    ?(name = "SparseSegmentSqrtN")
    ?(control_inputs = [])
    (data : ([< `float | `double ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tidx) t)
    (segment_ids : [ `int32 ] t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSegmentSqrtN in
  let inputs = [ (`single (P data)); (`single (P indices)); (`single (P segment_ids)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSegmentSqrtNGrad
    ?(name = "SparseSegmentSqrtNGrad")
    ?(control_inputs = [])
    (grad : ([< `float | `double ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tidx) t)
    (segment_ids : [ `int32 ] t)
    (output_dim0 : [ `int32 ] t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type grad)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSegmentSqrtNGrad in
  let inputs = [ (`single (P grad)); (`single (P indices)); (`single (P segment_ids)); (`single (P output_dim0)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type grad)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSegmentSqrtNWithNumSegments
    ?(name = "SparseSegmentSqrtNWithNumSegments")
    ?(control_inputs = [])
    (data : ([< `float | `double ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tidx) t)
    (segment_ids : [ `int32 ] t)
    (num_segments : ([< `int32 | `int64 ] as 'tnumsegments) t)
  =
  let attributes = [ "Tnumsegments", Type (P (Node.output_type num_segments)) ;  "Tidx", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSegmentSqrtNWithNumSegments in
  let inputs = [ (`single (P data)); (`single (P indices)); (`single (P segment_ids)); (`single (P num_segments)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSegmentSum
    ?(name = "SparseSegmentSum")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tidx) t)
    (segment_ids : [ `int32 ] t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSegmentSum in
  let inputs = [ (`single (P data)); (`single (P indices)); (`single (P segment_ids)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSegmentSumWithNumSegments
    ?(name = "SparseSegmentSumWithNumSegments")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (indices : ([< `int32 | `int64 ] as 'tidx) t)
    (segment_ids : [ `int32 ] t)
    (num_segments : ([< `int32 | `int64 ] as 'tnumsegments) t)
  =
  let attributes = [ "Tnumsegments", Type (P (Node.output_type num_segments)) ;  "Tidx", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSegmentSumWithNumSegments in
  let inputs = [ (`single (P data)); (`single (P indices)); (`single (P segment_ids)); (`single (P num_segments)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSlice
    ?(name = "SparseSlice")
    ?(control_inputs = [])
    (indices : [ `int64 ] t)
    (values : 't t)
    (shape : [ `int64 ] t)
    (start : [ `int64 ] t)
    (size : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSlice in
  let inputs = [ (`single (P indices)); (`single (P values)); (`single (P shape)); (`single (P start)); (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let sparseSliceGrad
    ?(name = "SparseSliceGrad")
    ?(control_inputs = [])
    (backprop_val_grad : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (input_indices : [ `int64 ] t)
    (input_start : [ `int64 ] t)
    (output_indices : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type backprop_val_grad)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSliceGrad in
  let inputs = [ (`single (P backprop_val_grad)); (`single (P input_indices)); (`single (P input_start)); (`single (P output_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type backprop_val_grad)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSoftmax
    ?(name = "SparseSoftmax")
    ?(control_inputs = [])
    (sp_indices : [ `int64 ] t)
    (sp_values : ([< `float | `double ] as 't) t)
    (sp_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type sp_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSoftmax in
  let inputs = [ (`single (P sp_indices)); (`single (P sp_values)); (`single (P sp_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type sp_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseSoftmaxCrossEntropyWithLogits
    ?(name = "SparseSoftmaxCrossEntropyWithLogits")
    ?(control_inputs = [])
    (features : ([< `float | `double ] as 't) t)
    (labels : ([< `int32 | `int64 ] as 'tlabels) t)
  =
  let attributes = [ "Tlabels", Type (P (Node.output_type labels)) ;  "T", Type (P (Node.output_type features)) ;  "T", Type (P (Node.output_type features)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSoftmaxCrossEntropyWithLogits in
  let inputs = [ (`single (P features)); (`single (P labels)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type features)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let sparseSparseMaximum
    ?(name = "SparseSparseMaximum")
    ?(control_inputs = [])
    (a_indices : [ `int64 ] t)
    (a_values : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (a_shape : [ `int64 ] t)
    (b_indices : [ `int64 ] t)
    (b_values : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (b_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type a_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSparseMaximum in
  let inputs = [ (`single (P a_indices)); (`single (P a_values)); (`single (P a_shape)); (`single (P b_indices)); (`single (P b_values)); (`single (P b_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let sparseSparseMinimum
    ?(name = "SparseSparseMinimum")
    ?(control_inputs = [])
    (a_indices : [ `int64 ] t)
    (a_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (a_shape : [ `int64 ] t)
    (b_indices : [ `int64 ] t)
    (b_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (b_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type a_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseSparseMinimum in
  let inputs = [ (`single (P a_indices)); (`single (P a_values)); (`single (P a_shape)); (`single (P b_indices)); (`single (P b_values)); (`single (P b_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let sparseTensorDenseAdd
    ?(name = "SparseTensorDenseAdd")
    ?(control_inputs = [])
    (a_indices : ([< `int32 | `int64 ] as 'tindices) t)
    (a_values : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (a_shape : ([< `int32 | `int64 ] as 'tindices) t)
    (b : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type a_indices)) ;  "T", Type (P (Node.output_type a_values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseTensorDenseAdd in
  let inputs = [ (`single (P a_indices)); (`single (P a_values)); (`single (P a_shape)); (`single (P b)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseTensorDenseMatMul
    ?(name = "SparseTensorDenseMatMul")
    ?adjoint_a
    ?adjoint_b
    ?(control_inputs = [])
    (a_indices : ([< `int32 | `int64 ] as 'tindices) t)
    (a_values : 't t)
    (a_shape : [ `int64 ] t)
    (b : 't t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type a_indices)) ;  "T", Type (P (Node.output_type a_values)) ] in
  let attributes =
    match adjoint_a with | None -> attributes | Some adjoint_a -> ("adjoint_a", Bool adjoint_a) :: attributes
  in
  let attributes =
    match adjoint_b with | None -> attributes | Some adjoint_b -> ("adjoint_b", Bool adjoint_b) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseTensorDenseMatMul in
  let inputs = [ (`single (P a_indices)); (`single (P a_values)); (`single (P a_shape)); (`single (P b)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type a_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseTensorSliceDataset
    ?(name = "SparseTensorSliceDataset")
    ?(control_inputs = [])
    (indices : [ `int64 ] t)
    (values : 'tvalues t)
    (dense_shape : [ `int64 ] t)
  =
  let attributes = [ "Tvalues", Type (P (Node.output_type values)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sparseTensorSliceDataset in
  let inputs = [ (`single (P indices)); (`single (P values)); (`single (P dense_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseToDense
    ?(name = "SparseToDense")
    ?validate_indices
    ?(control_inputs = [])
    (sparse_indices : ([< `int32 | `int64 ] as 'tindices) t)
    (output_shape : ([< `int32 | `int64 ] as 'tindices) t)
    (sparse_values : 't t)
    (default_value : 't t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type sparse_indices)) ;  "T", Type (P (Node.output_type sparse_values)) ] in
  let attributes =
    match validate_indices with | None -> attributes | Some validate_indices -> ("validate_indices", Bool validate_indices) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseToDense in
  let inputs = [ (`single (P sparse_indices)); (`single (P output_shape)); (`single (P sparse_values)); (`single (P default_value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type sparse_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sparseToSparseSetOperation
    ?(name = "SparseToSparseSetOperation")
    ~set_operation
    ?validate_indices
    ?(control_inputs = [])
    (set1_indices : [ `int64 ] t)
    (set1_values : ([< `int32 | `int64 | `string ] as 't) t)
    (set1_shape : [ `int64 ] t)
    (set2_indices : [ `int64 ] t)
    (set2_values : ([< `int32 | `int64 | `string ] as 't) t)
    (set2_shape : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type set1_values)) ] in
  let attributes =
    ("set_operation", String set_operation) :: attributes
  in
  let attributes =
    match validate_indices with | None -> attributes | Some validate_indices -> ("validate_indices", Bool validate_indices) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sparseToSparseSetOperation in
  let inputs = [ (`single (P set1_indices)); (`single (P set1_values)); (`single (P set1_shape)); (`single (P set2_indices)); (`single (P set2_values)); (`single (P set2_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type set1_values)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let split
    ?(name = "Split")
    ~num_split
    ?(control_inputs = [])
    (split_dim : [ `int32 ] t)
    (value : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let attributes =
    ("num_split", Int num_split) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.split in
  let inputs = [ (`single (P split_dim)); (`single (P value)) ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:(Node.output_type value)
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init num_split ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let splitV
    ?(name = "SplitV")
    ~num_split
    ?(control_inputs = [])
    (value : 't t)
    (size_splits : ([< `int32 | `int64 ] as 'tlen) t)
    (split_dim : [ `int32 ] t)
  =
  let attributes = [ "Tlen", Type (P (Node.output_type size_splits)) ;  "T", Type (P (Node.output_type value)) ] in
  let attributes =
    ("num_split", Int num_split) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.splitV in
  let inputs = [ (`single (P value)); (`single (P size_splits)); (`single (P split_dim)) ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:(Node.output_type value)
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init num_split ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let sqrt
    ?(name = "Sqrt")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sqrt in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sqrtGrad
    ?(name = "SqrtGrad")
    ?(control_inputs = [])
    (y : ([< `float | `double | `complex64 ] as 't) t)
    (dy : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type y)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sqrtGrad in
  let inputs = [ (`single (P y)); (`single (P dy)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let square
    ?(name = "Square")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.square in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let squaredDifference
    ?(name = "SquaredDifference")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.squaredDifference in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let squeeze
    ?(name = "Squeeze")
    ?squeeze_dims
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match squeeze_dims with | None -> attributes | Some squeeze_dims -> ("squeeze_dims", List (Int squeeze_dims)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.squeeze in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stack
    ?(name = "Stack")
    ?stack_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match stack_name with | None -> attributes | Some stack_name -> ("stack_name", String stack_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stack in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stackClose
    ?(name = "StackClose")
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.stackClose in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stackPop
    ?(name = "StackPop")
    ~type_
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [ "elem_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.stackPop in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stackPush
    ?(name = "StackPush")
    ?swap_memory
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (elem : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type elem)) ] in
  let attributes =
    match swap_memory with | None -> attributes | Some swap_memory -> ("swap_memory", Bool swap_memory) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stackPush in
  let inputs = [ (`single (P handle)); (`single (P elem)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type elem)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stageClear
    ?(name = "StageClear")
    ?capacity
    ?memory_limit
    ~dtypes
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match memory_limit with | None -> attributes | Some memory_limit -> ("memory_limit", Int memory_limit) :: attributes
  in
  let attributes =
    ("dtypes", List (Type dtypes)) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stageClear in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stageSize
    ?(name = "StageSize")
    ?capacity
    ?memory_limit
    ~dtypes
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match capacity with | None -> attributes | Some capacity -> ("capacity", Int capacity) :: attributes
  in
  let attributes =
    match memory_limit with | None -> attributes | Some memory_limit -> ("memory_limit", Int memory_limit) :: attributes
  in
  let attributes =
    ("dtypes", List (Type dtypes)) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stageSize in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let statelessMultinomial
    ?(name = "StatelessMultinomial")
    ~type_
    ?(control_inputs = [])
    (logits : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (num_samples : [ `int32 ] t)
    (seed : ([< `int32 | `int64 ] as 'tseed) t)
  =
  let attributes = [ "Tseed", Type (P (Node.output_type seed)) ;  "T", Type (P (Node.output_type logits)) ;  "output_dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.statelessMultinomial in
  let inputs = [ (`single (P logits)); (`single (P num_samples)); (`single (P seed)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let statelessRandomNormal
    ?(name = "StatelessRandomNormal")
    ~type_
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 't) t)
    (seed : ([< `int32 | `int64 ] as 'tseed) t)
  =
  let attributes = [ "Tseed", Type (P (Node.output_type seed)) ;  "T", Type (P (Node.output_type shape)) ;  "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.statelessRandomNormal in
  let inputs = [ (`single (P shape)); (`single (P seed)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let statelessRandomUniform
    ?(name = "StatelessRandomUniform")
    ~type_
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 't) t)
    (seed : ([< `int32 | `int64 ] as 'tseed) t)
  =
  let attributes = [ "Tseed", Type (P (Node.output_type seed)) ;  "T", Type (P (Node.output_type shape)) ;  "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.statelessRandomUniform in
  let inputs = [ (`single (P shape)); (`single (P seed)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let statelessRandomUniformInt
    ?(name = "StatelessRandomUniformInt")
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 't) t)
    (seed : ([< `int32 | `int64 ] as 'tseed) t)
    (minval : ([< `int32 | `int64 ] as 'dtype) t)
    (maxval : ([< `int32 | `int64 ] as 'dtype) t)
  =
  let attributes = [ "Tseed", Type (P (Node.output_type seed)) ;  "T", Type (P (Node.output_type shape)) ;  "dtype", Type (P (Node.output_type minval)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.statelessRandomUniformInt in
  let inputs = [ (`single (P shape)); (`single (P seed)); (`single (P minval)); (`single (P maxval)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type minval)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let statelessTruncatedNormal
    ?(name = "StatelessTruncatedNormal")
    ~type_
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 't) t)
    (seed : ([< `int32 | `int64 ] as 'tseed) t)
  =
  let attributes = [ "Tseed", Type (P (Node.output_type seed)) ;  "T", Type (P (Node.output_type shape)) ;  "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.statelessTruncatedNormal in
  let inputs = [ (`single (P shape)); (`single (P seed)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let staticRegexFullMatch
    ?(name = "StaticRegexFullMatch")
    ~pattern
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("pattern", String pattern) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.staticRegexFullMatch in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Bool
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let staticRegexReplace
    ?(name = "StaticRegexReplace")
    ~pattern
    ~rewrite
    ?replace_global
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("pattern", String pattern) :: attributes
  in
  let attributes =
    ("rewrite", String rewrite) :: attributes
  in
  let attributes =
    match replace_global with | None -> attributes | Some replace_global -> ("replace_global", Bool replace_global) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.staticRegexReplace in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stopGradient
    ?(name = "StopGradient")
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.stopGradient in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stridedSlice
    ?(name = "StridedSlice")
    ?begin_mask
    ?end_mask
    ?ellipsis_mask
    ?new_axis_mask
    ?shrink_axis_mask
    ?(control_inputs = [])
    (input : 't t)
    (begin__ : ([< `int32 | `int64 ] as 'index) t)
    (end__ : ([< `int32 | `int64 ] as 'index) t)
    (strides : ([< `int32 | `int64 ] as 'index) t)
  =
  let attributes = [ "Index", Type (P (Node.output_type begin__)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match begin_mask with | None -> attributes | Some begin_mask -> ("begin_mask", Int begin_mask) :: attributes
  in
  let attributes =
    match end_mask with | None -> attributes | Some end_mask -> ("end_mask", Int end_mask) :: attributes
  in
  let attributes =
    match ellipsis_mask with | None -> attributes | Some ellipsis_mask -> ("ellipsis_mask", Int ellipsis_mask) :: attributes
  in
  let attributes =
    match new_axis_mask with | None -> attributes | Some new_axis_mask -> ("new_axis_mask", Int new_axis_mask) :: attributes
  in
  let attributes =
    match shrink_axis_mask with | None -> attributes | Some shrink_axis_mask -> ("shrink_axis_mask", Int shrink_axis_mask) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stridedSlice in
  let inputs = [ (`single (P input)); (`single (P begin__)); (`single (P end__)); (`single (P strides)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stridedSliceAssign
    ?(name = "StridedSliceAssign")
    ?begin_mask
    ?end_mask
    ?ellipsis_mask
    ?new_axis_mask
    ?shrink_axis_mask
    ?(control_inputs = [])
    (ref : 't t)
    (begin__ : ([< `int32 | `int64 ] as 'index) t)
    (end__ : ([< `int32 | `int64 ] as 'index) t)
    (strides : ([< `int32 | `int64 ] as 'index) t)
    (value : 't t)
  =
  let attributes = [ "Index", Type (P (Node.output_type begin__)) ;  "T", Type (P (Node.output_type ref)) ] in
  let attributes =
    match begin_mask with | None -> attributes | Some begin_mask -> ("begin_mask", Int begin_mask) :: attributes
  in
  let attributes =
    match end_mask with | None -> attributes | Some end_mask -> ("end_mask", Int end_mask) :: attributes
  in
  let attributes =
    match ellipsis_mask with | None -> attributes | Some ellipsis_mask -> ("ellipsis_mask", Int ellipsis_mask) :: attributes
  in
  let attributes =
    match new_axis_mask with | None -> attributes | Some new_axis_mask -> ("new_axis_mask", Int new_axis_mask) :: attributes
  in
  let attributes =
    match shrink_axis_mask with | None -> attributes | Some shrink_axis_mask -> ("shrink_axis_mask", Int shrink_axis_mask) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stridedSliceAssign in
  let inputs = [ (`single (P ref)); (`single (P begin__)); (`single (P end__)); (`single (P strides)); (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type ref)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stridedSliceGrad
    ?(name = "StridedSliceGrad")
    ?begin_mask
    ?end_mask
    ?ellipsis_mask
    ?new_axis_mask
    ?shrink_axis_mask
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 'index) t)
    (begin__ : ([< `int32 | `int64 ] as 'index) t)
    (end__ : ([< `int32 | `int64 ] as 'index) t)
    (strides : ([< `int32 | `int64 ] as 'index) t)
    (dy : 't t)
  =
  let attributes = [ "Index", Type (P (Node.output_type shape)) ;  "T", Type (P (Node.output_type dy)) ] in
  let attributes =
    match begin_mask with | None -> attributes | Some begin_mask -> ("begin_mask", Int begin_mask) :: attributes
  in
  let attributes =
    match end_mask with | None -> attributes | Some end_mask -> ("end_mask", Int end_mask) :: attributes
  in
  let attributes =
    match ellipsis_mask with | None -> attributes | Some ellipsis_mask -> ("ellipsis_mask", Int ellipsis_mask) :: attributes
  in
  let attributes =
    match new_axis_mask with | None -> attributes | Some new_axis_mask -> ("new_axis_mask", Int new_axis_mask) :: attributes
  in
  let attributes =
    match shrink_axis_mask with | None -> attributes | Some shrink_axis_mask -> ("shrink_axis_mask", Int shrink_axis_mask) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stridedSliceGrad in
  let inputs = [ (`single (P shape)); (`single (P begin__)); (`single (P end__)); (`single (P strides)); (`single (P dy)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type dy)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stringJoin
    ?(name = "StringJoin")
    ?separator
    ?(control_inputs = [])
    (inputs__ : [ `string ] t list)
  =
  let attributes = [] in
  let attributes =
    ("N", Int (List.length inputs__)) :: attributes
  in
  let attributes =
    match separator with | None -> attributes | Some separator -> ("separator", String separator) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stringJoin in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stringLength
    ?(name = "StringLength")
    ?unit
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match unit with | None -> attributes | Some unit -> ("unit", String unit) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stringLength in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stringSplit
    ?(name = "StringSplit")
    ?skip_empty
    ?(control_inputs = [])
    (input : [ `string ] t)
    (delimiter : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match skip_empty with | None -> attributes | Some skip_empty -> ("skip_empty", Bool skip_empty) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stringSplit in
  let inputs = [ (`single (P input)); (`single (P delimiter)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let stringSplitV2
    ?(name = "StringSplitV2")
    ?maxsplit
    ?(control_inputs = [])
    (input : [ `string ] t)
    (sep : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match maxsplit with | None -> attributes | Some maxsplit -> ("maxsplit", Int maxsplit) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stringSplitV2 in
  let inputs = [ (`single (P input)); (`single (P sep)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let stringStrip
    ?(name = "StringStrip")
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.stringStrip in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stringToHashBucket
    ?(name = "StringToHashBucket")
    ~num_buckets
    ?(control_inputs = [])
    (string_tensor : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_buckets", Int num_buckets) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stringToHashBucket in
  let inputs = [ (`single (P string_tensor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stringToHashBucketFast
    ?(name = "StringToHashBucketFast")
    ~num_buckets
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_buckets", Int num_buckets) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stringToHashBucketFast in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stringToHashBucketStrong
    ?(name = "StringToHashBucketStrong")
    ~num_buckets
    ~key
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_buckets", Int num_buckets) :: attributes
  in
  let attributes =
    ("key", List (Int key)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.stringToHashBucketStrong in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let stringToNumber
    ?(name = "StringToNumber")
    ~type_
    ?(control_inputs = [])
    (string_tensor : [ `string ] t)
  =
  let attributes = [ "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.stringToNumber in
  let inputs = [ (`single (P string_tensor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sub
    ?(name = "Sub")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.sub in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let substr
    ?(name = "Substr")
    ?unit
    ?(control_inputs = [])
    (input : [ `string ] t)
    (pos : ([< `int32 | `int64 ] as 't) t)
    (len : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type pos)) ] in
  let attributes =
    match unit with | None -> attributes | Some unit -> ("unit", String unit) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.substr in
  let inputs = [ (`single (P input)); (`single (P pos)); (`single (P len)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let sum
    ?(name = "Sum")
    ?keep_dims
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (reduction_indices : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type reduction_indices)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match keep_dims with | None -> attributes | Some keep_dims -> ("keep_dims", Bool keep_dims) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.sum in
  let inputs = [ (`single (P input)); (`single (P reduction_indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let svd
    ?(name = "Svd")
    ?compute_uv
    ?full_matrices
    ?(control_inputs = [])
    (input : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match compute_uv with | None -> attributes | Some compute_uv -> ("compute_uv", Bool compute_uv) :: attributes
  in
  let attributes =
    match full_matrices with | None -> attributes | Some full_matrices -> ("full_matrices", Bool full_matrices) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.svd in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let switch
    ?(name = "Switch")
    ?(control_inputs = [])
    (data : 't t)
    (pred : [ `bool ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type data)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.switch in
  let inputs = [ (`single (P data)); (`single (P pred)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let tFRecordDataset
    ?(name = "TFRecordDataset")
    ?(control_inputs = [])
    (filenames : [ `string ] t)
    (compression_type : [ `string ] t)
    (buffer_size : [ `int64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tFRecordDataset in
  let inputs = [ (`single (P filenames)); (`single (P compression_type)); (`single (P buffer_size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tFRecordReader
    ?(name = "TFRecordReader")
    ?container
    ?shared_name
    ?compression_type
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let attributes =
    match compression_type with | None -> attributes | Some compression_type -> ("compression_type", String compression_type) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tFRecordReader in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tPUCompilationResult
    ?(name = "TPUCompilationResult")
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tPUCompilationResult in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tPUEmbeddingActivations
    ?(name = "TPUEmbeddingActivations")
    ~table_id
    ~lookup_id
    ?(control_inputs = [])
    (embedding_variable : [ `float ] t)
    (sliced_activations : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("table_id", Int table_id) :: attributes
  in
  let attributes =
    ("lookup_id", Int lookup_id) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tPUEmbeddingActivations in
  let inputs = [ (`single (P embedding_variable)); (`single (P sliced_activations)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tPUOrdinalSelector
    ?(name = "TPUOrdinalSelector")
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tPUOrdinalSelector in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tPUReplicateMetadata
    ?(name = "TPUReplicateMetadata")
    ~num_replicas
    ?num_cores_per_replica
    ?topology
    ?use_tpu
    ?device_assignment
    ?computation_shape
    ?step_marker_location
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    ("num_replicas", Int num_replicas) :: attributes
  in
  let attributes =
    match num_cores_per_replica with | None -> attributes | Some num_cores_per_replica -> ("num_cores_per_replica", Int num_cores_per_replica) :: attributes
  in
  let attributes =
    match topology with | None -> attributes | Some topology -> ("topology", String topology) :: attributes
  in
  let attributes =
    match use_tpu with | None -> attributes | Some use_tpu -> ("use_tpu", Bool use_tpu) :: attributes
  in
  let attributes =
    match device_assignment with | None -> attributes | Some device_assignment -> ("device_assignment", List (Int device_assignment)) :: attributes
  in
  let attributes =
    match computation_shape with | None -> attributes | Some computation_shape -> ("computation_shape", List (Int computation_shape)) :: attributes
  in
  let attributes =
    match step_marker_location with | None -> attributes | Some step_marker_location -> ("step_marker_location", String step_marker_location) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tPUReplicateMetadata in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tPUReplicatedInput
    ?(name = "TPUReplicatedInput")
    ?(control_inputs = [])
    (inputs__ : 't t list)
  =
  let attributes = [ "T", Type (P (Node.output_type (List.hd_exn inputs__))) ] in
  let attributes =
    ("N", Int (List.length inputs__)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tPUReplicatedInput in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) inputs__)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type (List.hd_exn inputs__))
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tPUReplicatedOutput
    ?(name = "TPUReplicatedOutput")
    ~num_replicas
    ?(control_inputs = [])
    (input : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("num_replicas", Int num_replicas) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tPUReplicatedOutput in
  let inputs = [ (`single (P input)) ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:(Node.output_type input)
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init num_replicas ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let takeDataset
    ?(name = "TakeDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (count : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.takeDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P count)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let takeManySparseFromTensorsMap
    ?(name = "TakeManySparseFromTensorsMap")
    ~type_1
    ?container
    ?shared_name
    ?(control_inputs = [])
    (sparse_handles : [ `int64 ] t)
  =
  let attributes = [ "dtype", Type (P type_1) ] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.takeManySparseFromTensorsMap in
  let inputs = [ (`single (P sparse_handles)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let tan
    ?(name = "Tan")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tan in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tanh
    ?(name = "Tanh")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tanh in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tanhGrad
    ?(name = "TanhGrad")
    ?(control_inputs = [])
    (y : ([< `float | `double | `complex64 ] as 't) t)
    (dy : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type y)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tanhGrad in
  let inputs = [ (`single (P y)); (`single (P dy)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type y)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let temporaryVariable
    ?(name = "TemporaryVariable")
    ~type_
    ~shape
    ?var_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    ("shape", Shape shape) :: attributes
  in
  let attributes =
    match var_name with | None -> attributes | Some var_name -> ("var_name", String var_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.temporaryVariable in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArray
    ?(name = "TensorArray")
    ?dynamic_size
    ?clear_after_read
    ?tensor_array_name
    ?element_shape
    ?(control_inputs = [])
    (size : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match dynamic_size with | None -> attributes | Some dynamic_size -> ("dynamic_size", Bool dynamic_size) :: attributes
  in
  let attributes =
    match clear_after_read with | None -> attributes | Some clear_after_read -> ("clear_after_read", Bool clear_after_read) :: attributes
  in
  let attributes =
    match tensor_array_name with | None -> attributes | Some tensor_array_name -> ("tensor_array_name", String tensor_array_name) :: attributes
  in
  let attributes =
    match element_shape with | None -> attributes | Some element_shape -> ("element_shape", Shape element_shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArray in
  let inputs = [ (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayClose
    ?(name = "TensorArrayClose")
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayClose in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayCloseV2
    ?(name = "TensorArrayCloseV2")
    ?(control_inputs = [])
    (handle : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayCloseV2 in
  let inputs = [ (`single (P handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayConcat
    ?(name = "TensorArrayConcat")
    ~type_
    ?element_shape_except0
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match element_shape_except0 with | None -> attributes | Some element_shape_except0 -> ("element_shape_except0", Shape element_shape_except0) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayConcat in
  let inputs = [ (`single (P handle)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let tensorArrayConcatV2
    ?(name = "TensorArrayConcatV2")
    ~type_
    ?element_shape_except0
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match element_shape_except0 with | None -> attributes | Some element_shape_except0 -> ("element_shape_except0", Shape element_shape_except0) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayConcatV2 in
  let inputs = [ (`single (P handle)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let tensorArrayGather
    ?(name = "TensorArrayGather")
    ~type_
    ?element_shape
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (indices : [ `int32 ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match element_shape with | None -> attributes | Some element_shape -> ("element_shape", Shape element_shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayGather in
  let inputs = [ (`single (P handle)); (`single (P indices)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayGatherV2
    ?(name = "TensorArrayGatherV2")
    ~type_
    ?element_shape
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (indices : [ `int32 ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match element_shape with | None -> attributes | Some element_shape -> ("element_shape", Shape element_shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayGatherV2 in
  let inputs = [ (`single (P handle)); (`single (P indices)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayGrad
    ?(name = "TensorArrayGrad")
    ~source
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("source", String source) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayGrad in
  let inputs = [ (`single (P handle)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayGradV2
    ?(name = "TensorArrayGradV2")
    ~source
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  let attributes =
    ("source", String source) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayGradV2 in
  let inputs = [ (`single (P handle)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayPack
    ?(name = "TensorArrayPack")
    ~type_
    ?element_shape
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let attributes =
    match element_shape with | None -> attributes | Some element_shape -> ("element_shape", Shape element_shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayPack in
  let inputs = [ (`single (P handle)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayRead
    ?(name = "TensorArrayRead")
    ~type_
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (index : [ `int32 ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayRead in
  let inputs = [ (`single (P handle)); (`single (P index)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayReadV2
    ?(name = "TensorArrayReadV2")
    ~type_
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (index : [ `int32 ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayReadV2 in
  let inputs = [ (`single (P handle)); (`single (P index)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayScatter
    ?(name = "TensorArrayScatter")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (indices : [ `int32 ] t)
    (value : 't t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayScatter in
  let inputs = [ (`single (P handle)); (`single (P indices)); (`single (P value)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayScatterV2
    ?(name = "TensorArrayScatterV2")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (indices : [ `int32 ] t)
    (value : 't t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayScatterV2 in
  let inputs = [ (`single (P handle)); (`single (P indices)); (`single (P value)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArraySize
    ?(name = "TensorArraySize")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArraySize in
  let inputs = [ (`single (P handle)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArraySizeV2
    ?(name = "TensorArraySizeV2")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArraySizeV2 in
  let inputs = [ (`single (P handle)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArraySplit
    ?(name = "TensorArraySplit")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (value : 't t)
    (lengths : [ `int64 ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArraySplit in
  let inputs = [ (`single (P handle)); (`single (P value)); (`single (P lengths)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArraySplitV2
    ?(name = "TensorArraySplitV2")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (value : 't t)
    (lengths : [ `int64 ] t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArraySplitV2 in
  let inputs = [ (`single (P handle)); (`single (P value)); (`single (P lengths)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayUnpack
    ?(name = "TensorArrayUnpack")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (value : 't t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayUnpack in
  let inputs = [ (`single (P handle)); (`single (P value)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayV2
    ?(name = "TensorArrayV2")
    ?element_shape
    ?dynamic_size
    ?clear_after_read
    ?tensor_array_name
    ?(control_inputs = [])
    (size : [ `int32 ] t)
  =
  let attributes = [] in
  let attributes =
    match element_shape with | None -> attributes | Some element_shape -> ("element_shape", Shape element_shape) :: attributes
  in
  let attributes =
    match dynamic_size with | None -> attributes | Some dynamic_size -> ("dynamic_size", Bool dynamic_size) :: attributes
  in
  let attributes =
    match clear_after_read with | None -> attributes | Some clear_after_read -> ("clear_after_read", Bool clear_after_read) :: attributes
  in
  let attributes =
    match tensor_array_name with | None -> attributes | Some tensor_array_name -> ("tensor_array_name", String tensor_array_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayV2 in
  let inputs = [ (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayWrite
    ?(name = "TensorArrayWrite")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (index : [ `int32 ] t)
    (value : 't t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayWrite in
  let inputs = [ (`single (P handle)); (`single (P index)); (`single (P value)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorArrayWriteV2
    ?(name = "TensorArrayWriteV2")
    ?(control_inputs = [])
    (handle : [ `string ] t)
    (index : [ `int32 ] t)
    (value : 't t)
    (flow_in : [ `float ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorArrayWriteV2 in
  let inputs = [ (`single (P handle)); (`single (P index)); (`single (P value)); (`single (P flow_in)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListConcat
    ?(name = "TensorListConcat")
    ~type_
    ?element_shape
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
  =
  let attributes = [ "element_dtype", Type (P type_) ] in
  let attributes =
    match element_shape with | None -> attributes | Some element_shape -> ("element_shape", Shape element_shape) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListConcat in
  let inputs = [ (`single (P input_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let tensorListConcatLists
    ?(name = "TensorListConcatLists")
    ?(control_inputs = [])
    (input_a : [ `variant ] t)
    (input_b : [ `variant ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListConcatLists in
  let inputs = [ (`single (P input_a)); (`single (P input_b)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListConcatV2
    ?(name = "TensorListConcatV2")
    ~type_
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
    (element_shape : ([< `int32 | `int64 ] as 'shape_type) t)
    (leading_dims : [ `int64 ] t)
  =
  let attributes = [ "shape_type", Type (P (Node.output_type element_shape)) ;  "element_dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListConcatV2 in
  let inputs = [ (`single (P input_handle)); (`single (P element_shape)); (`single (P leading_dims)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let tensorListElementShape
    ?(name = "TensorListElementShape")
    ~type_
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
  =
  let attributes = [ "shape_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListElementShape in
  let inputs = [ (`single (P input_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListFromTensor
    ?(name = "TensorListFromTensor")
    ?(control_inputs = [])
    (tensor : 'element_dtype t)
    (element_shape : ([< `int32 | `int64 ] as 'shape_type) t)
  =
  let attributes = [ "shape_type", Type (P (Node.output_type element_shape)) ;  "element_dtype", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListFromTensor in
  let inputs = [ (`single (P tensor)); (`single (P element_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListGather
    ?(name = "TensorListGather")
    ~type_
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
    (indices : [ `int32 ] t)
    (element_shape : [ `int32 ] t)
  =
  let attributes = [ "element_dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListGather in
  let inputs = [ (`single (P input_handle)); (`single (P indices)); (`single (P element_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListGetItem
    ?(name = "TensorListGetItem")
    ~type_
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
    (index : [ `int32 ] t)
    (element_shape : [ `int32 ] t)
  =
  let attributes = [ "element_dtype", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListGetItem in
  let inputs = [ (`single (P input_handle)); (`single (P index)); (`single (P element_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListLength
    ?(name = "TensorListLength")
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListLength in
  let inputs = [ (`single (P input_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListPopBack
    ?(name = "TensorListPopBack")
    ~type_1
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
    (element_shape : [ `int32 ] t)
  =
  let attributes = [ "element_dtype", Type (P type_1) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListPopBack in
  let inputs = [ (`single (P input_handle)); (`single (P element_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let tensorListPushBack
    ?(name = "TensorListPushBack")
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
    (tensor : 'element_dtype t)
  =
  let attributes = [ "element_dtype", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListPushBack in
  let inputs = [ (`single (P input_handle)); (`single (P tensor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListPushBackBatch
    ?(name = "TensorListPushBackBatch")
    ?(control_inputs = [])
    (input_handles : [ `variant ] t)
    (tensor : 'element_dtype t)
  =
  let attributes = [ "element_dtype", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListPushBackBatch in
  let inputs = [ (`single (P input_handles)); (`single (P tensor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListReserve
    ?(name = "TensorListReserve")
    ?(control_inputs = [])
    (element_shape : ([< `int32 | `int64 ] as 'shape_type) t)
    (num_elements : [ `int32 ] t)
  =
  let attributes = [ "shape_type", Type (P (Node.output_type element_shape)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListReserve in
  let inputs = [ (`single (P element_shape)); (`single (P num_elements)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListResize
    ?(name = "TensorListResize")
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
    (size : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListResize in
  let inputs = [ (`single (P input_handle)); (`single (P size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListScatter
    ?(name = "TensorListScatter")
    ?(control_inputs = [])
    (tensor : 'element_dtype t)
    (indices : [ `int32 ] t)
    (element_shape : ([< `int32 | `int64 ] as 'shape_type) t)
  =
  let attributes = [ "shape_type", Type (P (Node.output_type element_shape)) ;  "element_dtype", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListScatter in
  let inputs = [ (`single (P tensor)); (`single (P indices)); (`single (P element_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListScatterIntoExistingList
    ?(name = "TensorListScatterIntoExistingList")
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
    (tensor : 'element_dtype t)
    (indices : [ `int32 ] t)
  =
  let attributes = [ "element_dtype", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListScatterIntoExistingList in
  let inputs = [ (`single (P input_handle)); (`single (P tensor)); (`single (P indices)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListScatterV2
    ?(name = "TensorListScatterV2")
    ?(control_inputs = [])
    (tensor : 'element_dtype t)
    (indices : [ `int32 ] t)
    (element_shape : ([< `int32 | `int64 ] as 'shape_type) t)
    (num_elements : [ `int32 ] t)
  =
  let attributes = [ "shape_type", Type (P (Node.output_type element_shape)) ;  "element_dtype", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListScatterV2 in
  let inputs = [ (`single (P tensor)); (`single (P indices)); (`single (P element_shape)); (`single (P num_elements)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListSetItem
    ?(name = "TensorListSetItem")
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
    (index : [ `int32 ] t)
    (item : 'element_dtype t)
  =
  let attributes = [ "element_dtype", Type (P (Node.output_type item)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListSetItem in
  let inputs = [ (`single (P input_handle)); (`single (P index)); (`single (P item)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListSplit
    ?(name = "TensorListSplit")
    ?(control_inputs = [])
    (tensor : 'element_dtype t)
    (element_shape : ([< `int32 | `int64 ] as 'shape_type) t)
    (lengths : [ `int64 ] t)
  =
  let attributes = [ "shape_type", Type (P (Node.output_type element_shape)) ;  "element_dtype", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListSplit in
  let inputs = [ (`single (P tensor)); (`single (P element_shape)); (`single (P lengths)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorListStack
    ?(name = "TensorListStack")
    ~type_
    ?num_elements
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
    (element_shape : [ `int32 ] t)
  =
  let attributes = [ "element_dtype", Type (P type_) ] in
  let attributes =
    match num_elements with | None -> attributes | Some num_elements -> ("num_elements", Int num_elements) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorListStack in
  let inputs = [ (`single (P input_handle)); (`single (P element_shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorScatterAdd
    ?(name = "TensorScatterAdd")
    ?(control_inputs = [])
    (tensor : 't t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : 't t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorScatterAdd in
  let inputs = [ (`single (P tensor)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type tensor)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorScatterSub
    ?(name = "TensorScatterSub")
    ?(control_inputs = [])
    (tensor : 't t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : 't t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorScatterSub in
  let inputs = [ (`single (P tensor)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type tensor)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorScatterUpdate
    ?(name = "TensorScatterUpdate")
    ?(control_inputs = [])
    (tensor : 't t)
    (indices : ([< `int32 | `int64 ] as 'tindices) t)
    (updates : 't t)
  =
  let attributes = [ "Tindices", Type (P (Node.output_type indices)) ;  "T", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorScatterUpdate in
  let inputs = [ (`single (P tensor)); (`single (P indices)); (`single (P updates)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type tensor)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorStridedSliceUpdate
    ?(name = "TensorStridedSliceUpdate")
    ?begin_mask
    ?end_mask
    ?ellipsis_mask
    ?new_axis_mask
    ?shrink_axis_mask
    ?(control_inputs = [])
    (input : 't t)
    (begin__ : ([< `int32 | `int64 ] as 'index) t)
    (end__ : ([< `int32 | `int64 ] as 'index) t)
    (strides : ([< `int32 | `int64 ] as 'index) t)
    (value : 't t)
  =
  let attributes = [ "Index", Type (P (Node.output_type begin__)) ;  "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match begin_mask with | None -> attributes | Some begin_mask -> ("begin_mask", Int begin_mask) :: attributes
  in
  let attributes =
    match end_mask with | None -> attributes | Some end_mask -> ("end_mask", Int end_mask) :: attributes
  in
  let attributes =
    match ellipsis_mask with | None -> attributes | Some ellipsis_mask -> ("ellipsis_mask", Int ellipsis_mask) :: attributes
  in
  let attributes =
    match new_axis_mask with | None -> attributes | Some new_axis_mask -> ("new_axis_mask", Int new_axis_mask) :: attributes
  in
  let attributes =
    match shrink_axis_mask with | None -> attributes | Some shrink_axis_mask -> ("shrink_axis_mask", Int shrink_axis_mask) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorStridedSliceUpdate in
  let inputs = [ (`single (P input)); (`single (P begin__)); (`single (P end__)); (`single (P strides)); (`single (P value)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorSummary
    ?(name = "TensorSummary")
    ?description
    ?display_name
    ?(control_inputs = [])
    (tensor : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type tensor)) ] in
  let attributes =
    match description with | None -> attributes | Some description -> ("description", String description) :: attributes
  in
  let attributes =
    match display_name with | None -> attributes | Some display_name -> ("display_name", String display_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tensorSummary in
  let inputs = [ (`single (P tensor)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tensorSummaryV2
    ?(name = "TensorSummaryV2")
    ?(control_inputs = [])
    (tag : [ `string ] t)
    (tensor : 't t)
    (serialized_summary_metadata : [ `string ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type tensor)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tensorSummaryV2 in
  let inputs = [ (`single (P tag)); (`single (P tensor)); (`single (P serialized_summary_metadata)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let textLineDataset
    ?(name = "TextLineDataset")
    ?(control_inputs = [])
    (filenames : [ `string ] t)
    (compression_type : [ `string ] t)
    (buffer_size : [ `int64 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.textLineDataset in
  let inputs = [ (`single (P filenames)); (`single (P compression_type)); (`single (P buffer_size)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let textLineReader
    ?(name = "TextLineReader")
    ?skip_header_lines
    ?container
    ?shared_name
    ?(control_inputs = [])
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
  let name = Name.of_string name in
  let op_name = Op_names.textLineReader in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let threadUnsafeUnigramCandidateSampler
    ?(name = "ThreadUnsafeUnigramCandidateSampler")
    ~num_true
    ~num_sampled
    ~unique
    ~range_max
    ?seed
    ?seed2
    ?(control_inputs = [])
    (true_classes : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_true", Int num_true) :: attributes
  in
  let attributes =
    ("num_sampled", Int num_sampled) :: attributes
  in
  let attributes =
    ("unique", Bool unique) :: attributes
  in
  let attributes =
    ("range_max", Int range_max) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.threadUnsafeUnigramCandidateSampler in
  let inputs = [ (`single (P true_classes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let tile
    ?(name = "Tile")
    ?(control_inputs = [])
    (input : 't t)
    (multiples : ([< `int32 | `int64 ] as 'tmultiples) t)
  =
  let attributes = [ "Tmultiples", Type (P (Node.output_type multiples)) ;  "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tile in
  let inputs = [ (`single (P input)); (`single (P multiples)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tileGrad
    ?(name = "TileGrad")
    ?(control_inputs = [])
    (input : 't t)
    (multiples : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tileGrad in
  let inputs = [ (`single (P input)); (`single (P multiples)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let timestamp
    ?(name = "Timestamp")
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.timestamp in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Double
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let topK
    ?(name = "TopK")
    ~k
    ?sorted
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    ("k", Int k) :: attributes
  in
  let attributes =
    match sorted with | None -> attributes | Some sorted -> ("sorted", Bool sorted) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.topK in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let topKV2
    ?(name = "TopKV2")
    ?sorted
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (k : [ `int32 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let attributes =
    match sorted with | None -> attributes | Some sorted -> ("sorted", Bool sorted) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.topKV2 in
  let inputs = [ (`single (P input)); (`single (P k)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let transpose
    ?(name = "Transpose")
    ?(control_inputs = [])
    (x : 't t)
    (perm : ([< `int32 | `int64 ] as 'tperm) t)
  =
  let attributes = [ "Tperm", Type (P (Node.output_type perm)) ;  "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.transpose in
  let inputs = [ (`single (P x)); (`single (P perm)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tridiagonalSolve
    ?(name = "TridiagonalSolve")
    ?(control_inputs = [])
    (diagonals : ([< `double | `float | `complex64 ] as 't) t)
    (rhs : ([< `double | `float | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type diagonals)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.tridiagonalSolve in
  let inputs = [ (`single (P diagonals)); (`single (P rhs)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type diagonals)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let truncateDiv
    ?(name = "TruncateDiv")
    ?(control_inputs = [])
    (x : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
    (y : ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.truncateDiv in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let truncateMod
    ?(name = "TruncateMod")
    ?(control_inputs = [])
    (x : ([< `int32 | `int64 | `float | `double ] as 't) t)
    (y : ([< `int32 | `int64 | `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.truncateMod in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let truncatedNormal
    ?(name = "TruncatedNormal")
    ~type_
    ?seed
    ?seed2
    ?(control_inputs = [])
    (shape : ([< `int32 | `int64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type shape)) ;  "dtype", Type (P type_) ] in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.truncatedNormal in
  let inputs = [ (`single (P shape)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let tryRpc
    ?(name = "TryRpc")
    ?protocol
    ?fail_fast
    ?timeout_in_ms
    ?(control_inputs = [])
    (address : [ `string ] t)
    (method__ : [ `string ] t)
    (request : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    match protocol with | None -> attributes | Some protocol -> ("protocol", String protocol) :: attributes
  in
  let attributes =
    match fail_fast with | None -> attributes | Some fail_fast -> ("fail_fast", Bool fail_fast) :: attributes
  in
  let attributes =
    match timeout_in_ms with | None -> attributes | Some timeout_in_ms -> ("timeout_in_ms", Int timeout_in_ms) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.tryRpc in
  let inputs = [ (`single (P address)); (`single (P method__)); (`single (P request)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let unbatch
    ?(name = "Unbatch")
    ~timeout_micros
    ?container
    ?shared_name
    ?(control_inputs = [])
    (batched_tensor : 't t)
    (batch_index : [ `int64 ] t)
    (id : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type batched_tensor)) ] in
  let attributes =
    ("timeout_micros", Int timeout_micros) :: attributes
  in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.unbatch in
  let inputs = [ (`single (P batched_tensor)); (`single (P batch_index)); (`single (P id)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type batched_tensor)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let unbatchGrad
    ?(name = "UnbatchGrad")
    ?container
    ?shared_name
    ?(control_inputs = [])
    (original_input : 't t)
    (batch_index : [ `int64 ] t)
    (grad : 't t)
    (id : [ `int64 ] t)
  =
  let attributes = [ "T", Type (P (Node.output_type original_input)) ] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.unbatchGrad in
  let inputs = [ (`single (P original_input)); (`single (P batch_index)); (`single (P grad)); (`single (P id)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type original_input)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let unicodeDecode
    ?(name = "UnicodeDecode")
    ~input_encoding
    ?errors
    ?replacement_char
    ?replace_control_characters
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("input_encoding", String input_encoding) :: attributes
  in
  let attributes =
    match errors with | None -> attributes | Some errors -> ("errors", String errors) :: attributes
  in
  let attributes =
    match replacement_char with | None -> attributes | Some replacement_char -> ("replacement_char", Int replacement_char) :: attributes
  in
  let attributes =
    match replace_control_characters with | None -> attributes | Some replace_control_characters -> ("replace_control_characters", Bool replace_control_characters) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.unicodeDecode in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let unicodeDecodeWithOffsets
    ?(name = "UnicodeDecodeWithOffsets")
    ~input_encoding
    ?errors
    ?replacement_char
    ?replace_control_characters
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("input_encoding", String input_encoding) :: attributes
  in
  let attributes =
    match errors with | None -> attributes | Some errors -> ("errors", String errors) :: attributes
  in
  let attributes =
    match replacement_char with | None -> attributes | Some replacement_char -> ("replacement_char", Int replacement_char) :: attributes
  in
  let attributes =
    match replace_control_characters with | None -> attributes | Some replace_control_characters -> ("replace_control_characters", Bool replace_control_characters) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.unicodeDecodeWithOffsets in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let unicodeEncode
    ?(name = "UnicodeEncode")
    ?errors
    ~output_encoding
    ?replacement_char
    ?(control_inputs = [])
    (input_values : [ `int32 ] t)
    (input_splits : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    match errors with | None -> attributes | Some errors -> ("errors", String errors) :: attributes
  in
  let attributes =
    ("output_encoding", String output_encoding) :: attributes
  in
  let attributes =
    match replacement_char with | None -> attributes | Some replacement_char -> ("replacement_char", Int replacement_char) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.unicodeEncode in
  let inputs = [ (`single (P input_values)); (`single (P input_splits)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let unicodeScript
    ?(name = "UnicodeScript")
    ?(control_inputs = [])
    (input : [ `int32 ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.unicodeScript in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int32
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let unicodeTranscode
    ?(name = "UnicodeTranscode")
    ~input_encoding
    ~output_encoding
    ?errors
    ?replacement_char
    ?replace_control_characters
    ?(control_inputs = [])
    (input : [ `string ] t)
  =
  let attributes = [] in
  let attributes =
    ("input_encoding", String input_encoding) :: attributes
  in
  let attributes =
    ("output_encoding", String output_encoding) :: attributes
  in
  let attributes =
    match errors with | None -> attributes | Some errors -> ("errors", String errors) :: attributes
  in
  let attributes =
    match replacement_char with | None -> attributes | Some replacement_char -> ("replacement_char", Int replacement_char) :: attributes
  in
  let attributes =
    match replace_control_characters with | None -> attributes | Some replace_control_characters -> ("replace_control_characters", Bool replace_control_characters) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.unicodeTranscode in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let uniformCandidateSampler
    ?(name = "UniformCandidateSampler")
    ~num_true
    ~num_sampled
    ~unique
    ~range_max
    ?seed
    ?seed2
    ?(control_inputs = [])
    (true_classes : [ `int64 ] t)
  =
  let attributes = [] in
  let attributes =
    ("num_true", Int num_true) :: attributes
  in
  let attributes =
    ("num_sampled", Int num_sampled) :: attributes
  in
  let attributes =
    ("unique", Bool unique) :: attributes
  in
  let attributes =
    ("range_max", Int range_max) :: attributes
  in
  let attributes =
    match seed with | None -> attributes | Some seed -> ("seed", Int seed) :: attributes
  in
  let attributes =
    match seed2 with | None -> attributes | Some seed2 -> ("seed2", Int seed2) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.uniformCandidateSampler in
  let inputs = [ (`single (P true_classes)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Float
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let unique
    ?(name = "Unique")
    ~type_1
    ?(control_inputs = [])
    (x : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ;  "out_idx", Type (P type_1) ] in
  let name = Name.of_string name in
  let op_name = Op_names.unique in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let uniqueV2
    ?(name = "UniqueV2")
    ~type_1
    ?(control_inputs = [])
    (x : 't t)
    (axis : ([< `int32 | `int64 ] as 'taxis) t)
  =
  let attributes = [ "Taxis", Type (P (Node.output_type axis)) ;  "T", Type (P (Node.output_type x)) ;  "out_idx", Type (P type_1) ] in
  let name = Name.of_string name in
  let op_name = Op_names.uniqueV2 in
  let inputs = [ (`single (P x)); (`single (P axis)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)

let uniqueWithCounts
    ?(name = "UniqueWithCounts")
    ~type_1
    ~type_2
    ?(control_inputs = [])
    (x : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ;  "out_idx", Type (P type_1) ;  "out_idx", Type (P type_2) ] in
  let name = Name.of_string name in
  let op_name = Op_names.uniqueWithCounts in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_2
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let uniqueWithCountsV2
    ?(name = "UniqueWithCountsV2")
    ~type_1
    ~type_2
    ?(control_inputs = [])
    (x : 't t)
    (axis : ([< `int32 | `int64 ] as 'taxis) t)
  =
  let attributes = [ "Taxis", Type (P (Node.output_type axis)) ;  "T", Type (P (Node.output_type x)) ;  "out_idx", Type (P type_1) ;  "out_idx", Type (P type_2) ] in
  let name = Name.of_string name in
  let op_name = Op_names.uniqueWithCountsV2 in
  let inputs = [ (`single (P x)); (`single (P axis)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 0)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_1
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 1)
  ,
  Node.create
    ~name
    ~op_name
    ~output_type:type_2
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:(Some 2)

let unpack
    ?(name = "Unpack")
    ~num
    ?axis
    ?(control_inputs = [])
    (value : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type value)) ] in
  let attributes =
    ("num", Int num) :: attributes
  in
  let attributes =
    match axis with | None -> attributes | Some axis -> ("axis", Int axis) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.unpack in
  let inputs = [ (`single (P value)) ] in
  let node =
    Node.create
      ~name
      ~op_name
      ~output_type:(Node.output_type value)
      ~inputs
      ~control_inputs
      ~attributes
      ~output_idx:None
  in
  List.init num ~f:(fun output_idx ->
    set_output_idx node (Some output_idx))

let unravelIndex
    ?(name = "UnravelIndex")
    ?(control_inputs = [])
    (indices : ([< `int32 | `int64 ] as 'tidx) t)
    (dims : ([< `int32 | `int64 ] as 'tidx) t)
  =
  let attributes = [ "Tidx", Type (P (Node.output_type indices)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.unravelIndex in
  let inputs = [ (`single (P indices)); (`single (P dims)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type indices)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let unsortedSegmentMax
    ?(name = "UnsortedSegmentMax")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
    (num_segments : ([< `int32 | `int64 ] as 'tnumsegments) t)
  =
  let attributes = [ "Tnumsegments", Type (P (Node.output_type num_segments)) ;  "Tindices", Type (P (Node.output_type segment_ids)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.unsortedSegmentMax in
  let inputs = [ (`single (P data)); (`single (P segment_ids)); (`single (P num_segments)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let unsortedSegmentMin
    ?(name = "UnsortedSegmentMin")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
    (num_segments : ([< `int32 | `int64 ] as 'tnumsegments) t)
  =
  let attributes = [ "Tnumsegments", Type (P (Node.output_type num_segments)) ;  "Tindices", Type (P (Node.output_type segment_ids)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.unsortedSegmentMin in
  let inputs = [ (`single (P data)); (`single (P segment_ids)); (`single (P num_segments)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let unsortedSegmentProd
    ?(name = "UnsortedSegmentProd")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
    (num_segments : ([< `int32 | `int64 ] as 'tnumsegments) t)
  =
  let attributes = [ "Tnumsegments", Type (P (Node.output_type num_segments)) ;  "Tindices", Type (P (Node.output_type segment_ids)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.unsortedSegmentProd in
  let inputs = [ (`single (P data)); (`single (P segment_ids)); (`single (P num_segments)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let unsortedSegmentSum
    ?(name = "UnsortedSegmentSum")
    ?(control_inputs = [])
    (data : ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t)
    (segment_ids : ([< `int32 | `int64 ] as 'tindices) t)
    (num_segments : ([< `int32 | `int64 ] as 'tnumsegments) t)
  =
  let attributes = [ "Tnumsegments", Type (P (Node.output_type num_segments)) ;  "Tindices", Type (P (Node.output_type segment_ids)) ;  "T", Type (P (Node.output_type data)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.unsortedSegmentSum in
  let inputs = [ (`single (P data)); (`single (P segment_ids)); (`single (P num_segments)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type data)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let unwrapDatasetVariant
    ?(name = "UnwrapDatasetVariant")
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.unwrapDatasetVariant in
  let inputs = [ (`single (P input_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let upperBound
    ?(name = "UpperBound")
    ~type_
    ?(control_inputs = [])
    (sorted_inputs : 't t)
    (values : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type sorted_inputs)) ;  "out_type", Type (P type_) ] in
  let name = Name.of_string name in
  let op_name = Op_names.upperBound in
  let inputs = [ (`single (P sorted_inputs)); (`single (P values)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let variable
    ?(name = "Variable")
    ~type_
    ~shape
    ?container
    ?shared_name
    ?(control_inputs = [])
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
  let name = Name.of_string name in
  let op_name = Op_names.variable in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let variableV2
    ?(name = "VariableV2")
    ~type_
    ~shape
    ?container
    ?shared_name
    ?(control_inputs = [])
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
  let name = Name.of_string name in
  let op_name = Op_names.variableV2 in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:type_
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let where
    ?(name = "Where")
    ?(control_inputs = [])
    (input : ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type input)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.where in
  let inputs = [ (`single (P input)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Int64
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let wholeFileReader
    ?(name = "WholeFileReader")
    ?container
    ?shared_name
    ?(control_inputs = [])
    ()
  =
  let attributes = [] in
  let attributes =
    match container with | None -> attributes | Some container -> ("container", String container) :: attributes
  in
  let attributes =
    match shared_name with | None -> attributes | Some shared_name -> ("shared_name", String shared_name) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.wholeFileReader in
  let inputs = [  ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let windowDataset
    ?(name = "WindowDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_dataset : [ `variant ] t)
    (size : [ `int64 ] t)
    (shift : [ `int64 ] t)
    (stride : [ `int64 ] t)
    (drop_remainder : [ `bool ] t)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.windowDataset in
  let inputs = [ (`single (P input_dataset)); (`single (P size)); (`single (P shift)); (`single (P stride)); (`single (P drop_remainder)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let workerHeartbeat
    ?(name = "WorkerHeartbeat")
    ?(control_inputs = [])
    (request : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.workerHeartbeat in
  let inputs = [ (`single (P request)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.String
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let wrapDatasetVariant
    ?(name = "WrapDatasetVariant")
    ?(control_inputs = [])
    (input_handle : [ `variant ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.wrapDatasetVariant in
  let inputs = [ (`single (P input_handle)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let writeFile
    ?(name = "WriteFile")
    ?(control_inputs = [])
    (filename : [ `string ] t)
    (contents : [ `string ] t)
  =
  let attributes = [] in
  let name = Name.of_string name in
  let op_name = Op_names.writeFile in
  let inputs = [ (`single (P filename)); (`single (P contents)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Unit
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let xdivy
    ?(name = "Xdivy")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
    (y : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.xdivy in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let xlogy
    ?(name = "Xlogy")
    ?(control_inputs = [])
    (x : ([< `float | `double | `complex64 ] as 't) t)
    (y : ([< `float | `double | `complex64 ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.xlogy in
  let inputs = [ (`single (P x)); (`single (P y)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let zerosLike
    ?(name = "ZerosLike")
    ?(control_inputs = [])
    (x : 't t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.zerosLike in
  let inputs = [ (`single (P x)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let zeta
    ?(name = "Zeta")
    ?(control_inputs = [])
    (x : ([< `float | `double ] as 't) t)
    (q : ([< `float | `double ] as 't) t)
  =
  let attributes = [ "T", Type (P (Node.output_type x)) ] in
  let name = Name.of_string name in
  let op_name = Op_names.zeta in
  let inputs = [ (`single (P x)); (`single (P q)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:(Node.output_type x)
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

let zipDataset
    ?(name = "ZipDataset")
    ~output_types
    ~output_shapes
    ?(control_inputs = [])
    (input_datasets : [ `variant ] t list)
  =
  let attributes = [] in
  let attributes =
    ("output_types", List (Type output_types)) :: attributes
  in
  let attributes =
    ("output_shapes", List (Shape output_shapes)) :: attributes
  in
  let attributes =
    ("N", Int (List.length input_datasets)) :: attributes
  in
  let name = Name.of_string name in
  let op_name = Op_names.zipDataset in
  let inputs = [ (`multi (List.map ~f:(fun n -> P n) input_datasets)) ] in
  Node.create
    ~name
    ~op_name
    ~output_type:Type.Variant
    ~inputs
    ~control_inputs
    ~attributes
    ~output_idx:None

