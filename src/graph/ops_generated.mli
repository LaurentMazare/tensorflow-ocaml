(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)
open Node

module Op_names : sig
  val abort : Op_name.t
  val abs : Op_name.t
  val accumulateNV2 : Op_name.t
  val accumulatorApplyGradient : Op_name.t
  val accumulatorNumAccumulated : Op_name.t
  val accumulatorSetGlobalStep : Op_name.t
  val accumulatorTakeGradient : Op_name.t
  val acos : Op_name.t
  val acosh : Op_name.t
  val add : Op_name.t
  val addManySparseToTensorsMap : Op_name.t
  val addN : Op_name.t
  val addSparseToTensorsMap : Op_name.t
  val addV2 : Op_name.t
  val adjustContrast : Op_name.t
  val adjustContrastv2 : Op_name.t
  val adjustHue : Op_name.t
  val adjustSaturation : Op_name.t
  val all : Op_name.t
  val allCandidateSampler : Op_name.t
  val allToAll : Op_name.t
  val angle : Op_name.t
  val any : Op_name.t
  val applyAdaMax : Op_name.t
  val applyAdadelta : Op_name.t
  val applyAdagrad : Op_name.t
  val applyAdagradDA : Op_name.t
  val applyAdam : Op_name.t
  val applyAddSign : Op_name.t
  val applyCenteredRMSProp : Op_name.t
  val applyFtrl : Op_name.t
  val applyFtrlV2 : Op_name.t
  val applyGradientDescent : Op_name.t
  val applyMomentum : Op_name.t
  val applyPowerSign : Op_name.t
  val applyProximalAdagrad : Op_name.t
  val applyProximalGradientDescent : Op_name.t
  val applyRMSProp : Op_name.t
  val approximateEqual : Op_name.t
  val argMax : Op_name.t
  val argMin : Op_name.t
  val asString : Op_name.t
  val asin : Op_name.t
  val asinh : Op_name.t
  val assign : Op_name.t
  val assignAdd : Op_name.t
  val assignSub : Op_name.t
  val atan : Op_name.t
  val atan2 : Op_name.t
  val atanh : Op_name.t
  val audioSpectrogram : Op_name.t
  val audioSummary : Op_name.t
  val audioSummaryV2 : Op_name.t
  val avgPool : Op_name.t
  val avgPool3D : Op_name.t
  val avgPool3DGrad : Op_name.t
  val avgPoolGrad : Op_name.t
  val barrier : Op_name.t
  val barrierClose : Op_name.t
  val barrierIncompleteSize : Op_name.t
  val barrierInsertMany : Op_name.t
  val barrierReadySize : Op_name.t
  val batchCholesky : Op_name.t
  val batchCholeskyGrad : Op_name.t
  val batchDataset : Op_name.t
  val batchDatasetV2 : Op_name.t
  val batchFFT : Op_name.t
  val batchFFT2D : Op_name.t
  val batchFFT3D : Op_name.t
  val batchIFFT : Op_name.t
  val batchIFFT2D : Op_name.t
  val batchIFFT3D : Op_name.t
  val batchMatMul : Op_name.t
  val batchMatMulV2 : Op_name.t
  val batchMatrixBandPart : Op_name.t
  val batchMatrixDeterminant : Op_name.t
  val batchMatrixDiag : Op_name.t
  val batchMatrixDiagPart : Op_name.t
  val batchMatrixInverse : Op_name.t
  val batchMatrixSetDiag : Op_name.t
  val batchMatrixSolve : Op_name.t
  val batchMatrixSolveLs : Op_name.t
  val batchMatrixTriangularSolve : Op_name.t
  val batchNormWithGlobalNormalization : Op_name.t
  val batchNormWithGlobalNormalizationGrad : Op_name.t
  val batchSelfAdjointEig : Op_name.t
  val batchSelfAdjointEigV2 : Op_name.t
  val batchSvd : Op_name.t
  val batchToSpace : Op_name.t
  val batchToSpaceND : Op_name.t
  val besselI0e : Op_name.t
  val besselI1e : Op_name.t
  val betainc : Op_name.t
  val biasAdd : Op_name.t
  val biasAddGrad : Op_name.t
  val biasAddV1 : Op_name.t
  val bincount : Op_name.t
  val bitcast : Op_name.t
  val bitwiseAnd : Op_name.t
  val bitwiseOr : Op_name.t
  val bitwiseXor : Op_name.t
  val boostedTreesAggregateStats : Op_name.t
  val boostedTreesBucketize : Op_name.t
  val boostedTreesCalculateBestFeatureSplit : Op_name.t
  val boostedTreesMakeQuantileSummaries : Op_name.t
  val boostedTreesMakeStatsSummary : Op_name.t
  val broadcastArgs : Op_name.t
  val broadcastGradientArgs : Op_name.t
  val broadcastTo : Op_name.t
  val bucketize : Op_name.t
  val cTCGreedyDecoder : Op_name.t
  val cTCLoss : Op_name.t
  val cacheDataset : Op_name.t
  val cast : Op_name.t
  val ceil : Op_name.t
  val checkNumerics : Op_name.t
  val cholesky : Op_name.t
  val choleskyGrad : Op_name.t
  val clipByValue : Op_name.t
  val collectiveBcastRecv : Op_name.t
  val collectiveBcastSend : Op_name.t
  val collectiveGather : Op_name.t
  val collectivePermute : Op_name.t
  val collectiveReduce : Op_name.t
  val combinedNonMaxSuppression : Op_name.t
  val complex : Op_name.t
  val complexAbs : Op_name.t
  val computeAccidentalHits : Op_name.t
  val concat : Op_name.t
  val concatOffset : Op_name.t
  val concatV2 : Op_name.t
  val concatenateDataset : Op_name.t
  val conditionalAccumulator : Op_name.t
  val configureDistributedTPU : Op_name.t
  val conj : Op_name.t
  val conjugateTranspose : Op_name.t
  val consumeMutexLock : Op_name.t
  val controlTrigger : Op_name.t
  val conv2D : Op_name.t
  val conv2DBackpropFilter : Op_name.t
  val conv2DBackpropInput : Op_name.t
  val conv3D : Op_name.t
  val conv3DBackpropFilter : Op_name.t
  val conv3DBackpropFilterV2 : Op_name.t
  val conv3DBackpropInput : Op_name.t
  val conv3DBackpropInputV2 : Op_name.t
  val copy : Op_name.t
  val copyHost : Op_name.t
  val cos : Op_name.t
  val cosh : Op_name.t
  val countUpTo : Op_name.t
  val cropAndResize : Op_name.t
  val cropAndResizeGradBoxes : Op_name.t
  val cropAndResizeGradImage : Op_name.t
  val cross : Op_name.t
  val crossReplicaSum : Op_name.t
  val cudnnRNN : Op_name.t
  val cudnnRNNBackprop : Op_name.t
  val cudnnRNNCanonicalToParams : Op_name.t
  val cudnnRNNParamsSize : Op_name.t
  val cumprod : Op_name.t
  val cumsum : Op_name.t
  val dataFormatDimMap : Op_name.t
  val dataFormatVecPermute : Op_name.t
  val datasetToGraph : Op_name.t
  val debugGradientIdentity : Op_name.t
  val debugGradientRefIdentity : Op_name.t
  val debugIdentity : Op_name.t
  val debugNanCount : Op_name.t
  val debugNumericSummary : Op_name.t
  val decodeBase64 : Op_name.t
  val decodeCompressed : Op_name.t
  val decodeJSONExample : Op_name.t
  val decodePaddedRaw : Op_name.t
  val decodePng : Op_name.t
  val decodeRaw : Op_name.t
  val decodeWav : Op_name.t
  val deepCopy : Op_name.t
  val deleteSessionTensor : Op_name.t
  val denseToDenseSetOperation : Op_name.t
  val denseToSparseSetOperation : Op_name.t
  val depthToSpace : Op_name.t
  val depthwiseConv2dNative : Op_name.t
  val depthwiseConv2dNativeBackpropFilter : Op_name.t
  val depthwiseConv2dNativeBackpropInput : Op_name.t
  val dequantize : Op_name.t
  val deserializeManySparse : Op_name.t
  val deserializeSparse : Op_name.t
  val destroyTemporaryVariable : Op_name.t
  val diag : Op_name.t
  val diagPart : Op_name.t
  val digamma : Op_name.t
  val dilation2D : Op_name.t
  val dilation2DBackpropFilter : Op_name.t
  val dilation2DBackpropInput : Op_name.t
  val div : Op_name.t
  val divNoNan : Op_name.t
  val drawBoundingBoxes : Op_name.t
  val drawBoundingBoxesV2 : Op_name.t
  val dynamicPartition : Op_name.t
  val dynamicStitch : Op_name.t
  val editDistance : Op_name.t
  val elu : Op_name.t
  val eluGrad : Op_name.t
  val empty : Op_name.t
  val emptyTensorList : Op_name.t
  val encodeBase64 : Op_name.t
  val encodePng : Op_name.t
  val encodeWav : Op_name.t
  val enqueueTPUEmbeddingIntegerBatch : Op_name.t
  val enqueueTPUEmbeddingSparseBatch : Op_name.t
  val enqueueTPUEmbeddingSparseTensorBatch : Op_name.t
  val ensureShape : Op_name.t
  val enter : Op_name.t
  val equal : Op_name.t
  val erf : Op_name.t
  val erfc : Op_name.t
  val euclideanNorm : Op_name.t
  val exit : Op_name.t
  val exp : Op_name.t
  val expandDims : Op_name.t
  val experimentalAssertNextDataset : Op_name.t
  val experimentalAutoShardDataset : Op_name.t
  val experimentalBytesProducedStatsDataset : Op_name.t
  val experimentalChooseFastestDataset : Op_name.t
  val experimentalDatasetCardinality : Op_name.t
  val experimentalDatasetToTFRecord : Op_name.t
  val experimentalDenseToSparseBatchDataset : Op_name.t
  val experimentalDirectedInterleaveDataset : Op_name.t
  val experimentalIgnoreErrorsDataset : Op_name.t
  val experimentalLMDBDataset : Op_name.t
  val experimentalLatencyStatsDataset : Op_name.t
  val experimentalMatchingFilesDataset : Op_name.t
  val experimentalMaxIntraOpParallelismDataset : Op_name.t
  val experimentalNonSerializableDataset : Op_name.t
  val experimentalPrivateThreadPoolDataset : Op_name.t
  val experimentalRandomDataset : Op_name.t
  val experimentalRebatchDataset : Op_name.t
  val experimentalSleepDataset : Op_name.t
  val experimentalSlidingWindowDataset : Op_name.t
  val experimentalSqlDataset : Op_name.t
  val experimentalUnbatchDataset : Op_name.t
  val experimentalUniqueDataset : Op_name.t
  val expm1 : Op_name.t
  val extractGlimpse : Op_name.t
  val extractImagePatches : Op_name.t
  val extractJpegShape : Op_name.t
  val extractVolumePatches : Op_name.t
  val fFT : Op_name.t
  val fFT2D : Op_name.t
  val fFT3D : Op_name.t
  val fIFOQueue : Op_name.t
  val fact : Op_name.t
  val fakeParam : Op_name.t
  val fakeQuantWithMinMaxArgs : Op_name.t
  val fakeQuantWithMinMaxArgsGradient : Op_name.t
  val fakeQuantWithMinMaxVars : Op_name.t
  val fakeQuantWithMinMaxVarsGradient : Op_name.t
  val fakeQuantWithMinMaxVarsPerChannel : Op_name.t
  val fakeQuantWithMinMaxVarsPerChannelGradient : Op_name.t
  val fill : Op_name.t
  val filterByLastComponentDataset : Op_name.t
  val fixedLengthRecordDataset : Op_name.t
  val fixedLengthRecordDatasetV2 : Op_name.t
  val fixedLengthRecordReader : Op_name.t
  val fixedUnigramCandidateSampler : Op_name.t
  val floor : Op_name.t
  val floorDiv : Op_name.t
  val floorMod : Op_name.t
  val fractionalAvgPool : Op_name.t
  val fractionalAvgPoolGrad : Op_name.t
  val fractionalMaxPool : Op_name.t
  val fractionalMaxPoolGrad : Op_name.t
  val fusedBatchNorm : Op_name.t
  val fusedBatchNormGrad : Op_name.t
  val fusedBatchNormGradV2 : Op_name.t
  val fusedBatchNormV2 : Op_name.t
  val fusedPadConv2D : Op_name.t
  val fusedResizeAndPadConv2D : Op_name.t
  val gather : Op_name.t
  val gatherNd : Op_name.t
  val gatherV2 : Op_name.t
  val generateVocabRemapping : Op_name.t
  val getSessionHandle : Op_name.t
  val getSessionTensor : Op_name.t
  val greater : Op_name.t
  val greaterEqual : Op_name.t
  val guaranteeConst : Op_name.t
  val hSVToRGB : Op_name.t
  val hashTable : Op_name.t
  val histogramFixedWidth : Op_name.t
  val histogramSummary : Op_name.t
  val hostConst : Op_name.t
  val iFFT : Op_name.t
  val iFFT2D : Op_name.t
  val iFFT3D : Op_name.t
  val iRFFT : Op_name.t
  val iRFFT2D : Op_name.t
  val iRFFT3D : Op_name.t
  val identity : Op_name.t
  val identityReader : Op_name.t
  val igamma : Op_name.t
  val igammaGradA : Op_name.t
  val igammac : Op_name.t
  val imag : Op_name.t
  val imageSummary : Op_name.t
  val immutableConst : Op_name.t
  val inTopK : Op_name.t
  val inTopKV2 : Op_name.t
  val infeedDequeue : Op_name.t
  val infeedEnqueue : Op_name.t
  val infeedEnqueuePrelinearizedBuffer : Op_name.t
  val initializeTable : Op_name.t
  val initializeTableFromTextFile : Op_name.t
  val inplaceAdd : Op_name.t
  val inplaceSub : Op_name.t
  val inplaceUpdate : Op_name.t
  val inv : Op_name.t
  val invGrad : Op_name.t
  val invert : Op_name.t
  val invertPermutation : Op_name.t
  val isFinite : Op_name.t
  val isInf : Op_name.t
  val isNan : Op_name.t
  val isVariableInitialized : Op_name.t
  val kMC2ChainInitialization : Op_name.t
  val kmeansPlusPlusInitialization : Op_name.t
  val l2Loss : Op_name.t
  val lMDBReader : Op_name.t
  val lRN : Op_name.t
  val lRNGrad : Op_name.t
  val leakyRelu : Op_name.t
  val leakyReluGrad : Op_name.t
  val learnedUnigramCandidateSampler : Op_name.t
  val leftShift : Op_name.t
  val less : Op_name.t
  val lessEqual : Op_name.t
  val lgamma : Op_name.t
  val linSpace : Op_name.t
  val listDiff : Op_name.t
  val loadAndRemapMatrix : Op_name.t
  val loadTPUEmbeddingADAMParameters : Op_name.t
  val loadTPUEmbeddingADAMParametersGradAccumDebug : Op_name.t
  val loadTPUEmbeddingAdadeltaParameters : Op_name.t
  val loadTPUEmbeddingAdadeltaParametersGradAccumDebug : Op_name.t
  val loadTPUEmbeddingAdagradParameters : Op_name.t
  val loadTPUEmbeddingAdagradParametersGradAccumDebug : Op_name.t
  val loadTPUEmbeddingCenteredRMSPropParameters : Op_name.t
  val loadTPUEmbeddingFTRLParameters : Op_name.t
  val loadTPUEmbeddingFTRLParametersGradAccumDebug : Op_name.t
  val loadTPUEmbeddingMDLAdagradLightParameters : Op_name.t
  val loadTPUEmbeddingMomentumParameters : Op_name.t
  val loadTPUEmbeddingMomentumParametersGradAccumDebug : Op_name.t
  val loadTPUEmbeddingProximalAdagradParameters : Op_name.t
  val loadTPUEmbeddingProximalAdagradParametersGradAccumDebug : Op_name.t
  val loadTPUEmbeddingRMSPropParameters : Op_name.t
  val loadTPUEmbeddingRMSPropParametersGradAccumDebug : Op_name.t
  val loadTPUEmbeddingStochasticGradientDescentParameters : Op_name.t
  val log : Op_name.t
  val log1p : Op_name.t
  val logMatrixDeterminant : Op_name.t
  val logSoftmax : Op_name.t
  val logUniformCandidateSampler : Op_name.t
  val logicalAnd : Op_name.t
  val logicalNot : Op_name.t
  val logicalOr : Op_name.t
  val lookupTableExport : Op_name.t
  val lookupTableFind : Op_name.t
  val lookupTableImport : Op_name.t
  val lookupTableInsert : Op_name.t
  val lookupTableSize : Op_name.t
  val loopCond : Op_name.t
  val lowerBound : Op_name.t
  val lu : Op_name.t
  val mapClear : Op_name.t
  val mapIncompleteSize : Op_name.t
  val mapSize : Op_name.t
  val matMul : Op_name.t
  val matchingFiles : Op_name.t
  val matrixBandPart : Op_name.t
  val matrixDeterminant : Op_name.t
  val matrixDiag : Op_name.t
  val matrixDiagPart : Op_name.t
  val matrixExponential : Op_name.t
  val matrixInverse : Op_name.t
  val matrixLogarithm : Op_name.t
  val matrixSetDiag : Op_name.t
  val matrixSolve : Op_name.t
  val matrixSolveLs : Op_name.t
  val matrixSquareRoot : Op_name.t
  val matrixTriangularSolve : Op_name.t
  val max : Op_name.t
  val maxPool : Op_name.t
  val maxPool3D : Op_name.t
  val maxPool3DGrad : Op_name.t
  val maxPool3DGradGrad : Op_name.t
  val maxPoolGrad : Op_name.t
  val maxPoolGradGrad : Op_name.t
  val maxPoolGradGradV2 : Op_name.t
  val maxPoolGradGradWithArgmax : Op_name.t
  val maxPoolGradV2 : Op_name.t
  val maxPoolGradWithArgmax : Op_name.t
  val maxPoolV2 : Op_name.t
  val maxPoolWithArgmax : Op_name.t
  val maximum : Op_name.t
  val mean : Op_name.t
  val merge : Op_name.t
  val mergeSummary : Op_name.t
  val mergeV2Checkpoints : Op_name.t
  val mfcc : Op_name.t
  val min : Op_name.t
  val minimum : Op_name.t
  val mirrorPad : Op_name.t
  val mirrorPadGrad : Op_name.t
  val mod_ : Op_name.t
  val modelDataset : Op_name.t
  val mul : Op_name.t
  val mulNoNan : Op_name.t
  val multinomial : Op_name.t
  val mutableDenseHashTable : Op_name.t
  val mutableHashTable : Op_name.t
  val mutableHashTableOfTensors : Op_name.t
  val ncclAllReduce : Op_name.t
  val ncclBroadcast : Op_name.t
  val ncclReduce : Op_name.t
  val nearestNeighbors : Op_name.t
  val neg : Op_name.t
  val negTrain : Op_name.t
  val nextAfter : Op_name.t
  val nextIteration : Op_name.t
  val noOp : Op_name.t
  val nonDeterministicInts : Op_name.t
  val nonMaxSuppression : Op_name.t
  val nonMaxSuppressionV2 : Op_name.t
  val nonMaxSuppressionV3 : Op_name.t
  val nonMaxSuppressionV4 : Op_name.t
  val nonMaxSuppressionWithOverlaps : Op_name.t
  val notEqual : Op_name.t
  val nthElement : Op_name.t
  val oneHot : Op_name.t
  val onesLike : Op_name.t
  val optimizeDataset : Op_name.t
  val optionalHasValue : Op_name.t
  val optionalNone : Op_name.t
  val orderedMapClear : Op_name.t
  val orderedMapIncompleteSize : Op_name.t
  val orderedMapSize : Op_name.t
  val outfeedDequeue : Op_name.t
  val outfeedEnqueue : Op_name.t
  val pack : Op_name.t
  val pad : Op_name.t
  val padV2 : Op_name.t
  val paddingFIFOQueue : Op_name.t
  val parallelConcat : Op_name.t
  val parallelDynamicStitch : Op_name.t
  val parameterizedTruncatedNormal : Op_name.t
  val parseTensor : Op_name.t
  val placeholder : Op_name.t
  val placeholderV2 : Op_name.t
  val placeholderWithDefault : Op_name.t
  val polygamma : Op_name.t
  val pow : Op_name.t
  val prefetchDataset : Op_name.t
  val prelinearize : Op_name.t
  val preventGradient : Op_name.t
  val printV2 : Op_name.t
  val priorityQueue : Op_name.t
  val prod : Op_name.t
  val qr : Op_name.t
  val quantizeAndDequantize : Op_name.t
  val quantizeAndDequantizeV2 : Op_name.t
  val quantizeAndDequantizeV3 : Op_name.t
  val quantizeDownAndShrinkRange : Op_name.t
  val quantizeV2 : Op_name.t
  val quantizedAdd : Op_name.t
  val quantizedAvgPool : Op_name.t
  val quantizedBatchNormWithGlobalNormalization : Op_name.t
  val quantizedBiasAdd : Op_name.t
  val quantizedConcat : Op_name.t
  val quantizedConv2D : Op_name.t
  val quantizedConv2DAndRelu : Op_name.t
  val quantizedConv2DAndReluAndRequantize : Op_name.t
  val quantizedConv2DAndRequantize : Op_name.t
  val quantizedConv2DPerChannel : Op_name.t
  val quantizedConv2DWithBias : Op_name.t
  val quantizedConv2DWithBiasAndRelu : Op_name.t
  val quantizedConv2DWithBiasAndReluAndRequantize : Op_name.t
  val quantizedConv2DWithBiasAndRequantize : Op_name.t
  val quantizedConv2DWithBiasSignedSumAndReluAndRequantize : Op_name.t
  val quantizedConv2DWithBiasSumAndRelu : Op_name.t
  val quantizedConv2DWithBiasSumAndReluAndRequantize : Op_name.t
  val quantizedDepthwiseConv2D : Op_name.t
  val quantizedDepthwiseConv2DWithBias : Op_name.t
  val quantizedDepthwiseConv2DWithBiasAndRelu : Op_name.t
  val quantizedDepthwiseConv2DWithBiasAndReluAndRequantize : Op_name.t
  val quantizedInstanceNorm : Op_name.t
  val quantizedMatMul : Op_name.t
  val quantizedMaxPool : Op_name.t
  val quantizedMul : Op_name.t
  val quantizedRelu : Op_name.t
  val quantizedRelu6 : Op_name.t
  val quantizedReluX : Op_name.t
  val quantizedReshape : Op_name.t
  val quantizedResizeBilinear : Op_name.t
  val queueClose : Op_name.t
  val queueIsClosed : Op_name.t
  val queueSize : Op_name.t
  val rFFT : Op_name.t
  val rFFT2D : Op_name.t
  val rFFT3D : Op_name.t
  val rGBToHSV : Op_name.t
  val raggedRange : Op_name.t
  val raggedTensorToSparse : Op_name.t
  val randomCrop : Op_name.t
  val randomGamma : Op_name.t
  val randomGammaGrad : Op_name.t
  val randomPoisson : Op_name.t
  val randomPoissonV2 : Op_name.t
  val randomShuffle : Op_name.t
  val randomShuffleQueue : Op_name.t
  val randomStandardNormal : Op_name.t
  val randomUniform : Op_name.t
  val randomUniformInt : Op_name.t
  val range : Op_name.t
  val rangeDataset : Op_name.t
  val rank : Op_name.t
  val readFile : Op_name.t
  val readerNumRecordsProduced : Op_name.t
  val readerNumWorkUnitsCompleted : Op_name.t
  val readerRead : Op_name.t
  val readerReadUpTo : Op_name.t
  val readerReset : Op_name.t
  val readerRestoreState : Op_name.t
  val readerSerializeState : Op_name.t
  val real : Op_name.t
  val realDiv : Op_name.t
  val reciprocal : Op_name.t
  val reciprocalGrad : Op_name.t
  val recordInput : Op_name.t
  val recvTPUEmbeddingActivations : Op_name.t
  val reduceJoin : Op_name.t
  val refEnter : Op_name.t
  val refExit : Op_name.t
  val refIdentity : Op_name.t
  val refMerge : Op_name.t
  val refNextIteration : Op_name.t
  val refSelect : Op_name.t
  val refSwitch : Op_name.t
  val regexFullMatch : Op_name.t
  val regexReplace : Op_name.t
  val relu : Op_name.t
  val relu6 : Op_name.t
  val relu6Grad : Op_name.t
  val reluGrad : Op_name.t
  val repeatDataset : Op_name.t
  val requantizationRange : Op_name.t
  val requantizationRangePerChannel : Op_name.t
  val requantize : Op_name.t
  val requantizePerChannel : Op_name.t
  val reshape : Op_name.t
  val resizeArea : Op_name.t
  val resizeBicubic : Op_name.t
  val resizeBicubicGrad : Op_name.t
  val resizeBilinear : Op_name.t
  val resizeBilinearGrad : Op_name.t
  val resizeNearestNeighbor : Op_name.t
  val resizeNearestNeighborGrad : Op_name.t
  val restore : Op_name.t
  val restoreSlice : Op_name.t
  val retrieveTPUEmbeddingADAMParameters : Op_name.t
  val retrieveTPUEmbeddingADAMParametersGradAccumDebug : Op_name.t
  val retrieveTPUEmbeddingAdadeltaParameters : Op_name.t
  val retrieveTPUEmbeddingAdadeltaParametersGradAccumDebug : Op_name.t
  val retrieveTPUEmbeddingAdagradParameters : Op_name.t
  val retrieveTPUEmbeddingAdagradParametersGradAccumDebug : Op_name.t
  val retrieveTPUEmbeddingCenteredRMSPropParameters : Op_name.t
  val retrieveTPUEmbeddingFTRLParameters : Op_name.t
  val retrieveTPUEmbeddingFTRLParametersGradAccumDebug : Op_name.t
  val retrieveTPUEmbeddingMDLAdagradLightParameters : Op_name.t
  val retrieveTPUEmbeddingMomentumParameters : Op_name.t
  val retrieveTPUEmbeddingMomentumParametersGradAccumDebug : Op_name.t
  val retrieveTPUEmbeddingProximalAdagradParameters : Op_name.t
  val retrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug : Op_name.t
  val retrieveTPUEmbeddingRMSPropParameters : Op_name.t
  val retrieveTPUEmbeddingRMSPropParametersGradAccumDebug : Op_name.t
  val retrieveTPUEmbeddingStochasticGradientDescentParameters : Op_name.t
  val reverse : Op_name.t
  val reverseSequence : Op_name.t
  val reverseV2 : Op_name.t
  val rightShift : Op_name.t
  val rint : Op_name.t
  val roll : Op_name.t
  val round : Op_name.t
  val rpc : Op_name.t
  val rsqrt : Op_name.t
  val rsqrtGrad : Op_name.t
  val sampleDistortedBoundingBox : Op_name.t
  val sampleDistortedBoundingBoxV2 : Op_name.t
  val samplingDataset : Op_name.t
  val scalarSummary : Op_name.t
  val scaleAndTranslate : Op_name.t
  val scaleAndTranslateGrad : Op_name.t
  val scatterAdd : Op_name.t
  val scatterDiv : Op_name.t
  val scatterMax : Op_name.t
  val scatterMin : Op_name.t
  val scatterMul : Op_name.t
  val scatterNd : Op_name.t
  val scatterNdAdd : Op_name.t
  val scatterNdNonAliasingAdd : Op_name.t
  val scatterNdSub : Op_name.t
  val scatterNdUpdate : Op_name.t
  val scatterSub : Op_name.t
  val scatterUpdate : Op_name.t
  val sdcaFprint : Op_name.t
  val sdcaShrinkL1 : Op_name.t
  val segmentMax : Op_name.t
  val segmentMean : Op_name.t
  val segmentMin : Op_name.t
  val segmentProd : Op_name.t
  val segmentSum : Op_name.t
  val select : Op_name.t
  val selfAdjointEig : Op_name.t
  val selfAdjointEigV2 : Op_name.t
  val selu : Op_name.t
  val seluGrad : Op_name.t
  val sendTPUEmbeddingGradients : Op_name.t
  val serializeManySparse : Op_name.t
  val serializeSparse : Op_name.t
  val serializeTensor : Op_name.t
  val setSize : Op_name.t
  val shape : Op_name.t
  val shapeN : Op_name.t
  val shardDataset : Op_name.t
  val shardedFilename : Op_name.t
  val shardedFilespec : Op_name.t
  val shuffleAndRepeatDataset : Op_name.t
  val shuffleDataset : Op_name.t
  val shutdownDistributedTPU : Op_name.t
  val sigmoid : Op_name.t
  val sigmoidGrad : Op_name.t
  val sign : Op_name.t
  val sin : Op_name.t
  val sinh : Op_name.t
  val size : Op_name.t
  val skipDataset : Op_name.t
  val skipgram : Op_name.t
  val slice : Op_name.t
  val snapshot : Op_name.t
  val softmax : Op_name.t
  val softmaxCrossEntropyWithLogits : Op_name.t
  val softplus : Op_name.t
  val softplusGrad : Op_name.t
  val softsign : Op_name.t
  val softsignGrad : Op_name.t
  val spaceToBatch : Op_name.t
  val spaceToBatchND : Op_name.t
  val spaceToDepth : Op_name.t
  val sparseAccumulatorApplyGradient : Op_name.t
  val sparseAccumulatorTakeGradient : Op_name.t
  val sparseAdd : Op_name.t
  val sparseAddGrad : Op_name.t
  val sparseApplyAdadelta : Op_name.t
  val sparseApplyAdagrad : Op_name.t
  val sparseApplyAdagradDA : Op_name.t
  val sparseApplyCenteredRMSProp : Op_name.t
  val sparseApplyFtrl : Op_name.t
  val sparseApplyFtrlV2 : Op_name.t
  val sparseApplyMomentum : Op_name.t
  val sparseApplyProximalAdagrad : Op_name.t
  val sparseApplyProximalGradientDescent : Op_name.t
  val sparseApplyRMSProp : Op_name.t
  val sparseConcat : Op_name.t
  val sparseConditionalAccumulator : Op_name.t
  val sparseDenseCwiseAdd : Op_name.t
  val sparseDenseCwiseDiv : Op_name.t
  val sparseDenseCwiseMul : Op_name.t
  val sparseFillEmptyRows : Op_name.t
  val sparseFillEmptyRowsGrad : Op_name.t
  val sparseMatMul : Op_name.t
  val sparseReduceMax : Op_name.t
  val sparseReduceMaxSparse : Op_name.t
  val sparseReduceSum : Op_name.t
  val sparseReduceSumSparse : Op_name.t
  val sparseReorder : Op_name.t
  val sparseReshape : Op_name.t
  val sparseSegmentMean : Op_name.t
  val sparseSegmentMeanGrad : Op_name.t
  val sparseSegmentMeanWithNumSegments : Op_name.t
  val sparseSegmentSqrtN : Op_name.t
  val sparseSegmentSqrtNGrad : Op_name.t
  val sparseSegmentSqrtNWithNumSegments : Op_name.t
  val sparseSegmentSum : Op_name.t
  val sparseSegmentSumWithNumSegments : Op_name.t
  val sparseSlice : Op_name.t
  val sparseSliceGrad : Op_name.t
  val sparseSoftmax : Op_name.t
  val sparseSoftmaxCrossEntropyWithLogits : Op_name.t
  val sparseSparseMaximum : Op_name.t
  val sparseSparseMinimum : Op_name.t
  val sparseTensorDenseAdd : Op_name.t
  val sparseTensorDenseMatMul : Op_name.t
  val sparseTensorSliceDataset : Op_name.t
  val sparseToDense : Op_name.t
  val sparseToSparseSetOperation : Op_name.t
  val split : Op_name.t
  val splitV : Op_name.t
  val sqrt : Op_name.t
  val sqrtGrad : Op_name.t
  val square : Op_name.t
  val squaredDifference : Op_name.t
  val squeeze : Op_name.t
  val stack : Op_name.t
  val stackClose : Op_name.t
  val stackPop : Op_name.t
  val stackPush : Op_name.t
  val stageClear : Op_name.t
  val stageSize : Op_name.t
  val statelessMultinomial : Op_name.t
  val statelessRandomNormal : Op_name.t
  val statelessRandomUniform : Op_name.t
  val statelessRandomUniformInt : Op_name.t
  val statelessTruncatedNormal : Op_name.t
  val staticRegexFullMatch : Op_name.t
  val staticRegexReplace : Op_name.t
  val stopGradient : Op_name.t
  val stridedSlice : Op_name.t
  val stridedSliceAssign : Op_name.t
  val stridedSliceGrad : Op_name.t
  val stringJoin : Op_name.t
  val stringLength : Op_name.t
  val stringSplit : Op_name.t
  val stringSplitV2 : Op_name.t
  val stringStrip : Op_name.t
  val stringToHashBucket : Op_name.t
  val stringToHashBucketFast : Op_name.t
  val stringToHashBucketStrong : Op_name.t
  val stringToNumber : Op_name.t
  val sub : Op_name.t
  val substr : Op_name.t
  val sum : Op_name.t
  val svd : Op_name.t
  val switch : Op_name.t
  val tFRecordDataset : Op_name.t
  val tFRecordReader : Op_name.t
  val tPUCompilationResult : Op_name.t
  val tPUEmbeddingActivations : Op_name.t
  val tPUOrdinalSelector : Op_name.t
  val tPUReplicateMetadata : Op_name.t
  val tPUReplicatedInput : Op_name.t
  val tPUReplicatedOutput : Op_name.t
  val takeDataset : Op_name.t
  val takeManySparseFromTensorsMap : Op_name.t
  val tan : Op_name.t
  val tanh : Op_name.t
  val tanhGrad : Op_name.t
  val temporaryVariable : Op_name.t
  val tensorArray : Op_name.t
  val tensorArrayClose : Op_name.t
  val tensorArrayCloseV2 : Op_name.t
  val tensorArrayConcat : Op_name.t
  val tensorArrayConcatV2 : Op_name.t
  val tensorArrayGather : Op_name.t
  val tensorArrayGatherV2 : Op_name.t
  val tensorArrayGrad : Op_name.t
  val tensorArrayGradV2 : Op_name.t
  val tensorArrayPack : Op_name.t
  val tensorArrayRead : Op_name.t
  val tensorArrayReadV2 : Op_name.t
  val tensorArrayScatter : Op_name.t
  val tensorArrayScatterV2 : Op_name.t
  val tensorArraySize : Op_name.t
  val tensorArraySizeV2 : Op_name.t
  val tensorArraySplit : Op_name.t
  val tensorArraySplitV2 : Op_name.t
  val tensorArrayUnpack : Op_name.t
  val tensorArrayV2 : Op_name.t
  val tensorArrayWrite : Op_name.t
  val tensorArrayWriteV2 : Op_name.t
  val tensorListConcat : Op_name.t
  val tensorListConcatLists : Op_name.t
  val tensorListConcatV2 : Op_name.t
  val tensorListElementShape : Op_name.t
  val tensorListFromTensor : Op_name.t
  val tensorListGather : Op_name.t
  val tensorListGetItem : Op_name.t
  val tensorListLength : Op_name.t
  val tensorListPopBack : Op_name.t
  val tensorListPushBack : Op_name.t
  val tensorListPushBackBatch : Op_name.t
  val tensorListReserve : Op_name.t
  val tensorListResize : Op_name.t
  val tensorListScatter : Op_name.t
  val tensorListScatterIntoExistingList : Op_name.t
  val tensorListScatterV2 : Op_name.t
  val tensorListSetItem : Op_name.t
  val tensorListSplit : Op_name.t
  val tensorListStack : Op_name.t
  val tensorScatterAdd : Op_name.t
  val tensorScatterSub : Op_name.t
  val tensorScatterUpdate : Op_name.t
  val tensorStridedSliceUpdate : Op_name.t
  val tensorSummary : Op_name.t
  val tensorSummaryV2 : Op_name.t
  val textLineDataset : Op_name.t
  val textLineReader : Op_name.t
  val threadUnsafeUnigramCandidateSampler : Op_name.t
  val tile : Op_name.t
  val tileGrad : Op_name.t
  val timestamp : Op_name.t
  val topK : Op_name.t
  val topKV2 : Op_name.t
  val transpose : Op_name.t
  val tridiagonalSolve : Op_name.t
  val truncateDiv : Op_name.t
  val truncateMod : Op_name.t
  val truncatedNormal : Op_name.t
  val tryRpc : Op_name.t
  val unbatch : Op_name.t
  val unbatchGrad : Op_name.t
  val unicodeDecode : Op_name.t
  val unicodeDecodeWithOffsets : Op_name.t
  val unicodeEncode : Op_name.t
  val unicodeScript : Op_name.t
  val unicodeTranscode : Op_name.t
  val uniformCandidateSampler : Op_name.t
  val unique : Op_name.t
  val uniqueV2 : Op_name.t
  val uniqueWithCounts : Op_name.t
  val uniqueWithCountsV2 : Op_name.t
  val unpack : Op_name.t
  val unravelIndex : Op_name.t
  val unsortedSegmentMax : Op_name.t
  val unsortedSegmentMin : Op_name.t
  val unsortedSegmentProd : Op_name.t
  val unsortedSegmentSum : Op_name.t
  val unwrapDatasetVariant : Op_name.t
  val upperBound : Op_name.t
  val variable : Op_name.t
  val variableV2 : Op_name.t
  val where : Op_name.t
  val wholeFileReader : Op_name.t
  val windowDataset : Op_name.t
  val workerHeartbeat : Op_name.t
  val wrapDatasetVariant : Op_name.t
  val writeFile : Op_name.t
  val xdivy : Op_name.t
  val xlogy : Op_name.t
  val zerosLike : Op_name.t
  val zeta : Op_name.t
  val zipDataset : Op_name.t
end

val abort
  :  ?name:string
  -> ?error_msg:string
  -> ?exit_without_error:bool
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

val abs
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val accumulateNV2
  :  ?name:string
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val accumulatorApplyGradient
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 'dtype) t
  -> [ `unit ] t

val accumulatorNumAccumulated
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t

val accumulatorSetGlobalStep
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `unit ] t

val accumulatorTakeGradient
  :  ?name:string
  -> type_:([< `float | `double | `int32 | `complex64 | `int64 ] as 'dtype) Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 'dtype) t

val acos
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val acosh
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val add
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t

val addManySparseToTensorsMap
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t

val addN
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 | `variant ] as 't) t list
  -> ([< `float | `double | `int32 | `complex64 | `int64 | `variant ] as 't) t

val addSparseToTensorsMap
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t

val addV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val adjustContrast
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

val adjustContrastv2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> [ `float ] t
  -> ([< `float ] as 't) t

val adjustHue
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> [ `float ] t
  -> ([< `float ] as 't) t

val adjustSaturation
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> [ `float ] t
  -> ([< `float ] as 't) t

val all
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `bool ] t

val allCandidateSampler
  :  ?name:string
  -> num_true:int
  -> num_sampled:int
  -> unique:bool
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

val allToAll
  :  ?name:string
  -> concat_dimension:int
  -> split_dimension:int
  -> split_count:int
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t

val angle
  :  ?name:string
  -> type_:([< `float | `double ] as 'tout) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 't) t
  -> ([< `float | `double ] as 'tout) t

val any
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `bool ] t

val applyAdaMax
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyAdadelta
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ?update_slots:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyAdagradDA
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyAdam
  :  ?name:string
  -> ?use_locking:bool
  -> ?use_nesterov:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyAddSign
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyCenteredRMSProp
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyFtrl
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyFtrlV2
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyGradientDescent
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyMomentum
  :  ?name:string
  -> ?use_locking:bool
  -> ?use_nesterov:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyPowerSign
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyProximalAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyProximalGradientDescent
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val applyRMSProp
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val approximateEqual
  :  ?name:string
  -> ?tolerance:float
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `bool ] t

val argMax
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'output_type) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `int32 | `int64 ] as 'output_type) t

val argMin
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'output_type) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `int32 | `int64 ] as 'output_type) t

val asString
  :  ?name:string
  -> ?precision:int
  -> ?scientific:bool
  -> ?shortest:bool
  -> ?width:int
  -> ?fill:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `complex64 | `float | `double | `bool ] as 't) t
  -> [ `string ] t

val asin
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val asinh
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val assign
  :  ?name:string
  -> ?validate_shape:bool
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t
  -> 't t

val assignAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val assignSub
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val atan
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val atan2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val atanh
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val audioSpectrogram
  :  ?name:string
  -> window_size:int
  -> stride:int
  -> ?magnitude_squared:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t

val audioSummary
  :  ?name:string
  -> sample_rate:float
  -> ?max_outputs:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> [ `string ] t

val audioSummaryV2
  :  ?name:string
  -> ?max_outputs:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `string ] t

val avgPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val avgPool3D
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val avgPool3DGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val avgPoolGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val barrier
  :  ?name:string
  -> component_types:Type.p list
  -> ?shapes:Dim.t list list
  -> ?capacity:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val barrierClose
  :  ?name:string
  -> ?cancel_pending_enqueues:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

val barrierIncompleteSize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t

val barrierInsertMany
  :  ?name:string
  -> component_index:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> 't t
  -> [ `unit ] t

val barrierReadySize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t

val batchCholesky
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

val batchCholeskyGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val batchDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val batchDatasetV2
  :  ?name:string
  -> ?parallel_copy:bool
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `bool ] t
  -> [ `variant ] t

val batchFFT
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

val batchFFT2D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

val batchFFT3D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

val batchIFFT
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

val batchIFFT2D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

val batchIFFT3D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

val batchMatMul
  :  ?name:string
  -> ?adj_x:bool
  -> ?adj_y:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val batchMatMulV2
  :  ?name:string
  -> ?adj_x:bool
  -> ?adj_y:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val batchMatrixBandPart
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> 't t

val batchMatrixDeterminant
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val batchMatrixDiag
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val batchMatrixDiagPart
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val batchMatrixInverse
  :  ?name:string
  -> ?adjoint:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

val batchMatrixSetDiag
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t
  -> 't t

val batchMatrixSolve
  :  ?name:string
  -> ?adjoint:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

val batchMatrixSolveLs
  :  ?name:string
  -> ?fast:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> [ `double ] t
  -> ([< `double | `float ] as 't) t

val batchMatrixTriangularSolve
  :  ?name:string
  -> ?lower:bool
  -> ?adjoint:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

val batchNormWithGlobalNormalization
  :  ?name:string
  -> variance_epsilon:float
  -> scale_after_normalization:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val batchNormWithGlobalNormalizationGrad
  :  ?name:string
  -> variance_epsilon:float
  -> scale_after_normalization:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t * ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t * ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t * ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t * ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val batchSelfAdjointEig
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

val batchSelfAdjointEigV2
  :  ?name:string
  -> ?compute_v:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t * ([< `double | `float ] as 't) t

val batchSvd
  :  ?name:string
  -> ?compute_uv:bool
  -> ?full_matrices:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t * ([< `double | `float | `complex64 ] as 't) t * ([< `double | `float | `complex64 ] as 't) t

val batchToSpace
  :  ?name:string
  -> block_size:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> 't t

val batchToSpaceND
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tblock_shape) t
  -> ([< `int32 | `int64 ] as 'tcrops) t
  -> 't t

val besselI0e
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val besselI1e
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val betainc
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val biasAdd
  :  ?name:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val biasAddGrad
  :  ?name:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val biasAddV1
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val bincount
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

val bitcast
  :  ?name:string
  -> type_:([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) t

val bitwiseAnd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val bitwiseOr
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val bitwiseXor
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val boostedTreesAggregateStats
  :  ?name:string
  -> max_splits:int
  -> num_buckets:int
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `float ] t

val boostedTreesBucketize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t list
  -> [ `float ] t list
  -> [ `int32 ] t list

val boostedTreesCalculateBestFeatureSplit
  :  ?name:string
  -> logits_dimension:int
  -> ?split_type:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t * [ `float ] t * [ `int32 ] t * [ `int32 ] t * [ `float ] t * [ `float ] t * [ `string ] t

val boostedTreesMakeQuantileSummaries
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t list

val boostedTreesMakeStatsSummary
  :  ?name:string
  -> max_splits:int
  -> num_buckets:int
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t list
  -> [ `float ] t

val broadcastArgs
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val broadcastGradientArgs
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t * ([< `int32 | `int64 ] as 't) t

val broadcastTo
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> 't t

val bucketize
  :  ?name:string
  -> boundaries:float list
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t

val cTCGreedyDecoder
  :  ?name:string
  -> ?merge_repeated:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `int64 ] t * [ `int64 ] t * [ `int64 ] t * [ `float ] t

val cTCLoss
  :  ?name:string
  -> ?preprocess_collapse_repeated:bool
  -> ?ctc_merge_repeated:bool
  -> ?ignore_longer_outputs_than_inputs:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `float ] t * [ `float ] t

val cacheDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `string ] t
  -> [ `variant ] t

val cast
  :  ?name:string
  -> type_:'dstT Type.t
  -> ?truncate:bool
  -> ?control_inputs:Node.p list
  -> 'srcT t
  -> 'dstT t

val ceil
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val checkNumerics
  :  ?name:string
  -> message:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val cholesky
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t

val choleskyGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val clipByValue
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val collectiveBcastRecv
  :  ?name:string
  -> type_:([< `float | `double | `int32 | `int64 ] as 't) Type.t
  -> group_size:int
  -> group_key:int
  -> instance_key:int
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> unit
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val collectiveBcastSend
  :  ?name:string
  -> group_size:int
  -> group_key:int
  -> instance_key:int
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val collectiveGather
  :  ?name:string
  -> group_size:int
  -> group_key:int
  -> instance_key:int
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val collectivePermute
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val collectiveReduce
  :  ?name:string
  -> group_size:int
  -> group_key:int
  -> instance_key:int
  -> merge_op:string
  -> final_op:string
  -> subdiv_offsets:int list
  -> ?wait_for:int list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val combinedNonMaxSuppression
  :  ?name:string
  -> ?pad_per_class:bool
  -> ?clip_boxes:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t * [ `float ] t * [ `float ] t * [ `int32 ] t

val complex
  :  ?name:string
  -> type_:([< `complex64 ] as 'tout) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `complex64 ] as 'tout) t

val complexAbs
  :  ?name:string
  -> type_:([< `float | `double ] as 'tout) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 't) t
  -> ([< `float | `double ] as 'tout) t

val computeAccidentalHits
  :  ?name:string
  -> num_true:int
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int32 ] t * [ `int64 ] t * [ `float ] t

val concat
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 't t list
  -> 't t

val concatOffset
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `int32 ] t list
  -> [ `int32 ] t list

val concatV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t list
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> 't t

val concatenateDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t
  -> [ `variant ] t

val conditionalAccumulator
  :  ?name:string
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> ?reduction_type:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val configureDistributedTPU
  :  ?name:string
  -> ?embedding_config:string
  -> ?tpu_embedding_config:string
  -> ?is_global_init:bool
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val conj
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `complex64 | `variant ] as 't) t
  -> ([< `complex64 | `variant ] as 't) t

val conjugateTranspose
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tperm) t
  -> 't t

val consumeMutexLock
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `unit ] t

val controlTrigger
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

val conv2D
  :  ?name:string
  -> strides:int list
  -> ?use_cudnn_on_gpu:bool
  -> padding:string
  -> ?explicit_paddings:int list
  -> ?data_format:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val conv2DBackpropFilter
  :  ?name:string
  -> strides:int list
  -> ?use_cudnn_on_gpu:bool
  -> padding:string
  -> ?explicit_paddings:int list
  -> ?data_format:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val conv2DBackpropInput
  :  ?name:string
  -> strides:int list
  -> ?use_cudnn_on_gpu:bool
  -> padding:string
  -> ?explicit_paddings:int list
  -> ?data_format:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val conv3D
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val conv3DBackpropFilter
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val conv3DBackpropFilterV2
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val conv3DBackpropInput
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val conv3DBackpropInputV2
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tshape) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Copy Op. *)
(* Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
device on which the tensor is allocated.
N.B.: If the all downstream attached debug ops are disabled given the current
gRPC gating status, the output will simply forward the input tensor without
deep-copying. See the documentation of Debug* ops for more details.

Unlike the CopyHost Op, this op does not have HostMemory constraint on its
input or output. *)
val copy
  :  ?name:string
  -> ?tensor_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Copy Host Op. *)
(* Performs CPU-to-CPU deep-copying of tensor.
N.B.: If the all downstream attached debug ops are disabled given the current
gRPC gating status, the output will simply forward the input tensor without
deep-copying. See the documentation of Debug* ops for more details.

Unlike the Copy Op, this op has HostMemory constraint on its input or output. *)
val copyHost
  :  ?name:string
  -> ?tensor_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val cos
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val cosh
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val countUpTo
  :  ?name:string
  -> limit:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val cropAndResize
  :  ?name:string
  -> ?method_:string
  -> ?extrapolation_value:float
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `float ] t

val cropAndResizeGradBoxes
  :  ?name:string
  -> ?method_:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `float ] t

val cropAndResizeGradImage
  :  ?name:string
  -> type_:([< `float | `double ] as 't) Type.t
  -> ?method_:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

val cross
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val crossReplicaSum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `int32 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `int32 ] as 't) t

val cudnnRNN
  :  ?name:string
  -> ?rnn_mode:string
  -> ?input_mode:string
  -> ?direction:string
  -> ?dropout:float
  -> ?seed:int
  -> ?seed2:int
  -> ?is_training:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t

val cudnnRNNBackprop
  :  ?name:string
  -> ?rnn_mode:string
  -> ?input_mode:string
  -> ?direction:string
  -> ?dropout:float
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t

val cudnnRNNCanonicalToParams
  :  ?name:string
  -> ?rnn_mode:string
  -> ?input_mode:string
  -> ?direction:string
  -> ?dropout:float
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t list
  -> ([< `float | `double ] as 't) t list
  -> ([< `float | `double ] as 't) t

val cudnnRNNParamsSize
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 's) Type.t
  -> ?rnn_mode:string
  -> ?input_mode:string
  -> ?direction:string
  -> ?dropout:float
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 ] as 's) t

val cumprod
  :  ?name:string
  -> ?exclusive:bool
  -> ?reverse:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val cumsum
  :  ?name:string
  -> ?exclusive:bool
  -> ?reverse:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val dataFormatDimMap
  :  ?name:string
  -> ?src_format:string
  -> ?dst_format:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val dataFormatVecPermute
  :  ?name:string
  -> ?src_format:string
  -> ?dst_format:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val datasetToGraph
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `string ] t

val debugGradientIdentity
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val debugGradientRefIdentity
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Debug Identity Op. *)
(* Provides an identity mapping of the non-Ref type input tensor for debugging. *)
val debugIdentity
  :  ?name:string
  -> ?device_name:string
  -> ?tensor_name:string
  -> ?gated_grpc:bool
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Debug NaN Value Counter Op *)
(* Counts number of NaNs in the input tensor, for debugging. *)
val debugNanCount
  :  ?name:string
  -> ?device_name:string
  -> ?tensor_name:string
  -> ?gated_grpc:bool
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int64 ] t

(* Debug Numeric Summary Op. *)
(* Provide a basic summary of numeric value types, range and distribution. *)
val debugNumericSummary
  :  ?name:string
  -> ?device_name:string
  -> ?tensor_name:string
  -> ?lower_bound:float
  -> ?upper_bound:float
  -> ?mute_if_healthy:bool
  -> ?gated_grpc:bool
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `double ] t

val decodeBase64
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val decodeCompressed
  :  ?name:string
  -> ?compression_type:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val decodeJSONExample
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val decodePaddedRaw
  :  ?name:string
  -> type_:([< `float | `double | `int32 | `int64 ] as 'out_type) Type.t
  -> ?little_endian:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 'out_type) t

val decodePng
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?channels:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'dtype t

val decodeRaw
  :  ?name:string
  -> type_:([< `float | `double | `int32 | `int64 | `complex64 | `bool ] as 'out_type) Type.t
  -> ?little_endian:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `bool ] as 'out_type) t

val decodeWav
  :  ?name:string
  -> ?desired_channels:int
  -> ?desired_samples:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t * [ `int32 ] t

val deepCopy
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val deleteSessionTensor
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

val denseToDenseSetOperation
  :  ?name:string
  -> set_operation:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> [ `int64 ] t * ([< `int32 | `int64 | `string ] as 't) t * [ `int64 ] t

val denseToSparseSetOperation
  :  ?name:string
  -> set_operation:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> [ `int64 ] t
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t * ([< `int32 | `int64 | `string ] as 't) t * [ `int64 ] t

val depthToSpace
  :  ?name:string
  -> block_size:int
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val depthwiseConv2dNative
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val depthwiseConv2dNativeBackpropFilter
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val depthwiseConv2dNativeBackpropInput
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val dequantize
  :  ?name:string
  -> ?mode:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

val deserializeManySparse
  :  ?name:string
  -> type_1:'dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t * 'dtype t * [ `int64 ] t

val deserializeSparse
  :  ?name:string
  -> type_1:'dtype Type.t
  -> ?control_inputs:Node.p list
  -> ([< `string | `variant ] as 'tserialized) t
  -> [ `int64 ] t * 'dtype t * [ `int64 ] t

val destroyTemporaryVariable
  :  ?name:string
  -> var_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val diag
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val diagPart
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val digamma
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val dilation2D
  :  ?name:string
  -> strides:int list
  -> rates:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val dilation2DBackpropFilter
  :  ?name:string
  -> strides:int list
  -> rates:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val dilation2DBackpropInput
  :  ?name:string
  -> strides:int list
  -> rates:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val div
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val divNoNan
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val drawBoundingBoxes
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> [ `float ] t
  -> ([< `float ] as 't) t

val drawBoundingBoxesV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> [ `float ] t
  -> [ `float ] t
  -> ([< `float ] as 't) t

val dynamicPartition
  :  ?name:string
  -> num_partitions:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int32 ] t
  -> 't t list

val dynamicStitch
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t list
  -> 't t list
  -> 't t

val editDistance
  :  ?name:string
  -> ?normalize:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `float ] t

val elu
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val eluGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val empty
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?init:bool
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 'dtype t

val emptyTensorList
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'shape_type) t
  -> [ `int32 ] t
  -> [ `variant ] t

val encodeBase64
  :  ?name:string
  -> ?pad:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val encodePng
  :  ?name:string
  -> ?compression:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `string ] t

val encodeWav
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `string ] t

val enqueueTPUEmbeddingIntegerBatch
  :  ?name:string
  -> ?device_ordinal:int
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t list
  -> [ `string ] t
  -> [ `unit ] t

val enqueueTPUEmbeddingSparseBatch
  :  ?name:string
  -> ?device_ordinal:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't1) t list
  -> ([< `int32 | `int64 ] as 't2) t list
  -> ([< `float | `double ] as 't3) t list
  -> [ `string ] t
  -> [ `unit ] t

val enqueueTPUEmbeddingSparseTensorBatch
  :  ?name:string
  -> ?device_ordinal:int
  -> table_ids:int list
  -> ?max_sequence_lengths:int list
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't1) t list
  -> ([< `int32 | `int64 ] as 't2) t list
  -> ([< `float | `double ] as 't3) t list
  -> [ `string ] t
  -> [ `unit ] t

val ensureShape
  :  ?name:string
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val enter
  :  ?name:string
  -> frame_name:string
  -> ?is_constant:bool
  -> ?parallel_iterations:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val equal
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> [ `bool ] t

val erf
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val erfc
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val euclideanNorm
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val exit
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val exp
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val expandDims
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tdim) t
  -> 't t

val experimentalAssertNextDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `string ] t
  -> [ `variant ] t

val experimentalAutoShardDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val experimentalBytesProducedStatsDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `string ] t
  -> [ `variant ] t

val experimentalChooseFastestDataset
  :  ?name:string
  -> num_experiments:int
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t list
  -> [ `variant ] t

val experimentalDatasetCardinality
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t

val experimentalDatasetToTFRecord
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `string ] t
  -> [ `string ] t
  -> [ `unit ] t

val experimentalDenseToSparseBatchDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val experimentalDirectedInterleaveDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t list
  -> [ `variant ] t

val experimentalIgnoreErrorsDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t

val experimentalLMDBDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `variant ] t

val experimentalLatencyStatsDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `string ] t
  -> [ `variant ] t

val experimentalMatchingFilesDataset
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `variant ] t

val experimentalMaxIntraOpParallelismDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val experimentalNonSerializableDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t

val experimentalPrivateThreadPoolDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val experimentalRandomDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val experimentalRebatchDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val experimentalSleepDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val experimentalSlidingWindowDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val experimentalSqlDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t
  -> [ `variant ] t

val experimentalUnbatchDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t

val experimentalUniqueDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t

val expm1
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val extractGlimpse
  :  ?name:string
  -> ?centered:bool
  -> ?normalized:bool
  -> ?uniform_noise:bool
  -> ?noise:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t

val extractImagePatches
  :  ?name:string
  -> ksizes:int list
  -> strides:int list
  -> rates:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val extractJpegShape
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'output_type) Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `int32 | `int64 ] as 'output_type) t

val extractVolumePatches
  :  ?name:string
  -> ksizes:int list
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val fFT
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 'tcomplex) t
  -> ([< `complex64 ] as 'tcomplex) t

val fFT2D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 'tcomplex) t
  -> ([< `complex64 ] as 'tcomplex) t

val fFT3D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 'tcomplex) t
  -> ([< `complex64 ] as 'tcomplex) t

val fIFOQueue
  :  ?name:string
  -> component_types:Type.p list
  -> ?shapes:Dim.t list list
  -> ?capacity:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val fact
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val fakeParam
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val fakeQuantWithMinMaxArgs
  :  ?name:string
  -> ?min:float
  -> ?max:float
  -> ?num_bits:int
  -> ?narrow_range:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t

val fakeQuantWithMinMaxArgsGradient
  :  ?name:string
  -> ?min:float
  -> ?max:float
  -> ?num_bits:int
  -> ?narrow_range:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

val fakeQuantWithMinMaxVars
  :  ?name:string
  -> ?num_bits:int
  -> ?narrow_range:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

val fakeQuantWithMinMaxVarsGradient
  :  ?name:string
  -> ?num_bits:int
  -> ?narrow_range:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t * [ `float ] t * [ `float ] t

val fakeQuantWithMinMaxVarsPerChannel
  :  ?name:string
  -> ?num_bits:int
  -> ?narrow_range:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

val fakeQuantWithMinMaxVarsPerChannelGradient
  :  ?name:string
  -> ?num_bits:int
  -> ?narrow_range:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t * [ `float ] t * [ `float ] t

val fill
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'index_type) t
  -> 't t
  -> 't t

val filterByLastComponentDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t

val fixedLengthRecordDataset
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val fixedLengthRecordDatasetV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `string ] t
  -> [ `variant ] t

val fixedLengthRecordReader
  :  ?name:string
  -> ?header_bytes:int
  -> record_bytes:int
  -> ?footer_bytes:int
  -> ?hop_bytes:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val fixedUnigramCandidateSampler
  :  ?name:string
  -> num_true:int
  -> num_sampled:int
  -> unique:bool
  -> range_max:int
  -> ?vocab_file:string
  -> ?distortion:float
  -> ?num_reserved_ids:int
  -> ?num_shards:int
  -> ?shard:int
  -> ?unigrams:float list
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

val floor
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val floorDiv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val floorMod
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

val fractionalAvgPool
  :  ?name:string
  -> pooling_ratio:float list
  -> ?pseudo_random:bool
  -> ?overlapping:bool
  -> ?deterministic:bool
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t * [ `int64 ] t * [ `int64 ] t

val fractionalAvgPoolGrad
  :  ?name:string
  -> ?overlapping:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val fractionalMaxPool
  :  ?name:string
  -> pooling_ratio:float list
  -> ?pseudo_random:bool
  -> ?overlapping:bool
  -> ?deterministic:bool
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t * [ `int64 ] t * [ `int64 ] t

val fractionalMaxPoolGrad
  :  ?name:string
  -> ?overlapping:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val fusedBatchNorm
  :  ?name:string
  -> ?epsilon:float
  -> ?data_format:string
  -> ?is_training:bool
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t * ([< `float ] as 't) t * ([< `float ] as 't) t * ([< `float ] as 't) t * ([< `float ] as 't) t

val fusedBatchNormGrad
  :  ?name:string
  -> ?epsilon:float
  -> ?data_format:string
  -> ?is_training:bool
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t * ([< `float ] as 't) t * ([< `float ] as 't) t * ([< `float ] as 't) t * ([< `float ] as 't) t

val fusedBatchNormGradV2
  :  ?name:string
  -> ?epsilon:float
  -> ?data_format:string
  -> ?is_training:bool
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> [ `float ] t
  -> ([< `float ] as 'u) t
  -> ([< `float ] as 'u) t
  -> ([< `float ] as 't) t * ([< `float ] as 'u) t * ([< `float ] as 'u) t * ([< `float ] as 'u) t * ([< `float ] as 'u) t

val fusedBatchNormV2
  :  ?name:string
  -> ?epsilon:float
  -> ?data_format:string
  -> ?is_training:bool
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 'u) t
  -> ([< `float ] as 'u) t
  -> ([< `float ] as 'u) t
  -> ([< `float ] as 'u) t
  -> ([< `float ] as 't) t * ([< `float ] as 'u) t * ([< `float ] as 'u) t * ([< `float ] as 'u) t * ([< `float ] as 'u) t

val fusedPadConv2D
  :  ?name:string
  -> mode:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val fusedResizeAndPadConv2D
  :  ?name:string
  -> ?resize_align_corners:bool
  -> mode:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val gather
  :  ?name:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> 'tparams t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 'tparams t

val gatherNd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'tparams t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 'tparams t

val gatherV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'tparams t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `int32 | `int64 ] as 'taxis) t
  -> 'tparams t

val generateVocabRemapping
  :  ?name:string
  -> new_vocab_offset:int
  -> num_new_vocab:int
  -> ?old_vocab_size:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `int64 ] t * [ `int32 ] t

val getSessionHandle
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `string ] t

val getSessionTensor
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'dtype t

val greater
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

val greaterEqual
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

val guaranteeConst
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val hSVToRGB
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val hashTable
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?use_node_name_sharing:bool
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val histogramFixedWidth
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'dtype) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 ] as 'dtype) t

val histogramSummary
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `string ] t

val hostConst
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val iFFT
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 'tcomplex) t
  -> ([< `complex64 ] as 'tcomplex) t

val iFFT2D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 'tcomplex) t
  -> ([< `complex64 ] as 'tcomplex) t

val iFFT3D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 'tcomplex) t
  -> ([< `complex64 ] as 'tcomplex) t

val iRFFT
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `int32 ] t
  -> [ `float ] t

val iRFFT2D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `int32 ] t
  -> [ `float ] t

val iRFFT3D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `int32 ] t
  -> [ `float ] t

val identity
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val identityReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val igamma
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val igammaGradA
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val igammac
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val imag
  :  ?name:string
  -> type_:([< `float | `double ] as 'tout) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 't) t
  -> ([< `float | `double ] as 'tout) t

val imageSummary
  :  ?name:string
  -> ?max_images:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float | `double ] as 't) t
  -> [ `string ] t

val immutableConst
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> memory_region_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val inTopK
  :  ?name:string
  -> k:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> ([< `int32 | `int64 ] as 't) t
  -> [ `bool ] t

val inTopKV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> [ `bool ] t

val infeedDequeue
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val infeedEnqueue
  :  ?name:string
  -> ?shape:Dim.t list
  -> ?layout:int list
  -> ?device_ordinal:int
  -> ?control_inputs:Node.p list
  -> 'dtype t
  -> [ `unit ] t

val infeedEnqueuePrelinearizedBuffer
  :  ?name:string
  -> ?device_ordinal:int
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `unit ] t

val initializeTable
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tkey t
  -> 'tval t
  -> [ `unit ] t

val initializeTableFromTextFile
  :  ?name:string
  -> key_index:int
  -> value_index:int
  -> ?vocab_size:int
  -> ?delimiter:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `unit ] t

val inplaceAdd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int32 ] t
  -> 't t
  -> 't t

val inplaceSub
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int32 ] t
  -> 't t
  -> 't t

val inplaceUpdate
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int32 ] t
  -> 't t
  -> 't t

val inv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val invGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val invert
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val invertPermutation
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val isFinite
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `bool ] t

val isInf
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `bool ] t

val isNan
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `bool ] t

val isVariableInitialized
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'dtype t
  -> [ `bool ] t

val kMC2ChainInitialization
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int64 ] t
  -> [ `int64 ] t

val kmeansPlusPlusInitialization
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `float ] t

val l2Loss
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val lMDBReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val lRN
  :  ?name:string
  -> ?depth_radius:int
  -> ?bias:float
  -> ?alpha:float
  -> ?beta:float
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t

val lRNGrad
  :  ?name:string
  -> ?depth_radius:int
  -> ?bias:float
  -> ?alpha:float
  -> ?beta:float
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t

val leakyRelu
  :  ?name:string
  -> ?alpha:float
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val leakyReluGrad
  :  ?name:string
  -> ?alpha:float
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val learnedUnigramCandidateSampler
  :  ?name:string
  -> num_true:int
  -> num_sampled:int
  -> unique:bool
  -> range_max:int
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

val leftShift
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val less
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

val lessEqual
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

val lgamma
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val linSpace
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double ] as 't) t

val listDiff
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t
  -> 't t * ([< `int32 | `int64 ] as 'out_idx) t

val loadAndRemapMatrix
  :  ?name:string
  -> num_rows:int
  -> num_cols:int
  -> ?max_rows_in_memory:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `float ] t
  -> [ `float ] t

val loadTPUEmbeddingADAMParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingADAMParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingAdadeltaParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingAdadeltaParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingAdagradParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingAdagradParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingCenteredRMSPropParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingFTRLParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingFTRLParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingMDLAdagradLightParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingMomentumParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingMomentumParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingProximalAdagradParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingProximalAdagradParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingRMSPropParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingRMSPropParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `unit ] t

val loadTPUEmbeddingStochasticGradientDescentParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `unit ] t

val log
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val log1p
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val logMatrixDeterminant
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t * ([< `float | `double | `complex64 ] as 't) t

val logSoftmax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val logUniformCandidateSampler
  :  ?name:string
  -> num_true:int
  -> num_sampled:int
  -> unique:bool
  -> range_max:int
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

val logicalAnd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> [ `bool ] t
  -> [ `bool ] t

val logicalNot
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> [ `bool ] t

val logicalOr
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> [ `bool ] t
  -> [ `bool ] t

val lookupTableExport
  :  ?name:string
  -> type_:'tkeys Type.t
  -> type_1:'tvalues Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tkeys t * 'tvalues t

val lookupTableFind
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tin t
  -> 'tout t
  -> 'tout t

val lookupTableImport
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tin t
  -> 'tout t
  -> [ `unit ] t

val lookupTableInsert
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tin t
  -> 'tout t
  -> [ `unit ] t

val lookupTableSize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

val loopCond
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> [ `bool ] t

val lowerBound
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t
  -> ([< `int32 | `int64 ] as 'out_type) t

val lu
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'output_idx_type) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t * ([< `int32 | `int64 ] as 'output_idx_type) t

val mapClear
  :  ?name:string
  -> ?capacity:int
  -> ?memory_limit:int
  -> dtypes:Type.p list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

val mapIncompleteSize
  :  ?name:string
  -> ?capacity:int
  -> ?memory_limit:int
  -> dtypes:Type.p list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `int32 ] t

val mapSize
  :  ?name:string
  -> ?capacity:int
  -> ?memory_limit:int
  -> dtypes:Type.p list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `int32 ] t

val matMul
  :  ?name:string
  -> ?transpose_a:bool
  -> ?transpose_b:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val matchingFiles
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val matrixBandPart
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindex) t
  -> ([< `int32 | `int64 ] as 'tindex) t
  -> 't t

val matrixDeterminant
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val matrixDiag
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val matrixDiagPart
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val matrixExponential
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t

val matrixInverse
  :  ?name:string
  -> ?adjoint:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t

val matrixLogarithm
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 't) t
  -> ([< `complex64 ] as 't) t

val matrixSetDiag
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t
  -> 't t

val matrixSolve
  :  ?name:string
  -> ?adjoint:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t

val matrixSolveLs
  :  ?name:string
  -> ?fast:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t
  -> [ `double ] t
  -> ([< `double | `float | `complex64 ] as 't) t

val matrixSquareRoot
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t

val matrixTriangularSolve
  :  ?name:string
  -> ?lower:bool
  -> ?adjoint:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t

val max
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val maxPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val maxPool3D
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t

val maxPool3DGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 'tInput) t
  -> ([< `float ] as 'tInput) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t

val maxPool3DGradGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val maxPoolGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val maxPoolGradGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val maxPoolGradGradV2
  :  ?name:string
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val maxPoolGradGradWithArgmax
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?include_batch_in_index:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'targmax) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val maxPoolGradV2
  :  ?name:string
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val maxPoolGradWithArgmax
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?include_batch_in_index:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'targmax) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val maxPoolV2
  :  ?name:string
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val maxPoolWithArgmax
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'targmax) Type.t
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?include_batch_in_index:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t * ([< `int32 | `int64 ] as 'targmax) t

val maximum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val mean
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val merge
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t list
  -> 't t * [ `int32 ] t

val mergeSummary
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t list
  -> [ `string ] t

val mergeV2Checkpoints
  :  ?name:string
  -> ?delete_old_dirs:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `unit ] t

val mfcc
  :  ?name:string
  -> ?upper_frequency_limit:float
  -> ?lower_frequency_limit:float
  -> ?filterbank_channel_count:int
  -> ?dct_coefficient_count:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `float ] t

val min
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val minimum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val mirrorPad
  :  ?name:string
  -> mode:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

val mirrorPadGrad
  :  ?name:string
  -> mode:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

val mod_
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

val modelDataset
  :  ?name:string
  -> ?cpu_budget:int
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t

val mul
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val mulNoNan
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val multinomial
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'output_dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 ] as 'output_dtype) t

val mutableDenseHashTable
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?use_node_name_sharing:bool
  -> ?value_shape:Dim.t list
  -> ?initial_num_buckets:int
  -> ?max_load_factor:float
  -> ?control_inputs:Node.p list
  -> 'key_dtype t
  -> [ `string ] t

val mutableHashTable
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?use_node_name_sharing:bool
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val mutableHashTableOfTensors
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?use_node_name_sharing:bool
  -> ?value_shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val ncclAllReduce
  :  ?name:string
  -> reduction:string
  -> num_devices:int
  -> shared_name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val ncclBroadcast
  :  ?name:string
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val ncclReduce
  :  ?name:string
  -> reduction:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val nearestNeighbors
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t

val neg
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val negTrain
  :  ?name:string
  -> vocab_count:int list
  -> num_negative_samples:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `unit ] t

val nextAfter
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

val nextIteration
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val noOp
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

val nonDeterministicInts
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?control_inputs:Node.p list
  -> 'shape_dtype t
  -> 'dtype t

val nonMaxSuppression
  :  ?name:string
  -> ?iou_threshold:float
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `int32 ] t

val nonMaxSuppressionV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `int32 ] t

val nonMaxSuppressionV3
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t

val nonMaxSuppressionV4
  :  ?name:string
  -> ?pad_to_max_output_size:bool
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t * [ `int32 ] t

val nonMaxSuppressionWithOverlaps
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t

val notEqual
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> [ `bool ] t

val nthElement
  :  ?name:string
  -> ?reverse:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val oneHot
  :  ?name:string
  -> ?axis:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tI) t
  -> [ `int32 ] t
  -> 't t
  -> 't t
  -> 't t

val onesLike
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `bool ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `bool ] as 't) t

val optimizeDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `string ] t
  -> [ `variant ] t

val optionalHasValue
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `bool ] t

val optionalNone
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `variant ] t

val orderedMapClear
  :  ?name:string
  -> ?capacity:int
  -> ?memory_limit:int
  -> dtypes:Type.p list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

val orderedMapIncompleteSize
  :  ?name:string
  -> ?capacity:int
  -> ?memory_limit:int
  -> dtypes:Type.p list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `int32 ] t

val orderedMapSize
  :  ?name:string
  -> ?capacity:int
  -> ?memory_limit:int
  -> dtypes:Type.p list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `int32 ] t

val outfeedDequeue
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?device_ordinal:int
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val outfeedEnqueue
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'dtype t
  -> [ `unit ] t

val pack
  :  ?name:string
  -> ?axis:int
  -> ?control_inputs:Node.p list
  -> 't t list
  -> 't t

val pad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

val padV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t
  -> 't t

val paddingFIFOQueue
  :  ?name:string
  -> component_types:Type.p list
  -> ?shapes:Dim.t list list
  -> ?capacity:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val parallelConcat
  :  ?name:string
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> 't t list
  -> 't t

val parallelDynamicStitch
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t list
  -> 't t list
  -> 't t

val parameterizedTruncatedNormal
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `float | `double ] as 'dtype) t
  -> ([< `float | `double ] as 'dtype) t
  -> ([< `float | `double ] as 'dtype) t
  -> ([< `float | `double ] as 'dtype) t
  -> ([< `float | `double ] as 'dtype) t

val parseTensor
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'out_type t

val placeholder
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val placeholderV2
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val placeholderWithDefault
  :  ?name:string
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> 'dtype t
  -> 'dtype t

val polygamma
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val pow
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val prefetchDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val prelinearize
  :  ?name:string
  -> ?shape:Dim.t list
  -> ?layout:int list
  -> ?control_inputs:Node.p list
  -> 'dtype t
  -> [ `variant ] t

val preventGradient
  :  ?name:string
  -> ?message:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val printV2
  :  ?name:string
  -> ?output_stream:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

val priorityQueue
  :  ?name:string
  -> ?component_types:Type.p list
  -> shapes:Dim.t list list
  -> ?capacity:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val prod
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val qr
  :  ?name:string
  -> ?full_matrices:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t * ([< `double | `float | `complex64 ] as 't) t

val quantizeAndDequantize
  :  ?name:string
  -> ?signed_input:bool
  -> ?num_bits:int
  -> ?range_given:bool
  -> ?input_min:float
  -> ?input_max:float
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val quantizeAndDequantizeV2
  :  ?name:string
  -> ?signed_input:bool
  -> ?num_bits:int
  -> ?range_given:bool
  -> ?round_mode:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val quantizeAndDequantizeV3
  :  ?name:string
  -> ?signed_input:bool
  -> ?range_given:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

val quantizeDownAndShrinkRange
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizeV2
  :  ?name:string
  -> type_:'t Type.t
  -> ?mode:string
  -> ?round_mode:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 't t * [ `float ] t * [ `float ] t

val quantizedAdd
  :  ?name:string
  -> type_:'toutput Type.t
  -> ?control_inputs:Node.p list
  -> 't1 t
  -> 't2 t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'toutput t * [ `float ] t * [ `float ] t

val quantizedAvgPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `float ] t
  -> [ `float ] t
  -> 't t * [ `float ] t * [ `float ] t

val quantizedBatchNormWithGlobalNormalization
  :  ?name:string
  -> type_:'out_type Type.t
  -> variance_epsilon:float
  -> scale_after_normalization:bool
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedBiasAdd
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 't1 t
  -> 't2 t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConcat
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 't t list
  -> [ `float ] t list
  -> [ `float ] t list
  -> 't t * [ `float ] t * [ `float ] t

val quantizedConv2D
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DAndRelu
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DAndReluAndRequantize
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DAndRequantize
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DPerChannel
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DWithBias
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DWithBiasAndRelu
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DWithBiasAndReluAndRequantize
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> ([< `float ] as 'tbias) t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DWithBiasAndRequantize
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> ([< `float ] as 'tbias) t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DWithBiasSignedSumAndReluAndRequantize
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> ([< `float ] as 'tbias) t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'tsummand t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DWithBiasSumAndRelu
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedConv2DWithBiasSumAndReluAndRequantize
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?padding_list:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> ([< `float ] as 'tbias) t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'tsummand t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedDepthwiseConv2D
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedDepthwiseConv2DWithBias
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedDepthwiseConv2DWithBiasAndRelu
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedDepthwiseConv2DWithBiasAndReluAndRequantize
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?dilations:int list
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> ([< `float ] as 'tbias) t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedInstanceNorm
  :  ?name:string
  -> ?output_range_given:bool
  -> ?given_y_min:float
  -> ?given_y_max:float
  -> ?variance_epsilon:float
  -> ?min_separation:float
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `float ] t
  -> [ `float ] t
  -> 't t * [ `float ] t * [ `float ] t

val quantizedMatMul
  :  ?name:string
  -> type_:'toutput Type.t
  -> ?transpose_a:bool
  -> ?transpose_b:bool
  -> ?control_inputs:Node.p list
  -> 't1 t
  -> 't2 t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'toutput t * [ `float ] t * [ `float ] t

val quantizedMaxPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `float ] t
  -> [ `float ] t
  -> 't t * [ `float ] t * [ `float ] t

val quantizedMul
  :  ?name:string
  -> type_:'toutput Type.t
  -> ?control_inputs:Node.p list
  -> 't1 t
  -> 't2 t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'toutput t * [ `float ] t * [ `float ] t

val quantizedRelu
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedRelu6
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedReluX
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val quantizedReshape
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tshape) t
  -> [ `float ] t
  -> [ `float ] t
  -> 't t * [ `float ] t * [ `float ] t

val quantizedResizeBilinear
  :  ?name:string
  -> ?align_corners:bool
  -> ?half_pixel_centers:bool
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t
  -> ([< `float ] as 't) t * [ `float ] t * [ `float ] t

val queueClose
  :  ?name:string
  -> ?cancel_pending_enqueues:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

val queueIsClosed
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `bool ] t

val queueSize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t

val rFFT
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `complex64 ] t

val rFFT2D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `complex64 ] t

val rFFT3D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `complex64 ] t

val rGBToHSV
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val raggedRange
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int64 ] t * ([< `float | `double | `int32 | `int64 ] as 't) t

val raggedTensorToSparse
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t list
  -> 't t
  -> [ `int64 ] t * 't t * [ `int64 ] t

val randomCrop
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int64 ] t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

val randomGamma
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 's) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val randomGammaGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val randomPoisson
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 's) t
  -> ([< `float | `double ] as 'dtype) t
  -> ([< `float | `double ] as 'dtype) t

val randomPoissonV2
  :  ?name:string
  -> type_:([< `float | `double | `int32 | `int64 ] as 'dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 's) t
  -> ([< `float | `double | `int32 | `int64 ] as 'r) t
  -> ([< `float | `double | `int32 | `int64 ] as 'dtype) t

val randomShuffle
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val randomShuffleQueue
  :  ?name:string
  -> component_types:Type.p list
  -> ?shapes:Dim.t list list
  -> ?capacity:int
  -> ?min_after_dequeue:int
  -> ?seed:int
  -> ?seed2:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val randomStandardNormal
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `float | `double ] as 'dtype) t

val randomUniform
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `float | `double ] as 'dtype) t

val randomUniformInt
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tout) t
  -> ([< `int32 | `int64 ] as 'tout) t
  -> ([< `int32 | `int64 ] as 'tout) t

val range
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `int64 ] as 'tidx) t

val rangeDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val rank
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int32 ] t

val readFile
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val readerNumRecordsProduced
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

val readerNumWorkUnitsCompleted
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

val readerRead
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t * [ `string ] t

val readerReadUpTo
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `string ] t * [ `string ] t

val readerReset
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

val readerRestoreState
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `unit ] t

val readerSerializeState
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val real
  :  ?name:string
  -> type_:([< `float | `double ] as 'tout) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 't) t
  -> ([< `float | `double ] as 'tout) t

val realDiv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val reciprocal
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val reciprocalGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val recordInput
  :  ?name:string
  -> file_pattern:string
  -> ?file_random_seed:int
  -> ?file_shuffle_shift_ratio:float
  -> ?file_buffer_size:int
  -> ?file_parallelism:int
  -> ?batch_size:int
  -> ?compression_type:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val recvTPUEmbeddingActivations
  :  ?name:string
  -> num_outputs:int
  -> config:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t list

val reduceJoin
  :  ?name:string
  -> ?keep_dims:bool
  -> ?separator:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `string ] t

val refEnter
  :  ?name:string
  -> frame_name:string
  -> ?is_constant:bool
  -> ?parallel_iterations:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val refExit
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val refIdentity
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val refMerge
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t list
  -> 't t * [ `int32 ] t

val refNextIteration
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val refSelect
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 't t list
  -> 't t

val refSwitch
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `bool ] t
  -> 't t * 't t

val regexFullMatch
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `bool ] t

val regexReplace
  :  ?name:string
  -> ?replace_global:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t

val relu
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val relu6
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val relu6Grad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val reluGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val repeatDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val requantizationRange
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t * [ `float ] t

val requantizationRangePerChannel
  :  ?name:string
  -> clip_value_max:float
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t * [ `float ] t

val requantize
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val requantizePerChannel
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

val reshape
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tshape) t
  -> 't t

val resizeArea
  :  ?name:string
  -> ?align_corners:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t

val resizeBicubic
  :  ?name:string
  -> ?align_corners:bool
  -> ?half_pixel_centers:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t

val resizeBicubicGrad
  :  ?name:string
  -> ?align_corners:bool
  -> ?half_pixel_centers:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val resizeBilinear
  :  ?name:string
  -> ?align_corners:bool
  -> ?half_pixel_centers:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t

val resizeBilinearGrad
  :  ?name:string
  -> ?align_corners:bool
  -> ?half_pixel_centers:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val resizeNearestNeighbor
  :  ?name:string
  -> ?align_corners:bool
  -> ?half_pixel_centers:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

val resizeNearestNeighborGrad
  :  ?name:string
  -> ?align_corners:bool
  -> ?half_pixel_centers:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `int32 | `float | `double ] as 't) t

val restore
  :  ?name:string
  -> type_:'dt Type.t
  -> ?preferred_shard:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> 'dt t

val restoreSlice
  :  ?name:string
  -> type_:'dt Type.t
  -> ?preferred_shard:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t
  -> 'dt t

val retrieveTPUEmbeddingADAMParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingADAMParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingAdadeltaParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingAdadeltaParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingAdagradParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingAdagradParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingCenteredRMSPropParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingFTRLParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingFTRLParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingMDLAdagradLightParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingMomentumParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingMomentumParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingProximalAdagradParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingRMSPropParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingRMSPropParametersGradAccumDebug
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t * [ `float ] t * [ `float ] t * [ `float ] t

val retrieveTPUEmbeddingStochasticGradientDescentParameters
  :  ?name:string
  -> ?table_id:int
  -> ?table_name:string
  -> num_shards:int
  -> shard_id:int
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `float ] t

val reverse
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `bool | `float | `double | `complex64 | `string ] as 't) t
  -> [ `bool ] t
  -> ([< `int32 | `int64 | `bool | `float | `double | `complex64 | `string ] as 't) t

val reverseSequence
  :  ?name:string
  -> seq_dim:int
  -> ?batch_dim:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tlen) t
  -> 't t

val reverseV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `bool | `float | `double | `complex64 | `string ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `int32 | `int64 | `bool | `float | `double | `complex64 | `string ] as 't) t

val rightShift
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

val rint
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val roll
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tshift) t
  -> ([< `int32 | `int64 ] as 'taxis) t
  -> 't t

val round
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val rpc
  :  ?name:string
  -> ?protocol:string
  -> ?fail_fast:bool
  -> ?timeout_in_ms:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t

val rsqrt
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val rsqrtGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val sampleDistortedBoundingBox
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?min_object_covered:float
  -> ?aspect_ratio_range:float list
  -> ?area_range:float list
  -> ?max_attempts:int
  -> ?use_image_if_no_bounding_boxes:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> [ `float ] t
  -> ([< `int32 | `int64 ] as 't) t * ([< `int32 | `int64 ] as 't) t * [ `float ] t

val sampleDistortedBoundingBoxV2
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?aspect_ratio_range:float list
  -> ?area_range:float list
  -> ?max_attempts:int
  -> ?use_image_if_no_bounding_boxes:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> [ `float ] t
  -> [ `float ] t
  -> ([< `int32 | `int64 ] as 't) t * ([< `int32 | `int64 ] as 't) t * [ `float ] t

val samplingDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `float ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val scalarSummary
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `string ] t

val scaleAndTranslate
  :  ?name:string
  -> ?kernel_type:string
  -> ?antialias:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

val scaleAndTranslateGrad
  :  ?name:string
  -> ?kernel_type:string
  -> ?antialias:bool
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> [ `float ] t
  -> [ `float ] t
  -> ([< `float ] as 't) t

val scatterAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val scatterDiv
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val scatterMax
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val scatterMin
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val scatterMul
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val scatterNd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t

val scatterNdAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val scatterNdNonAliasingAdd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t

val scatterNdSub
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val scatterNdUpdate
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t

val scatterSub
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val scatterUpdate
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t

val sdcaFprint
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

val sdcaShrinkL1
  :  ?name:string
  -> l1:float
  -> l2:float
  -> ?control_inputs:Node.p list
  -> [ `float ] t list
  -> [ `unit ] t

val segmentMax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val segmentMean
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val segmentMin
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val segmentProd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val segmentSum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val select
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> 't t
  -> 't t
  -> 't t

val selfAdjointEig
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

val selfAdjointEigV2
  :  ?name:string
  -> ?compute_v:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t * ([< `double | `float | `complex64 ] as 't) t

val selu
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val seluGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val sendTPUEmbeddingGradients
  :  ?name:string
  -> config:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t list
  -> [ `float ] t list
  -> [ `unit ] t

val serializeManySparse
  :  ?name:string
  -> type_:([< `string | `variant ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> ([< `string | `variant ] as 'out_type) t

val serializeSparse
  :  ?name:string
  -> type_:([< `string | `variant ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> ([< `string | `variant ] as 'out_type) t

val serializeTensor
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `string ] t

val setSize
  :  ?name:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> [ `int64 ] t
  -> [ `int32 ] t

val shape
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'out_type) t

val shapeN
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> 't t list
  -> ([< `int32 | `int64 ] as 'out_type) t list

val shardDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val shardedFilename
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `string ] t

val shardedFilespec
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `string ] t

val shuffleAndRepeatDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val shuffleDataset
  :  ?name:string
  -> ?reshuffle_each_iteration:bool
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val shutdownDistributedTPU
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

val sigmoid
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val sigmoidGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val sign
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val sin
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val sinh
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val size
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'out_type) t

val skipDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val skipgram
  :  ?name:string
  -> filename:string
  -> batch_size:int
  -> ?window_size:int
  -> ?min_count:int
  -> ?subsample:float
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t * [ `int32 ] t * [ `int64 ] t * [ `int32 ] t * [ `int64 ] t * [ `int32 ] t * [ `int32 ] t

val slice
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> 't t

val snapshot
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val softmax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val softmaxCrossEntropyWithLogits
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t

val softplus
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val softplusGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val softsign
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val softsignGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val spaceToBatch
  :  ?name:string
  -> block_size:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

val spaceToBatchND
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tblock_shape) t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

val spaceToDepth
  :  ?name:string
  -> block_size:int
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val sparseAccumulatorApplyGradient
  :  ?name:string
  -> has_known_shape:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 'dtype) t
  -> [ `int64 ] t
  -> [ `unit ] t

val sparseAccumulatorTakeGradient
  :  ?name:string
  -> type_1:([< `float | `double | `int32 | `complex64 | `int64 ] as 'dtype) Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `int64 ] t * ([< `float | `double | `int32 | `complex64 | `int64 ] as 'dtype) t * [ `int64 ] t

val sparseAdd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 'treal) t
  -> [ `int64 ] t * ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t * [ `int64 ] t

val sparseAddGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t * ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyAdadelta
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ?update_slots:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyAdagradDA
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyCenteredRMSProp
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyFtrl
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyFtrlV2
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyMomentum
  :  ?name:string
  -> ?use_locking:bool
  -> ?use_nesterov:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyProximalAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyProximalGradientDescent
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseApplyRMSProp
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseConcat
  :  ?name:string
  -> concat_dim:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t list
  -> 't t list
  -> [ `int64 ] t list
  -> [ `int64 ] t * 't t * [ `int64 ] t

val sparseConditionalAccumulator
  :  ?name:string
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> ?reduction_type:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val sparseDenseCwiseAdd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseDenseCwiseDiv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseDenseCwiseMul
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseFillEmptyRows
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t * 't t * [ `bool ] t * [ `int64 ] t

val sparseFillEmptyRowsGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> 't t * 't t

val sparseMatMul
  :  ?name:string
  -> ?transpose_a:bool
  -> ?transpose_b:bool
  -> ?a_is_sparse:bool
  -> ?b_is_sparse:bool
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 'ta) t
  -> ([< `float ] as 'tb) t
  -> [ `float ] t

val sparseReduceMax
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val sparseReduceMaxSparse
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> [ `int64 ] t * ([< `float | `double | `int32 | `int64 ] as 't) t * [ `int64 ] t

val sparseReduceSum
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseReduceSumSparse
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> [ `int64 ] t * ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t * [ `int64 ] t

val sparseReorder
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t * 't t

val sparseReshape
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `int64 ] t

val sparseSegmentMean
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

val sparseSegmentMeanGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

val sparseSegmentMeanWithNumSegments
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 ] as 'tnumsegments) t
  -> ([< `float | `double ] as 't) t

val sparseSegmentSqrtN
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

val sparseSegmentSqrtNGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

val sparseSegmentSqrtNWithNumSegments
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 ] as 'tnumsegments) t
  -> ([< `float | `double ] as 't) t

val sparseSegmentSum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val sparseSegmentSumWithNumSegments
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 ] as 'tnumsegments) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val sparseSlice
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t * 't t * [ `int64 ] t

val sparseSliceGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseSoftmax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double ] as 't) t

val sparseSoftmaxCrossEntropyWithLogits
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tlabels) t
  -> ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t

val sparseSparseMaximum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t * ([< `float | `double | `int32 | `int64 ] as 't) t

val sparseSparseMinimum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t * ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseTensorDenseAdd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val sparseTensorDenseMatMul
  :  ?name:string
  -> ?adjoint_a:bool
  -> ?adjoint_b:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> [ `int64 ] t
  -> 't t
  -> 't t

val sparseTensorSliceDataset
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 'tvalues t
  -> [ `int64 ] t
  -> [ `variant ] t

val sparseToDense
  :  ?name:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t
  -> 't t

val sparseToSparseSetOperation
  :  ?name:string
  -> set_operation:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t * ([< `int32 | `int64 | `string ] as 't) t * [ `int64 ] t

val split
  :  ?name:string
  -> num_split:int
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 't t
  -> 't t list

val splitV
  :  ?name:string
  -> num_split:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tlen) t
  -> [ `int32 ] t
  -> 't t list

val sqrt
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val sqrtGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val square
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val squaredDifference
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val squeeze
  :  ?name:string
  -> ?squeeze_dims:int list
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val stack
  :  ?name:string
  -> ?stack_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val stackClose
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

val stackPop
  :  ?name:string
  -> type_:'elem_type Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'elem_type t

val stackPush
  :  ?name:string
  -> ?swap_memory:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 't t
  -> 't t

val stageClear
  :  ?name:string
  -> ?capacity:int
  -> ?memory_limit:int
  -> dtypes:Type.p list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

val stageSize
  :  ?name:string
  -> ?capacity:int
  -> ?memory_limit:int
  -> dtypes:Type.p list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `int32 ] t

val statelessMultinomial
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'output_dtype) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 ] as 'tseed) t
  -> ([< `int32 | `int64 ] as 'output_dtype) t

val statelessRandomNormal
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tseed) t
  -> ([< `float | `double ] as 'dtype) t

val statelessRandomUniform
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tseed) t
  -> ([< `float | `double ] as 'dtype) t

val statelessRandomUniformInt
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tseed) t
  -> ([< `int32 | `int64 ] as 'dtype) t
  -> ([< `int32 | `int64 ] as 'dtype) t
  -> ([< `int32 | `int64 ] as 'dtype) t

val statelessTruncatedNormal
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tseed) t
  -> ([< `float | `double ] as 'dtype) t

val staticRegexFullMatch
  :  ?name:string
  -> pattern:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `bool ] t

val staticRegexReplace
  :  ?name:string
  -> pattern:string
  -> rewrite:string
  -> ?replace_global:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val stopGradient
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val stridedSlice
  :  ?name:string
  -> ?begin_mask:int
  -> ?end_mask:int
  -> ?ellipsis_mask:int
  -> ?new_axis_mask:int
  -> ?shrink_axis_mask:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> 't t

val stridedSliceAssign
  :  ?name:string
  -> ?begin_mask:int
  -> ?end_mask:int
  -> ?ellipsis_mask:int
  -> ?new_axis_mask:int
  -> ?shrink_axis_mask:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> 't t
  -> 't t

val stridedSliceGrad
  :  ?name:string
  -> ?begin_mask:int
  -> ?end_mask:int
  -> ?ellipsis_mask:int
  -> ?new_axis_mask:int
  -> ?shrink_axis_mask:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> 't t
  -> 't t

val stringJoin
  :  ?name:string
  -> ?separator:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t list
  -> [ `string ] t

val stringLength
  :  ?name:string
  -> ?unit:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t

val stringSplit
  :  ?name:string
  -> ?skip_empty:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `int64 ] t * [ `string ] t * [ `int64 ] t

val stringSplitV2
  :  ?name:string
  -> ?maxsplit:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `int64 ] t * [ `string ] t * [ `int64 ] t

val stringStrip
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val stringToHashBucket
  :  ?name:string
  -> num_buckets:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

val stringToHashBucketFast
  :  ?name:string
  -> num_buckets:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

val stringToHashBucketStrong
  :  ?name:string
  -> num_buckets:int
  -> key:int list
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

val stringToNumber
  :  ?name:string
  -> type_:([< `float | `double | `int32 | `int64 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float | `double | `int32 | `int64 ] as 'out_type) t

val sub
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val substr
  :  ?name:string
  -> ?unit:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> [ `string ] t

val sum
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val svd
  :  ?name:string
  -> ?compute_uv:bool
  -> ?full_matrices:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t * ([< `double | `float | `complex64 ] as 't) t * ([< `double | `float | `complex64 ] as 't) t

val switch
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `bool ] t
  -> 't t * 't t

val tFRecordDataset
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val tFRecordReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?compression_type:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val tPUCompilationResult
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val tPUEmbeddingActivations
  :  ?name:string
  -> table_id:int
  -> lookup_id:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

val tPUOrdinalSelector
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `int32 ] t

val tPUReplicateMetadata
  :  ?name:string
  -> num_replicas:int
  -> ?num_cores_per_replica:int
  -> ?topology:string
  -> ?use_tpu:bool
  -> ?device_assignment:int list
  -> ?computation_shape:int list
  -> ?step_marker_location:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

val tPUReplicatedInput
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t list
  -> 't t

val tPUReplicatedOutput
  :  ?name:string
  -> num_replicas:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t list

val takeDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val takeManySparseFromTensorsMap
  :  ?name:string
  -> type_1:'dtype Type.t
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t * 'dtype t * [ `int64 ] t

val tan
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val tanh
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val tanhGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val temporaryVariable
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?var_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val tensorArray
  :  ?name:string
  -> ?dynamic_size:bool
  -> ?clear_after_read:bool
  -> ?tensor_array_name:string
  -> ?element_shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `string ] t

val tensorArrayClose
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

val tensorArrayCloseV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

val tensorArrayConcat
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?element_shape_except0:Dim.t list
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> 'dtype t * [ `int64 ] t

val tensorArrayConcatV2
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?element_shape_except0:Dim.t list
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> 'dtype t * [ `int64 ] t

val tensorArrayGather
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?element_shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> 'dtype t

val tensorArrayGatherV2
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?element_shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> 'dtype t

val tensorArrayGrad
  :  ?name:string
  -> source:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> [ `string ] t

val tensorArrayGradV2
  :  ?name:string
  -> source:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> [ `string ] t

val tensorArrayPack
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?element_shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> 'dtype t

val tensorArrayRead
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> 'dtype t

val tensorArrayReadV2
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> 'dtype t

val tensorArrayScatter
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> 't t
  -> [ `float ] t
  -> [ `float ] t

val tensorArrayScatterV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> 't t
  -> [ `float ] t
  -> [ `float ] t

val tensorArraySize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> [ `int32 ] t

val tensorArraySizeV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> [ `int32 ] t

val tensorArraySplit
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `float ] t
  -> [ `float ] t

val tensorArraySplitV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `float ] t
  -> [ `float ] t

val tensorArrayUnpack
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 't t
  -> [ `float ] t
  -> [ `float ] t

val tensorArrayV2
  :  ?name:string
  -> ?element_shape:Dim.t list
  -> ?dynamic_size:bool
  -> ?clear_after_read:bool
  -> ?tensor_array_name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `string ] t

val tensorArrayWrite
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> 't t
  -> [ `float ] t
  -> [ `float ] t

val tensorArrayWriteV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> 't t
  -> [ `float ] t
  -> [ `float ] t

val tensorListConcat
  :  ?name:string
  -> type_:'element_dtype Type.t
  -> ?element_shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> 'element_dtype t * [ `int64 ] t

val tensorListConcatLists
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t
  -> [ `variant ] t

val tensorListConcatV2
  :  ?name:string
  -> type_:'element_dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> ([< `int32 | `int64 ] as 'shape_type) t
  -> [ `int64 ] t
  -> 'element_dtype t * [ `int64 ] t

val tensorListElementShape
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'shape_type) Type.t
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> ([< `int32 | `int64 ] as 'shape_type) t

val tensorListFromTensor
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'element_dtype t
  -> ([< `int32 | `int64 ] as 'shape_type) t
  -> [ `variant ] t

val tensorListGather
  :  ?name:string
  -> type_:'element_dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> 'element_dtype t

val tensorListGetItem
  :  ?name:string
  -> type_:'element_dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> 'element_dtype t

val tensorListLength
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int32 ] t

val tensorListPopBack
  :  ?name:string
  -> type_1:'element_dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int32 ] t
  -> [ `variant ] t * 'element_dtype t

val tensorListPushBack
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> 'element_dtype t
  -> [ `variant ] t

val tensorListPushBackBatch
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> 'element_dtype t
  -> [ `variant ] t

val tensorListReserve
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'shape_type) t
  -> [ `int32 ] t
  -> [ `variant ] t

val tensorListResize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int32 ] t
  -> [ `variant ] t

val tensorListScatter
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'element_dtype t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 ] as 'shape_type) t
  -> [ `variant ] t

val tensorListScatterIntoExistingList
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> 'element_dtype t
  -> [ `int32 ] t
  -> [ `variant ] t

val tensorListScatterV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'element_dtype t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 ] as 'shape_type) t
  -> [ `int32 ] t
  -> [ `variant ] t

val tensorListSetItem
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int32 ] t
  -> 'element_dtype t
  -> [ `variant ] t

val tensorListSplit
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'element_dtype t
  -> ([< `int32 | `int64 ] as 'shape_type) t
  -> [ `int64 ] t
  -> [ `variant ] t

val tensorListStack
  :  ?name:string
  -> type_:'element_dtype Type.t
  -> ?num_elements:int
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int32 ] t
  -> 'element_dtype t

val tensorScatterAdd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t

val tensorScatterSub
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t

val tensorScatterUpdate
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t

val tensorStridedSliceUpdate
  :  ?name:string
  -> ?begin_mask:int
  -> ?end_mask:int
  -> ?ellipsis_mask:int
  -> ?new_axis_mask:int
  -> ?shrink_axis_mask:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> 't t
  -> 't t

val tensorSummary
  :  ?name:string
  -> ?description:string
  -> ?display_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `string ] t

val tensorSummaryV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 't t
  -> [ `string ] t
  -> [ `string ] t

val textLineDataset
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `variant ] t

val textLineReader
  :  ?name:string
  -> ?skip_header_lines:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val threadUnsafeUnigramCandidateSampler
  :  ?name:string
  -> num_true:int
  -> num_sampled:int
  -> unique:bool
  -> range_max:int
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

val tile
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tmultiples) t
  -> 't t

val tileGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int32 ] t
  -> 't t

val timestamp
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `double ] t

val topK
  :  ?name:string
  -> k:int
  -> ?sorted:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t * [ `int32 ] t

val topKV2
  :  ?name:string
  -> ?sorted:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t * [ `int32 ] t

val transpose
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tperm) t
  -> 't t

val tridiagonalSolve
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t

val truncateDiv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

val truncateMod
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

val truncatedNormal
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `float | `double ] as 'dtype) t

val tryRpc
  :  ?name:string
  -> ?protocol:string
  -> ?fail_fast:bool
  -> ?timeout_in_ms:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t * [ `int32 ] t * [ `string ] t

val unbatch
  :  ?name:string
  -> timeout_micros:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> 't t

val unbatchGrad
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> 't t

val unicodeDecode
  :  ?name:string
  -> input_encoding:string
  -> ?errors:string
  -> ?replacement_char:int
  -> ?replace_control_characters:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t * [ `int32 ] t

val unicodeDecodeWithOffsets
  :  ?name:string
  -> input_encoding:string
  -> ?errors:string
  -> ?replacement_char:int
  -> ?replace_control_characters:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t * [ `int32 ] t * [ `int64 ] t

val unicodeEncode
  :  ?name:string
  -> ?errors:string
  -> output_encoding:string
  -> ?replacement_char:int
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `int64 ] t
  -> [ `string ] t

val unicodeScript
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `int32 ] t

val unicodeTranscode
  :  ?name:string
  -> input_encoding:string
  -> output_encoding:string
  -> ?errors:string
  -> ?replacement_char:int
  -> ?replace_control_characters:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val uniformCandidateSampler
  :  ?name:string
  -> num_true:int
  -> num_sampled:int
  -> unique:bool
  -> range_max:int
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

val unique
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t * ([< `int32 | `int64 ] as 'out_idx) t

val uniqueV2
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'taxis) t
  -> 't t * ([< `int32 | `int64 ] as 'out_idx) t

val uniqueWithCounts
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> type_2:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t * ([< `int32 | `int64 ] as 'out_idx) t * ([< `int32 | `int64 ] as 'out_idx) t

val uniqueWithCountsV2
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> type_2:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'taxis) t
  -> 't t * ([< `int32 | `int64 ] as 'out_idx) t * ([< `int32 | `int64 ] as 'out_idx) t

val unpack
  :  ?name:string
  -> num:int
  -> ?axis:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t list

val unravelIndex
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `int32 | `int64 ] as 'tidx) t

val unsortedSegmentMax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `int32 | `int64 ] as 'tnumsegments) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val unsortedSegmentMin
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `int32 | `int64 ] as 'tnumsegments) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

val unsortedSegmentProd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `int32 | `int64 ] as 'tnumsegments) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val unsortedSegmentSum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `int32 | `int64 ] as 'tnumsegments) t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) t

val unwrapDatasetVariant
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t

val upperBound
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t
  -> ([< `int32 | `int64 ] as 'out_type) t

val variable
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val variableV2
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

val where
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 | `int64 | `bool ] as 't) t
  -> [ `int64 ] t

val wholeFileReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

val windowDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `bool ] t
  -> [ `variant ] t

val workerHeartbeat
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

val wrapDatasetVariant
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `variant ] t
  -> [ `variant ] t

val writeFile
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `unit ] t

val xdivy
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val xlogy
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

val zerosLike
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

val zeta
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

val zipDataset
  :  ?name:string
  -> output_types:Type.p list
  -> output_shapes:Dim.t list list
  -> ?control_inputs:Node.p list
  -> [ `variant ] t list
  -> [ `variant ] t

