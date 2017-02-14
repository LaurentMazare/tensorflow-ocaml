(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)
open Node

module Op_names : sig
  val abort : Op_name.t
  val abs : Op_name.t
  val accumulatorApplyGradient : Op_name.t
  val accumulatorNumAccumulated : Op_name.t
  val accumulatorSetGlobalStep : Op_name.t
  val accumulatorTakeGradient : Op_name.t
  val acos : Op_name.t
  val add : Op_name.t
  val addManySparseToTensorsMap : Op_name.t
  val addN : Op_name.t
  val addSparseToTensorsMap : Op_name.t
  val adjustContrast : Op_name.t
  val adjustContrastv2 : Op_name.t
  val adjustHue : Op_name.t
  val adjustSaturation : Op_name.t
  val all : Op_name.t
  val allCandidateSampler : Op_name.t
  val any : Op_name.t
  val applyAdadelta : Op_name.t
  val applyAdagrad : Op_name.t
  val applyAdagradDA : Op_name.t
  val applyAdam : Op_name.t
  val applyCenteredRMSProp : Op_name.t
  val applyFtrl : Op_name.t
  val applyGradientDescent : Op_name.t
  val applyMomentum : Op_name.t
  val applyProximalAdagrad : Op_name.t
  val applyProximalGradientDescent : Op_name.t
  val applyRMSProp : Op_name.t
  val argMax : Op_name.t
  val argMin : Op_name.t
  val asString : Op_name.t
  val asin : Op_name.t
  val assign : Op_name.t
  val assignAdd : Op_name.t
  val assignSub : Op_name.t
  val atan : Op_name.t
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
  val batchFFT : Op_name.t
  val batchFFT2D : Op_name.t
  val batchFFT3D : Op_name.t
  val batchIFFT : Op_name.t
  val batchIFFT2D : Op_name.t
  val batchIFFT3D : Op_name.t
  val batchMatMul : Op_name.t
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
  val betainc : Op_name.t
  val biasAdd : Op_name.t
  val biasAddGrad : Op_name.t
  val biasAddV1 : Op_name.t
  val bitcast : Op_name.t
  val broadcastArgs : Op_name.t
  val broadcastGradientArgs : Op_name.t
  val cTCGreedyDecoder : Op_name.t
  val cTCLoss : Op_name.t
  val cast : Op_name.t
  val ceil : Op_name.t
  val checkNumerics : Op_name.t
  val cholesky : Op_name.t
  val choleskyGrad : Op_name.t
  val complex : Op_name.t
  val complexAbs : Op_name.t
  val computeAccidentalHits : Op_name.t
  val concat : Op_name.t
  val concatOffset : Op_name.t
  val concatV2 : Op_name.t
  val conditionalAccumulator : Op_name.t
  val conj : Op_name.t
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
  val countUpTo : Op_name.t
  val cropAndResize : Op_name.t
  val cropAndResizeGradBoxes : Op_name.t
  val cropAndResizeGradImage : Op_name.t
  val cross : Op_name.t
  val cumprod : Op_name.t
  val cumsum : Op_name.t
  val debugIdentity : Op_name.t
  val debugNanCount : Op_name.t
  val debugNumericSummary : Op_name.t
  val decodeBase64 : Op_name.t
  val decodeJSONExample : Op_name.t
  val decodePng : Op_name.t
  val decodeRaw : Op_name.t
  val deleteSessionTensor : Op_name.t
  val denseToDenseSetOperation : Op_name.t
  val denseToSparseSetOperation : Op_name.t
  val depthToSpace : Op_name.t
  val depthwiseConv2dNative : Op_name.t
  val depthwiseConv2dNativeBackpropFilter : Op_name.t
  val depthwiseConv2dNativeBackpropInput : Op_name.t
  val dequantize : Op_name.t
  val deserializeManySparse : Op_name.t
  val destroyTemporaryVariable : Op_name.t
  val diag : Op_name.t
  val diagPart : Op_name.t
  val digamma : Op_name.t
  val dilation2D : Op_name.t
  val dilation2DBackpropFilter : Op_name.t
  val dilation2DBackpropInput : Op_name.t
  val div : Op_name.t
  val drawBoundingBoxes : Op_name.t
  val dynamicPartition : Op_name.t
  val dynamicStitch : Op_name.t
  val editDistance : Op_name.t
  val elu : Op_name.t
  val eluGrad : Op_name.t
  val encodeBase64 : Op_name.t
  val encodePng : Op_name.t
  val enter : Op_name.t
  val equal : Op_name.t
  val erf : Op_name.t
  val erfc : Op_name.t
  val exit : Op_name.t
  val exp : Op_name.t
  val expandDims : Op_name.t
  val expm1 : Op_name.t
  val extractGlimpse : Op_name.t
  val extractImagePatches : Op_name.t
  val fFT : Op_name.t
  val fFT2D : Op_name.t
  val fFT3D : Op_name.t
  val fIFOQueue : Op_name.t
  val fact : Op_name.t
  val fakeQuantWithMinMaxArgs : Op_name.t
  val fakeQuantWithMinMaxArgsGradient : Op_name.t
  val fakeQuantWithMinMaxVars : Op_name.t
  val fakeQuantWithMinMaxVarsGradient : Op_name.t
  val fakeQuantWithMinMaxVarsPerChannel : Op_name.t
  val fakeQuantWithMinMaxVarsPerChannelGradient : Op_name.t
  val fill : Op_name.t
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
  val fusedPadConv2D : Op_name.t
  val fusedResizeAndPadConv2D : Op_name.t
  val gather : Op_name.t
  val gatherNd : Op_name.t
  val getSessionHandle : Op_name.t
  val getSessionTensor : Op_name.t
  val greater : Op_name.t
  val greaterEqual : Op_name.t
  val hSVToRGB : Op_name.t
  val hashTable : Op_name.t
  val histogramSummary : Op_name.t
  val iFFT : Op_name.t
  val iFFT2D : Op_name.t
  val iFFT3D : Op_name.t
  val identity : Op_name.t
  val identityReader : Op_name.t
  val igamma : Op_name.t
  val igammac : Op_name.t
  val imag : Op_name.t
  val imageSummary : Op_name.t
  val immutableConst : Op_name.t
  val inTopK : Op_name.t
  val initializeTable : Op_name.t
  val initializeTableFromTextFile : Op_name.t
  val inv : Op_name.t
  val invGrad : Op_name.t
  val invertPermutation : Op_name.t
  val isFinite : Op_name.t
  val isInf : Op_name.t
  val isNan : Op_name.t
  val isVariableInitialized : Op_name.t
  val l2Loss : Op_name.t
  val lRN : Op_name.t
  val lRNGrad : Op_name.t
  val learnedUnigramCandidateSampler : Op_name.t
  val less : Op_name.t
  val lessEqual : Op_name.t
  val lgamma : Op_name.t
  val linSpace : Op_name.t
  val listDiff : Op_name.t
  val log : Op_name.t
  val log1p : Op_name.t
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
  val matMul : Op_name.t
  val matchingFiles : Op_name.t
  val matrixBandPart : Op_name.t
  val matrixDeterminant : Op_name.t
  val matrixDiag : Op_name.t
  val matrixDiagPart : Op_name.t
  val matrixInverse : Op_name.t
  val matrixSetDiag : Op_name.t
  val matrixSolve : Op_name.t
  val matrixSolveLs : Op_name.t
  val matrixTriangularSolve : Op_name.t
  val max : Op_name.t
  val maxPool : Op_name.t
  val maxPool3D : Op_name.t
  val maxPool3DGrad : Op_name.t
  val maxPoolGrad : Op_name.t
  val maxPoolGradWithArgmax : Op_name.t
  val maxPoolWithArgmax : Op_name.t
  val maximum : Op_name.t
  val mean : Op_name.t
  val merge : Op_name.t
  val mergeSummary : Op_name.t
  val mergeV2Checkpoints : Op_name.t
  val min : Op_name.t
  val minimum : Op_name.t
  val mirrorPad : Op_name.t
  val mirrorPadGrad : Op_name.t
  val mod_ : Op_name.t
  val mul : Op_name.t
  val multinomial : Op_name.t
  val mutableDenseHashTable : Op_name.t
  val mutableHashTable : Op_name.t
  val mutableHashTableOfTensors : Op_name.t
  val neg : Op_name.t
  val negTrain : Op_name.t
  val nextIteration : Op_name.t
  val noOp : Op_name.t
  val nonMaxSuppression : Op_name.t
  val notEqual : Op_name.t
  val oneHot : Op_name.t
  val pack : Op_name.t
  val pad : Op_name.t
  val paddingFIFOQueue : Op_name.t
  val parallelConcat : Op_name.t
  val parameterizedTruncatedNormal : Op_name.t
  val parseTensor : Op_name.t
  val placeholder : Op_name.t
  val placeholderV2 : Op_name.t
  val placeholderWithDefault : Op_name.t
  val polygamma : Op_name.t
  val pow : Op_name.t
  val preventGradient : Op_name.t
  val priorityQueue : Op_name.t
  val prod : Op_name.t
  val qr : Op_name.t
  val quantizeAndDequantize : Op_name.t
  val quantizeDownAndShrinkRange : Op_name.t
  val quantizeV2 : Op_name.t
  val quantizedAvgPool : Op_name.t
  val quantizedBatchNormWithGlobalNormalization : Op_name.t
  val quantizedBiasAdd : Op_name.t
  val quantizedConcat : Op_name.t
  val quantizedConv2D : Op_name.t
  val quantizedInstanceNorm : Op_name.t
  val quantizedMatMul : Op_name.t
  val quantizedMaxPool : Op_name.t
  val quantizedRelu : Op_name.t
  val quantizedRelu6 : Op_name.t
  val quantizedReluX : Op_name.t
  val quantizedReshape : Op_name.t
  val queueClose : Op_name.t
  val queueSize : Op_name.t
  val rGBToHSV : Op_name.t
  val randomCrop : Op_name.t
  val randomGamma : Op_name.t
  val randomShuffle : Op_name.t
  val randomShuffleQueue : Op_name.t
  val randomStandardNormal : Op_name.t
  val randomUniform : Op_name.t
  val randomUniformInt : Op_name.t
  val range : Op_name.t
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
  val reduceJoin : Op_name.t
  val refEnter : Op_name.t
  val refExit : Op_name.t
  val refIdentity : Op_name.t
  val refMerge : Op_name.t
  val refNextIteration : Op_name.t
  val refSelect : Op_name.t
  val refSwitch : Op_name.t
  val relu : Op_name.t
  val relu6 : Op_name.t
  val relu6Grad : Op_name.t
  val reluGrad : Op_name.t
  val requantizationRange : Op_name.t
  val requantize : Op_name.t
  val reshape : Op_name.t
  val resizeArea : Op_name.t
  val resizeBicubic : Op_name.t
  val resizeBilinear : Op_name.t
  val resizeBilinearGrad : Op_name.t
  val resizeNearestNeighbor : Op_name.t
  val resizeNearestNeighborGrad : Op_name.t
  val restore : Op_name.t
  val restoreSlice : Op_name.t
  val reverse : Op_name.t
  val reverseSequence : Op_name.t
  val reverseV2 : Op_name.t
  val rint : Op_name.t
  val round : Op_name.t
  val rsqrt : Op_name.t
  val rsqrtGrad : Op_name.t
  val sampleDistortedBoundingBox : Op_name.t
  val scalarSummary : Op_name.t
  val scatterAdd : Op_name.t
  val scatterDiv : Op_name.t
  val scatterMul : Op_name.t
  val scatterNd : Op_name.t
  val scatterNdAdd : Op_name.t
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
  val serializeManySparse : Op_name.t
  val serializeSparse : Op_name.t
  val setSize : Op_name.t
  val shape : Op_name.t
  val shapeN : Op_name.t
  val shardedFilename : Op_name.t
  val shardedFilespec : Op_name.t
  val sigmoid : Op_name.t
  val sigmoidGrad : Op_name.t
  val sign : Op_name.t
  val sin : Op_name.t
  val size : Op_name.t
  val skipgram : Op_name.t
  val slice : Op_name.t
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
  val sparseApplyMomentum : Op_name.t
  val sparseApplyProximalAdagrad : Op_name.t
  val sparseApplyProximalGradientDescent : Op_name.t
  val sparseApplyRMSProp : Op_name.t
  val sparseConcat : Op_name.t
  val sparseConditionalAccumulator : Op_name.t
  val sparseDenseCwiseAdd : Op_name.t
  val sparseDenseCwiseDiv : Op_name.t
  val sparseDenseCwiseMul : Op_name.t
  val sparseMatMul : Op_name.t
  val sparseReduceSum : Op_name.t
  val sparseReduceSumSparse : Op_name.t
  val sparseReorder : Op_name.t
  val sparseReshape : Op_name.t
  val sparseSegmentMean : Op_name.t
  val sparseSegmentMeanGrad : Op_name.t
  val sparseSegmentSqrtN : Op_name.t
  val sparseSegmentSqrtNGrad : Op_name.t
  val sparseSegmentSum : Op_name.t
  val sparseSoftmax : Op_name.t
  val sparseSoftmaxCrossEntropyWithLogits : Op_name.t
  val sparseSparseMaximum : Op_name.t
  val sparseSparseMinimum : Op_name.t
  val sparseTensorDenseAdd : Op_name.t
  val sparseTensorDenseMatMul : Op_name.t
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
  val stopGradient : Op_name.t
  val stridedSlice : Op_name.t
  val stridedSliceAssign : Op_name.t
  val stridedSliceGrad : Op_name.t
  val stringJoin : Op_name.t
  val stringSplit : Op_name.t
  val stringToHashBucket : Op_name.t
  val stringToHashBucketFast : Op_name.t
  val stringToHashBucketStrong : Op_name.t
  val stringToNumber : Op_name.t
  val sub : Op_name.t
  val substr : Op_name.t
  val sum : Op_name.t
  val svd : Op_name.t
  val switch : Op_name.t
  val tFRecordReader : Op_name.t
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
  val tensorSummary : Op_name.t
  val textLineReader : Op_name.t
  val threadUnsafeUnigramCandidateSampler : Op_name.t
  val tile : Op_name.t
  val tileGrad : Op_name.t
  val topK : Op_name.t
  val topKV2 : Op_name.t
  val transpose : Op_name.t
  val truncateDiv : Op_name.t
  val truncateMod : Op_name.t
  val truncatedNormal : Op_name.t
  val uniformCandidateSampler : Op_name.t
  val unique : Op_name.t
  val uniqueWithCounts : Op_name.t
  val unpack : Op_name.t
  val unsortedSegmentSum : Op_name.t
  val variable : Op_name.t
  val variableV2 : Op_name.t
  val where : Op_name.t
  val wholeFileReader : Op_name.t
  val writeFile : Op_name.t
  val zerosLike : Op_name.t
  val zeta : Op_name.t
end

(* Raise a exception to abort the process when called. If exit_without_error is true, the process will exit normally, otherwise it will exit with a SIGABORT signal. *)
(* Returns nothing but an exception. *)
val abort
  :  ?name:string
  -> ?error_msg:string
  -> ?exit_without_error:bool
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

(* Computes the absolute value of a tensor. *)
(* Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\). *)
val abs
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Applies a gradient to a given accumulator. Does not add if local_step is lesser *)
(* than the accumulator's global_step. *)
val accumulatorApplyGradient
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 'dtype) t
  -> [ `unit ] t

(* Returns the number of gradients aggregated in the given accumulators. *)
val accumulatorNumAccumulated
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t

(* Updates the accumulator with a new value for global_step. Logs warning if the *)
(* accumulator's value is already higher than new_global_step. *)
val accumulatorSetGlobalStep
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `unit ] t

(* Extracts the average gradient in the given ConditionalAccumulator, provided *)
(* that sufficient (i.e., more than num_required) gradients have been accumulated.
The op blocks until sufficient gradients have been accumulated.
If the accumulator has already aggregated more than num_required gradients, it
returns the average of the accumulated gradients.
Also automatically increments the recorded global_step in the accumulator by 1,
and resets the aggregate to 0. *)
val accumulatorTakeGradient
  :  ?name:string
  -> type_:([< `float | `double | `int64 | `int32 | `complex64 ] as 'dtype) Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 'dtype) t

(* Computes acos of x element-wise. *)
val acos
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Returns x + y element-wise. *)
(* *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val add
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t

(* Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles. *)
(* A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
`sparse_values`, and `sparse_shape`, where

```sparse_indices.shape[1] == sparse_shape.shape[0] == R```

An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
having a first `sparse_indices` column taking values between `[0, N)`, where
the minibatch size `N == sparse_shape[0]`.

The input `SparseTensor` must have rank `R` greater than 1, and the first
dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The stored
`SparseTensor` objects pointed to by each row of the output `sparse_handles`
will have rank `R-1`.

The `SparseTensor` values can then be read out as part of a minibatch by passing
the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
the correct `SparseTensorsMap` is accessed, ensure that the same
`container` and `shared_name` are passed to that Op.  If no `shared_name`
is provided here, instead use the *name* of the Operation created by calling
`AddManySparseToTensorsMap` as the `shared_name` passed to
`TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated. *)
val addManySparseToTensorsMap
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t

(* Add all input tensors element wise. *)
val addN
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Add a `SparseTensor` to a `SparseTensorsMap` return its handle. *)
(* A `SparseTensor` is represented by three tensors: `sparse_indices`,
`sparse_values`, and `sparse_shape`.

This operator takes the given `SparseTensor` and adds it to a container
object (a `SparseTensorsMap`).  A unique key within this container is generated
in the form of an `int64`, and this is the value that is returned.

The `SparseTensor` can then be read out as part of a minibatch by passing
the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
the correct `SparseTensorsMap` is accessed, ensure that the same
`container` and `shared_name` are passed to that Op.  If no `shared_name`
is provided here, instead use the *name* of the Operation created by calling
`AddSparseToTensorsMap` as the `shared_name` passed to
`TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated. *)
val addSparseToTensorsMap
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t

(* Deprecated. Disallowed in GraphDef version >= 2. *)
val adjustContrast
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Adjust the contrast of one or more images. *)
(* `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`

Contrast is adjusted independently for each channel of each image.

For each channel, the Op first computes the mean of the image pixels in the
channel and then adjusts each component of each pixel to
`(x - mean) * contrast_factor + mean`. *)
val adjustContrastv2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Adjust the hue of one or more images. *)
(* `images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.

The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A delta is then applied all the hue values,
and then remapped back to RGB colorspace. *)
val adjustHue
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Adjust the saturation of one or more images. *)
(* `images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.

The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A scale is then applied all the saturation
values, and then remapped back to RGB colorspace. *)
val adjustSaturation
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Computes the 'logical and' of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val all
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `bool ] t

(* Generates labels for candidate sampling with a learned unigram distribution. *)
(* See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels. *)
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

(* Computes the 'logical or' of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val any
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `bool ] t

(* Update '*var' according to the adadelta scheme. *)
(* accum = rho() * accum + (1 - rho()) * grad.square();
update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
update_accum = rho() * update_accum + (1 - rho()) * update.square();
var -= update; *)
val applyAdadelta
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the adagrad scheme. *)
(* accum += grad * grad
var -= lr * grad * (1 / sqrt(accum)) *)
val applyAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the proximal adagrad scheme. *)
val applyAdagradDA
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the Adam algorithm. *)
(* lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon) *)
val applyAdam
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the centered RMSProp algorithm. *)
(* The centered RMSProp algorithm uses an estimate of the centered second moment
(i.e., the variance) for normalization, as opposed to regular RMSProp, which
uses the (uncentered) second moment. This often helps with training, but is
slightly more expensive in terms of computation and memory.

Note that in dense implementation of this algorithm, mg, ms, and mom will
update even if the grad is zero, but in this sparse implementation, mg, ms,
and mom will not update in iterations during which the grad is zero.

mean_square = decay * mean_square + (1-decay) * gradient ** 2
mean_grad = decay * mean_grad + (1-decay) * gradient

Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

mg <- rho * mg_{t-1} + (1-rho) * grad
ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
var <- var - mom *)
val applyCenteredRMSProp
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the Ftrl-proximal scheme. *)
(* accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new *)
val applyFtrl
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' by subtracting 'alpha' * 'delta' from it. *)
val applyGradientDescent
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the momentum scheme. Set use_nesterov = True if you *)
(* want to use Nesterov momentum.

accum = accum * momentum + grad
var -= lr * accum *)
val applyMomentum
  :  ?name:string
  -> ?use_locking:bool
  -> ?use_nesterov:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' and '*accum' according to FOBOS with Adagrad learning rate. *)
(* accum += grad * grad
prox_v = var - lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{ |prox_v|-lr*l1,0} *)
val applyProximalAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' as FOBOS algorithm with fixed learning rate. *)
(* prox_v = var - alpha * delta
var = sign(prox_v)/(1+alpha*l2) * max{ |prox_v|-alpha*l1,0} *)
val applyProximalGradientDescent
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the RMSProp algorithm. *)
(* Note that in dense implementation of this algorithm, ms and mom will
update even if the grad is zero, but in this sparse implementation, ms
and mom will not update in iterations during which the grad is zero.

mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom *)
val applyRMSProp
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Returns the index with the largest value across dimensions of a tensor. *)
val argMax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int64 ] t

(* Returns the index with the smallest value across dimensions of a tensor. *)
val argMin
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int64 ] t

(* Converts each entry in the given tensor to strings.  Supports many numeric *)
(* types and boolean. *)
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

(* Computes asin of x element-wise. *)
val asin
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Update 'ref' by assigning 'value' to it. *)
(* This operation outputs 'ref' after the assignment is done.
This makes it easier to chain operations that need to use the reset value. *)
val assign
  :  ?name:string
  -> ?validate_shape:bool
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t
  -> 't t

(* Update 'ref' by adding 'value' to it. *)
(* This operation outputs 'ref' after the update is done.
This makes it easier to chain operations that need to use the reset value. *)
val assignAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update 'ref' by subtracting 'value' from it. *)
(* This operation outputs 'ref' after the update is done.
This makes it easier to chain operations that need to use the reset value. *)
val assignSub
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes atan of x element-wise. *)
val atan
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Outputs a `Summary` protocol buffer with audio. *)
(* The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc. *)
val audioSummary
  :  ?name:string
  -> sample_rate:float
  -> ?max_outputs:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> [ `string ] t

(* Outputs a `Summary` protocol buffer with audio. *)
(* The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc. *)
val audioSummaryV2
  :  ?name:string
  -> ?max_outputs:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `string ] t

(* Performs average pooling on the input. *)
(* Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. *)
val avgPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Performs 3D average pooling on the input. *)
val avgPool3D
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes gradients of average pooling function. *)
val avgPool3DGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes gradients of the average pooling function. *)
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

(* Defines a barrier that persists across different graph executions. *)
(* A barrier represents a key-value map, where each key is a string, and
each value is a tuple of tensors.

At runtime, the barrier contains 'complete' and 'incomplete'
elements. A complete element has defined tensors for all components of
its value tuple, and may be accessed using BarrierTakeMany. An
incomplete element has some undefined components in its value tuple,
and may be updated using BarrierInsertMany. *)
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

(* Closes the given barrier. *)
(* This operation signals that no more new elements will be inserted in the
given barrier. Subsequent InsertMany that try to introduce a new key will fail.
Subsequent InsertMany operations that just add missing components to already
existing elements will continue to succeed. Subsequent TakeMany operations will
continue to succeed if sufficient completed elements remain in the barrier.
Subsequent TakeMany operations that would block will fail immediately. *)
val barrierClose
  :  ?name:string
  -> ?cancel_pending_enqueues:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

(* Computes the number of incomplete elements in the given barrier. *)
val barrierIncompleteSize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t

(* For each key, assigns the respective value to the specified component. *)
(* If a key is not found in the barrier, this operation will create a new
incomplete element. If a key is found in the barrier, and the element
already has a value at component_index, this operation will fail with
INVALID_ARGUMENT, and leave the barrier in an undefined state. *)
val barrierInsertMany
  :  ?name:string
  -> component_index:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> 't t
  -> [ `unit ] t

(* Computes the number of complete elements in the given barrier. *)
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

(* Multiplies slices of two tensors in batches. *)
(* Multiplies all slices of `Tensor` `x` and `y` (each slice can be
viewed as an element of a batch), and arranges the individual results
in a single output tensor of the same batch size. Each of the
individual slices can optionally be adjointed (to adjoint a matrix
means to transpose and conjugate it) before multiplication by setting
the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
and `[..., r_y, c_y]`.

The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:

    r_o = c_x if adj_x else r_x
    c_o = r_y if adj_y else c_y

It is computed as:

    output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :]) *)
val batchMatMul
  :  ?name:string
  -> ?adj_x:bool
  -> ?adj_y:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t

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
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

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

(* Batch normalization. *)
(* This op is deprecated. Prefer `tf.nn.batch_normalization`. *)
val batchNormWithGlobalNormalization
  :  ?name:string
  -> variance_epsilon:float
  -> scale_after_normalization:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Gradients for batch normalization. *)
(* This op is deprecated. See `tf.nn.batch_normalization`. *)
val batchNormWithGlobalNormalizationGrad
  :  ?name:string
  -> variance_epsilon:float
  -> scale_after_normalization:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

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

(* BatchToSpace for 4-D tensors of type T. *)
(* This is a legacy version of the more general BatchToSpaceND.

Rearranges (permutes) data from batch into blocks of spatial data, followed by
cropping. This is the reverse transformation of SpaceToBatch. More specifically,
this op outputs a copy of the input tensor where values from the `batch`
dimension are moved in spatial blocks to the `height` and `width` dimensions,
followed by cropping along the `height` and `width` dimensions. *)
val batchToSpace
  :  ?name:string
  -> block_size:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> 't t

(* BatchToSpace for N-D tensors of type T. *)
(* This operation reshapes the 'batch' dimension 0 into `M + 1` dimensions of shape
`block_shape + [batch]`, interleaves these blocks back into the grid defined by
the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
the input.  The spatial dimensions of this intermediate result are then
optionally cropped according to `crops` to produce the output.  This is the
reverse of SpaceToBatch.  See below for a precise description. *)
val batchToSpaceND
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tblock_shape) t
  -> ([< `int32 | `int64 ] as 'tcrops) t
  -> 't t

(* Compute the regularized incomplete beta integral \\(I_x(a, b)\\). *)
(* The regularized incomplete beta integral is defined as:

```
I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}
```
where

```
B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt
```

is the incomplete beta function and \\(B(a, b)\\) is the *complete*
beta function. *)
val betainc
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Adds `bias` to `value`. *)
(* This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions. *)
val biasAdd
  :  ?name:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* The backward operation for 'BiasAdd' on the 'bias' tensor. *)
(* It accumulates all the values from out_backprop into the feature dimension.
For NHWC data format, the feature dimension is the last. For NCHW data format,
the feature dimension is the third-to-last. *)
val biasAddGrad
  :  ?name:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Adds `bias` to `value`. *)
(* This is a deprecated version of BiasAdd and will be soon removed.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions. *)
val biasAddV1
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Bitcasts a tensor from one type to another without copying data. *)
(* Given a tensor `input`, this operation returns a tensor that has the same buffer
data as `input` with datatype `type`.

If the input datatype `T` is larger than the output datatype `type` then the
shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].

If `T` is smaller than `type`, the operator requires that the rightmost
dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
[..., sizeof(`type`)/sizeof(`T`)] to [...].

*NOTE*: Bitcast is implemented as a low-level cast, so machines with different
endian orderings will give different results. *)
val bitcast
  :  ?name:string
  -> type_:([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) t

(* Return the shape of s0 op s1 with broadcast. *)
(* Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
broadcasted shape. `s0`, `s1` and `r0` are all integer vectors. *)
val broadcastArgs
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

(* Return the reduction indices for computing gradients of s0 op s1 with broadcast. *)
(* This is typically used by gradient computations for a broadcasting operation. *)
val broadcastGradientArgs
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t * ([< `int32 | `int64 ] as 't) t

(* Performs greedy decoding on the logits given in inputs. *)
(* A note about the attribute merge_repeated: if enabled, when
consecutive logits' maximum indices are the same, only the first of
these is emitted.  Labeling the blank '*', the sequence 'A B B * B B'
becomes 'A B' if merge_repeated = True and 'A B B B B' if
merge_repeated = False.

Regardless of the value of merge_repeated, if the maximum index of a given
time and batch corresponds to the blank, index `(num_classes - 1)`, no new
element is emitted. *)
val cTCGreedyDecoder
  :  ?name:string
  -> ?merge_repeated:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `int64 ] t * [ `int64 ] t * [ `int64 ] t * [ `float ] t

(* Calculates the CTC Loss (log probability) for each batch entry.  Also calculates *)
(* the gradient.  This class performs the softmax operation for you, so inputs
should be e.g. linear projections of outputs by an LSTM. *)
val cTCLoss
  :  ?name:string
  -> ?preprocess_collapse_repeated:bool
  -> ?ctc_merge_repeated:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `float ] t * [ `float ] t

(* Cast x of type SrcT to y of DstT. *)
val cast
  :  ?name:string
  -> type_:'dstT Type.t
  -> ?control_inputs:Node.p list
  -> 'srcT t
  -> 'dstT t

(* Returns element-wise smallest integer in not less than x. *)
val ceil
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Checks a tensor for NaN and Inf values. *)
(* When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is. *)
val checkNumerics
  :  ?name:string
  -> message:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the Cholesky decomposition of one or more square matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix Cholesky
decomposition above. The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`. *)
val cholesky
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Computes the reverse mode backpropagated gradient of the Cholesky algorithm. *)
(* For an explanation see 'Differentiation of the Cholesky algorithm' by
Iain Murray http://arxiv.org/abs/1602.07527. *)
val choleskyGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Converts two real numbers to a complex number. *)
(* Given a tensor `real` representing the real part of a complex number, and a
tensor `imag` representing the imaginary part of a complex number, this
operation returns complex numbers elementwise of the form \\(a + bj\\), where
*a* represents the `real` part and *b* represents the `imag` part.

The input tensors `real` and `imag` must have the same shape.

For example:

```
# tensor 'real' is [2.25, 3.25]
# tensor `imag` is [4.75, 5.75]
tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
``` *)
val complex
  :  ?name:string
  -> type_:([< `complex64 ] as 'tout) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `complex64 ] as 'tout) t

(* Computes the complex absolute value of a tensor. *)
(* Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` or `double` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
value is computed as \\( \sqrt{a^2 + b^2}\\). *)
val complexAbs
  :  ?name:string
  -> type_:([< `float | `double ] as 'tout) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 't) t
  -> ([< `float | `double ] as 'tout) t

(* Computes the ids of the positions in sampled_candidates that match true_labels. *)
(* When doing log-odds NCE, the result of this op should be passed through a
SparseToDense op, then added to the logits of the sampled candidates. This has
the effect of 'removing' the sampled labels that match the true labels by
making the classifier sure that they are sampled labels. *)
val computeAccidentalHits
  :  ?name:string
  -> num_true:int
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int32 ] t * [ `int64 ] t * [ `float ] t

(* Concatenates tensors along one dimension. *)
val concat
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 't t list
  -> 't t

(* Computes offsets of concat inputs within its output. *)
(* For example:

```prettyprint
# 'x' is [2, 2, 7]
# 'y' is [2, 3, 7]
# 'z' is [2, 5, 7]
concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
``` *)
val concatOffset
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> [ `int32 ] t list
  -> [ `int32 ] t list

(* Concatenates tensors along one dimension. *)
val concatV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t list
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> 't t

(* A conditional accumulator for aggregating gradients. The accumulator accepts *)
(* gradients marked with local_step greater or equal to the most recent global_step
known to the accumulator. The average can be extracted from the accumulator,
provided sufficient gradients have been accumulated. Extracting the average
automatically resets the aggregate to 0, and increments the global_step recorded
by the accumulator. *)
val conditionalAccumulator
  :  ?name:string
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Returns the complex conjugate of a complex number. *)
(* Given a tensor `input` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `input`. The
complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
real part and *b* is the imaginary part.

The complex conjugate returned by this operation is of the form \\(a - bj\\).

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
``` *)
val conj
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 't) t
  -> ([< `complex64 ] as 't) t

(* Does nothing. Serves as a control trigger for scheduling. *)
(* Only useful as a placeholder for control edges. *)
val controlTrigger
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

(* Computes a 2-D convolution given 4-D `input` and `filter` tensors. *)
(* Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:

1. Flattens the filter to a 2-D matrix with shape
   `[filter_height * filter_width * in_channels, output_channels]`.
2. Extracts image patches from the input tensor to form a *virtual*
   tensor of shape `[batch, out_height, out_width,
   filter_height * filter_width * in_channels]`.
3. For each patch, right-multiplies the filter matrix and the image patch
   vector.

In detail, with the default NHWC format,

    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                        filter[di, dj, q, k]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`. *)
val conv2D
  :  ?name:string
  -> strides:int list
  -> ?use_cudnn_on_gpu:bool
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the gradients of convolution with respect to the filter. *)
val conv2DBackpropFilter
  :  ?name:string
  -> strides:int list
  -> ?use_cudnn_on_gpu:bool
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the gradients of convolution with respect to the input. *)
val conv2DBackpropInput
  :  ?name:string
  -> strides:int list
  -> ?use_cudnn_on_gpu:bool
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes a 3-D convolution given 5-D `input` and `filter` tensors. *)
(* In signal processing, cross-correlation is a measure of similarity of
two waveforms as a function of a time-lag applied to one of them. This
is also known as a sliding dot product or sliding inner-product.

Our Conv3D implements a form of cross-correlation. *)
val conv3D
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the gradients of 3-D convolution with respect to the filter. *)
val conv3DBackpropFilter
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the gradients of 3-D convolution with respect to the filter. *)
val conv3DBackpropFilterV2
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the gradients of 3-D convolution with respect to the input. *)
val conv3DBackpropInput
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the gradients of 3-D convolution with respect to the input. *)
val conv3DBackpropInputV2
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Copy Op. *)
(* Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
device on which the tensor is allocated.

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

Unlike the Copy Op, this op has HostMemory constraint on its input or output. *)
val copyHost
  :  ?name:string
  -> ?tensor_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Computes cos of x element-wise. *)
val cos
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Increments 'ref' until it reaches 'limit'. *)
val countUpTo
  :  ?name:string
  -> limit:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

(* Extracts crops from the input image tensor and bilinearly resizes them (possibly *)
(* with aspect ratio change) to a common output size specified by `crop_size`. This
is more general than the `crop_to_bounding_box` op which extracts a fixed size
slice from the input image and does not allow resizing or aspect ratio change.

Returns a tensor with `crops` from the input `image` at positions defined at the
bounding box locations in `boxes`. The cropped boxes are all resized (with
bilinear interpolation) to a fixed `size = [crop_height, crop_width]`. The
result is a 4-D tensor `[num_boxes, crop_height, crop_width, depth]`. *)
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

(* Computes the gradient of the crop_and_resize op wrt the input boxes tensor. *)
val cropAndResizeGradBoxes
  :  ?name:string
  -> ?method_:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `float ] t

(* Computes the gradient of the crop_and_resize op wrt the input image tensor. *)
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

(* Compute the pairwise cross product. *)
(* `a` and `b` must be the same shape; they can either be simple 3-element vectors,
or any shape where the innermost dimension is 3. In the latter case, each pair
of corresponding 3-element vectors is cross-multiplied independently. *)
val cross
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Compute the cumulative product of the tensor `x` along `axis`. *)
(* By default, this op performs an inclusive cumprod, which means that the first
element of the input is identical to the first element of the output:
```prettyprint
tf.cumprod([a, b, c]) ==> [a, a * b, a * b * c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
performed instead:
```prettyprint
tf.cumprod([a, b, c], exclusive=True) ==> [0, a, a * b]
```

By setting the `reverse` kwarg to `True`, the cumprod is performed in the
opposite direction:
```prettyprint
tf.cumprod([a, b, c], reverse=True) ==> [a * b * c, b * c, c]
```
This is more efficient than using separate `tf.reverse` ops.

The `reverse` and `exclusive` kwargs can also be combined:
```prettyprint
tf.cumprod([a, b, c], exclusive=True, reverse=True) ==> [b * c, c, 0]
``` *)
val cumprod
  :  ?name:string
  -> ?exclusive:bool
  -> ?reverse:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Compute the cumulative sum of the tensor `x` along `axis`. *)
(* By default, this op performs an inclusive cumsum, which means that the first
element of the input is identical to the first element of the output:
```prettyprint
tf.cumsum([a, b, c]) ==> [a, a + b, a + b + c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
performed instead:
```prettyprint
tf.cumsum([a, b, c], exclusive=True) ==> [0, a, a + b]
```

By setting the `reverse` kwarg to `True`, the cumsum is performed in the
opposite direction:
```prettyprint
tf.cumsum([a, b, c], reverse=True) ==> [a + b + c, b + c, c]
```
This is more efficient than using separate `tf.reverse` ops.

The `reverse` and `exclusive` kwargs can also be combined:
```prettyprint
tf.cumsum([a, b, c], exclusive=True, reverse=True) ==> [b + c, c, 0]
``` *)
val cumsum
  :  ?name:string
  -> ?exclusive:bool
  -> ?reverse:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Debug Identity Op. *)
(* Provides an identity mapping of the non-Ref type input tensor for debugging. *)
val debugIdentity
  :  ?name:string
  -> ?tensor_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Debug NaN Value Counter Op *)
(* Counts number of NaNs in the input tensor, for debugging. *)
val debugNanCount
  :  ?name:string
  -> ?tensor_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int64 ] t

(* Debug Numeric Summary Op. *)
(* Provide a basic summary of numeric value types, range and distribution. *)
val debugNumericSummary
  :  ?name:string
  -> ?tensor_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `double ] t

(* Decode web-safe base64-encoded strings. *)
(* Input may or may not have padding at the end. See EncodeBase64 for padding.
Web-safe means that input must use - and _ instead of + and /. *)
val decodeBase64
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

(* Convert JSON-encoded Example records to binary protocol buffer strings. *)
(* This op translates a tensor containing Example records, encoded using
the [standard JSON
mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
into a tensor containing the same records encoded as binary protocol
buffers. The resulting tensor can then be fed to any of the other
Example-parsing ops. *)
val decodeJSONExample
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

(* Decode a PNG-encoded image to a uint8 or uint16 tensor. *)
(* The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the PNG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.
*   4: output an RGBA image.

If needed, the PNG-encoded image is transformed to match the requested number
of color channels. *)
val decodePng
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?channels:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'dtype t

(* Reinterpret the bytes of a string as a vector of numbers. *)
val decodeRaw
  :  ?name:string
  -> type_:([< `float | `double | `int32 | `int64 ] as 'out_type) Type.t
  -> ?little_endian:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float | `double | `int32 | `int64 ] as 'out_type) t

(* Delete the tensor specified by its handle in the session. *)
val deleteSessionTensor
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

(* Applies set operation along last dimension of 2 `Tensor` inputs. *)
(* See SetOperationOp::SetOperationFromContext for values of `set_operation`.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`. *)
val denseToDenseSetOperation
  :  ?name:string
  -> set_operation:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> [ `int64 ] t * ([< `int32 | `int64 | `string ] as 't) t * [ `int64 ] t

(* Applies set operation along last dimension of `Tensor` and `SparseTensor`. *)
(* See SetOperationOp::SetOperationFromContext for values of `set_operation`.

Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

If `validate_indices` is `True`, this op validates the order and range of `set2`
indices.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`. *)
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

(* DepthToSpace for tensors of type T. *)
(* Rearranges data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically,
this op outputs a copy of the input tensor where values from the `depth`
dimension are moved in spatial blocks to the `height` and `width` dimensions.
The attr `block_size` indicates the input block size and how the data is moved.

  * Chunks of data of size `block_size * block_size` from depth are rearranged
    into non-overlapping blocks of size `block_size x block_size`
  * The width the output tensor is `input_depth * block_size`, whereas the
    height is `input_height * block_size`.
  * The depth of the input tensor must be divisible by
    `block_size * block_size`.

That is, assuming the input is in the shape:
`[batch, height, width, depth]`,
the shape of the output will be:
`[batch, height*block_size, width*block_size, depth/(block_size*block_size)]`

This operation requires that the input tensor be of rank 4, and that
`block_size` be >=1 and that `block_size * block_size` be a divisor of the
input depth.

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given this input of shape `[1, 1, 1, 4]`, and a block size of 2:

```prettyprint
x = [[[[1, 2, 3, 4]]]]

```

This operation will output a tensor of shape `[1, 2, 2, 1]`:

```prettyprint
   [[[[1], [2]],
     [[3], [4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
the corresponding output will have 2x2 elements and will have a depth of
1 channel (1 = `4 / (block_size * block_size)`).
The output element shape is `[2, 2, 1]`.

For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.

```prettyprint
x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

This operation, for block size of 2, will return the following tensor of shape
`[1, 2, 2, 3]`

```prettyprint
   [[[[1, 2, 3], [4, 5, 6]],
     [[7, 8, 9], [10, 11, 12]]]]

```

Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:

```prettyprint
x =  [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```

the operator will return the following tensor of shape `[1 4 4 1]`:

```prettyprint
x = [[ [1],   [2],  [5],  [6]],
     [ [3],   [4],  [7],  [8]],
     [ [9],  [10], [13],  [14]],
     [ [11], [12], [15],  [16]]]

``` *)
val depthToSpace
  :  ?name:string
  -> block_size:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors. *)
(* Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, channel_multiplier]`, containing
`in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
a different filter to each input channel (expanding from 1 channel to
`channel_multiplier` channels for each), then concatenates the results
together. Thus, the output has `in_channels * channel_multiplier` channels.

for k in 0..in_channels-1
  for q in 0..channel_multiplier-1
    output[b, i, j, k * channel_multiplier + q] =
      sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                        filter[di, dj, k, q]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`. *)
val depthwiseConv2dNative
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the gradients of depthwise convolution with respect to the filter. *)
val depthwiseConv2dNativeBackpropFilter
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the gradients of depthwise convolution with respect to the input. *)
val depthwiseConv2dNativeBackpropInput
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Dequantize the 'input' tensor into a float Tensor. *)
(* [min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
if T == qint8, in[i] += (range(T) + 1)/ 2.0
out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
```
here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

*MIN_COMBINED Mode Example*

If the input comes from a QuantizedRelu6, the output type is
quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
Dequantize on quint8 will take each value, cast to float, and multiply
by 6 / 255.
Note that if quantizedtype is qint8, the operation will additionally add
each value by 128 prior to casting.

If the mode is 'MIN_FIRST', then this approach is used:

```
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = range / number_of_steps
const double offset_input = static_cast<double>(input) - lowest_quantized;
result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
``` *)
val dequantize
  :  ?name:string
  -> ?mode:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Deserialize and concatenate `SparseTensors` from a serialized minibatch. *)
(* The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
`N` is the minibatch size and the rows correspond to packed outputs of
`SerializeSparse`.  The ranks of the original `SparseTensor` objects
must all match.  When the final `SparseTensor` is created, it has rank one
higher than the ranks of the incoming `SparseTensor` objects
(they have been concatenated along a new row dimension).

The output `SparseTensor` object's shape values for all dimensions but the
first are the max across the input `SparseTensor` objects' shape values
for the corresponding dimensions.  Its first shape value is `N`, the minibatch
size.

The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.

For example, if the serialized input is a `[2 x 3]` matrix representing two
original `SparseTensor` objects:

    index = [ 0]
            [10]
            [20]
    values = [1, 2, 3]
    shape = [50]

and

    index = [ 2]
            [10]
    values = [4, 5]
    shape = [30]

then the final deserialized `SparseTensor` will be:

    index = [0  0]
            [0 10]
            [0 20]
            [1  2]
            [1 10]
    values = [1, 2, 3, 4, 5]
    shape = [2 50] *)
val deserializeManySparse
  :  ?name:string
  -> type_1:'dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t * 'dtype t * [ `int64 ] t

(* Destroys the temporary variable and returns its final value. *)
(* Sets output to the value of the Tensor pointed to by 'ref', then destroys
the temporary variable called 'var_name'.
All other uses of 'ref' *must* have executed before this op.
This is typically achieved by chaining the ref through each assign op, or by
using control dependencies.

Outputs the final value of the tensor pointed to by 'ref'. *)
val destroyTemporaryVariable
  :  ?name:string
  -> var_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Returns a diagonal tensor with a given diagonal values. *)
(* Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:

`output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

For example:

```prettyprint
# 'diagonal' is [1, 2, 3, 4]
tf.diag(diagonal) ==> [[1, 0, 0, 0]
                       [0, 2, 0, 0]
                       [0, 0, 3, 0]
                       [0, 0, 0, 4]]
``` *)
val diag
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Returns the diagonal part of the tensor. *)
(* This operation returns a tensor with the `diagonal` part
of the `input`. The `diagonal` part is computed as follows:

Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
tensor of rank `k` with dimensions `[D1,..., Dk]` where:

`diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

For example:

```prettyprint
# 'input' is [[1, 0, 0, 0]
              [0, 2, 0, 0]
              [0, 0, 3, 0]
              [0, 0, 0, 4]]

tf.diag_part(input) ==> [1, 2, 3, 4]
``` *)
val diagPart
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes Psi, the derivative of Lgamma (the log of the absolute value of *)
(* `Gamma(x)`), element-wise. *)
val digamma
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors. *)
(* The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
`filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
input channel is processed independently of the others with its own structuring
function. The `output` tensor has shape
`[batch, out_height, out_width, depth]`. The spatial dimensions of the output
tensor depend on the `padding` algorithm. We currently only support the default
'NHWC' `data_format`.

In detail, the grayscale morphological 2-D dilation is the max-sum correlation
(for consistency with `conv2d`, we use unmirrored filters):

    output[b, y, x, c] =
       max_{dy, dx} input[b,
                          strides[1] * y + rates[1] * dy,
                          strides[2] * x + rates[2] * dx,
                          c] +
                    filter[dy, dx, c]

Max-pooling is a special case when the filter has size equal to the pooling
kernel size and contains all zeros.

Note on duality: The dilation of `input` by the `filter` is equal to the
negation of the erosion of `-input` by the reflected `filter`. *)
val dilation2D
  :  ?name:string
  -> strides:int list
  -> rates:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes the gradient of morphological 2-D dilation with respect to the filter. *)
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

(* Computes the gradient of morphological 2-D dilation with respect to the input. *)
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

(* Returns x / y element-wise. *)
(* *NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val div
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Draw bounding boxes on a batch of images. *)
(* Outputs a copy of `images` but draws on top of the pixels zero or more bounding
boxes specified by the locations in `boxes`. The coordinates of the each
bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example, if an image is 100 x 200 pixels and the bounding box is
`[0.1, 0.2, 0.5, 0.9]`, the bottom-left and upper-right coordinates of the
bounding box will be `(10, 40)` to `(50, 180)`.

Parts of the bounding box may fall outside the image. *)
val drawBoundingBoxes
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> [ `float ] t
  -> ([< `float ] as 't) t

(* Partitions `data` into `num_partitions` tensors using indices from `partitions`. *)
(* For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
are placed in `outputs[i]` in lexicographic order of `js`, and the first
dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
In detail,

```python
    outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

    outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
```

`data.shape` must start with `partitions.shape`.

For example:

```python
    # Scalar partitions.
    partitions = 1
    num_partitions = 2
    data = [10, 20]
    outputs[0] = []  # Empty with shape [0, 2]
    outputs[1] = [[10, 20]]

    # Vector partitions.
    partitions = [0, 0, 1, 1, 0]
    num_partitions = 2
    data = [10, 20, 30, 40, 50]
    outputs[0] = [10, 20, 50]
    outputs[1] = [30, 40]
```

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/DynamicPartition.png' alt>
</div> *)
val dynamicPartition
  :  ?name:string
  -> num_partitions:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int32 ] t
  -> 't t list

(* Interleave the values from the `data` tensors into a single tensor. *)
(* Builds a merged tensor such that

```python
    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
```

For example, if each `indices[m]` is scalar or vector, we have

```python
    # Scalar indices:
    merged[indices[m], ...] = data[m][...]

    # Vector indices:
    merged[indices[m][i], ...] = data[m][i, ...]
```

Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is

    merged.shape = [max(indices)] + constant

Values are merged in order, so if an index appears in both `indices[m][i]` and
`indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
merged result.

For example:

```python
    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]
```

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/DynamicStitch.png' alt>
</div> *)
val dynamicStitch
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t list
  -> 't t list
  -> 't t

(* Computes the (possibly normalized) Levenshtein Edit Distance. *)
(* The inputs are variable-length sequences provided by SparseTensors
  (hypothesis_indices, hypothesis_values, hypothesis_shape)
and
  (truth_indices, truth_values, truth_shape).

The inputs are: *)
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

(* Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise. *)
(* See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289) *)
val elu
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes gradients for the exponential linear (Elu) operation. *)
val eluGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Encode strings into web-safe base64 format. *)
(* Refer to the following article for more information on base64 format:
en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
end so that the encoded has length multiple of 4. See Padding section of the
link above.

Web-safe means that the encoder uses - and _ instead of + and /. *)
val encodeBase64
  :  ?name:string
  -> ?pad:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

(* PNG-encode an image. *)
(* `image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
where `channels` is:

*   1: for grayscale.
*   2: for grayscale + alpha.
*   3: for RGB.
*   4: for RGBA.

The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
default or a value from 0 to 9.  9 is the highest compression level, generating
the smallest output, but is slower. *)
val encodePng
  :  ?name:string
  -> ?compression:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `string ] t

(* Creates or finds a child frame, and makes `data` available to the child frame. *)
(* This op is used together with `Exit` to create loops in the graph.
The unique `frame_name` is used by the `Executor` to identify frames. If
`is_constant` is true, `output` is a constant in the child frame; otherwise
it may be changed in the child frame. At most `parallel_iterations` iterations
are run in parallel in the child frame. *)
val enter
  :  ?name:string
  -> frame_name:string
  -> ?is_constant:bool
  -> ?parallel_iterations:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Returns the truth value of (x == y) element-wise. *)
(* *NOTE*: `Equal` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val equal
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> [ `bool ] t

(* Computes the Gauss error function of `x` element-wise. *)
val erf
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the complementary error function of `x` element-wise. *)
val erfc
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Exits the current frame to its parent frame. *)
(* Exit makes its input `data` available to the parent frame. *)
val exit
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Computes exponential of x element-wise.  \\(y = e^x\\). *)
val exp
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Inserts a dimension of 1 into a tensor's shape. *)
(* Given a tensor `input`, this operation inserts a dimension of 1 at the
dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
zero; if you specify a negative number for `dim` it is counted backward from
the end.

This operation is useful if you want to add a batch dimension to a single
element. For example, if you have a single image of shape `[height, width,
channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
which will make the shape `[1, height, width, channels]`.

Other examples:

```prettyprint
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```

This operation requires that:

`-1-input.dims() <= dim <= input.dims()`

This operation is related to `squeeze()`, which removes dimensions of
size 1. *)
val expandDims
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tdim) t
  -> 't t

(* Computes exponential of x - 1 element-wise. *)
(* I.e., \\(y = (\exp x) - 1\\). *)
val expm1
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Extracts a glimpse from the input tensor. *)
(* Returns a set of windows called glimpses extracted at location
`offsets` from the input tensor. If the windows only partially
overlaps the inputs, the non overlapping areas will be filled with
random noise.

The result is a 4-D tensor of shape `[batch_size, glimpse_height,
glimpse_width, channels]`. The channels and batch dimensions are the
same as that of the input tensor. The height and width of the output
windows are specified in the `size` parameter.

The argument `normalized` and `centered` controls how the windows are built:

* If the coordinates are normalized but not centered, 0.0 and 1.0
  correspond to the minimum and maximum of each height and width
  dimension.
* If the coordinates are both normalized and centered, they range from
  -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
  left corner, the lower right corner is located at (1.0, 1.0) and the
  center is at (0, 0).
* If the coordinates are not normalized they are interpreted as
  numbers of pixels. *)
val extractGlimpse
  :  ?name:string
  -> ?centered:bool
  -> ?normalized:bool
  -> ?uniform_noise:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t

(* Extract `patches` from `images` and put them in the 'depth' output dimension. *)
val extractImagePatches
  :  ?name:string
  -> ksizes:int list
  -> strides:int list
  -> rates:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Compute the 1-dimensional discrete Fourier Transform over the inner-most *)
(* dimension of `input`. *)
val fFT
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the 2-dimensional discrete Fourier Transform over the inner-most *)
(* 2 dimensions of `input`. *)
val fFT2D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the 3-dimensional discrete Fourier Transform over the inner-most 3 *)
(* dimensions of `input`. *)
val fFT3D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* A queue that produces elements in first-in first-out order. *)
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

(* Output a fact about factorials. *)
val fact
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type. *)
(* Attributes [min; max] define the clamping range for the 'inputs' data.  Op
divides this range into 255 steps (total of 256 values), then replaces each
'inputs' value with the closest of the quantized step values.

Quantization is called fake since the output is still in floating point. *)
val fakeQuantWithMinMaxArgs
  :  ?name:string
  -> ?min:float
  -> ?max:float
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t

(* Compute gradients for a FakeQuantWithMinMaxArgs operation. *)
val fakeQuantWithMinMaxArgsGradient
  :  ?name:string
  -> ?min:float
  -> ?max:float
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Fake-quantize the 'inputs' tensor of type float and shape `[b, h, w, d]` via *)
(* global float scalars `min` and `max` to 'outputs' tensor of same shape as
`inputs`.

[min; max] is the clamping range for the 'inputs' data.  Op divides this range
into 255 steps (total of 256 values), then replaces each 'inputs' value with the
closest of the quantized step values.

This operation has a gradient and thus allows for training `min` and `max` values. *)
val fakeQuantWithMinMaxVars
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Compute gradients for a FakeQuantWithMinMaxVars operation. *)
val fakeQuantWithMinMaxVarsGradient
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t * [ `float ] t * [ `float ] t

(* Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`, *)
(* `[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
to 'outputs' tensor of same shape as `inputs`.

[min; max] is the clamping range for the 'inputs' data in the corresponding
depth channel.  Op divides this range into 255 steps (total of 256 values), then
replaces each 'inputs' value with the closest of the quantized step values.

This operation has a gradient and thus allows for training `min` and `max` values. *)
val fakeQuantWithMinMaxVarsPerChannel
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation. *)
val fakeQuantWithMinMaxVarsPerChannelGradient
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t * [ `float ] t * [ `float ] t

(* Creates a tensor filled with a scalar value. *)
(* This operation creates a tensor of shape `dims` and fills it with `value`.

For example:

```prettyprint
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
``` *)
val fill
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 't t
  -> 't t

(* A Reader that outputs fixed-length records from a file. *)
val fixedLengthRecordReader
  :  ?name:string
  -> ?header_bytes:int
  -> record_bytes:int
  -> ?footer_bytes:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Generates labels for candidate sampling with a learned unigram distribution. *)
(* A unigram sampler could use a fixed unigram distribution read from a
file or passed in as an in-memory array instead of building up the distribution
from data on the fly. There is also an option to skew the distribution by
applying a distortion power to the weights.

The vocabulary file should be in CSV-like format, with the last field
being the weight associated with the word.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels. *)
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

(* Returns element-wise largest integer not greater than x. *)
val floor
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Returns x // y element-wise. *)
(* *NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val floorDiv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Returns element-wise remainder of division. When `x < 0` xor `y < 0` is *)
(* true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

*NOTE*: `FloorMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val floorMod
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

(* Performs fractional average pooling on the input. *)
(* Fractional average pooling is similar to Fractional max pooling in the pooling
region generation step. The only difference is that after pooling regions are
generated, a mean operation is performed instead of a max operation in each
pooling region. *)
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

(* Computes gradient of the FractionalAvgPool function. *)
(* Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
out_backprop to those indices that form the same pooling cell. Therefore, we
just need to know the shape of original input tensor, instead of the whole
tensor. *)
val fractionalAvgPoolGrad
  :  ?name:string
  -> ?overlapping:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Performs fractional max pooling on the input. *)
(* Fractional max pooling is slightly different than regular max pooling.  In
regular max pooling, you downsize an input set by taking the maximum value of
smaller N x N subsections of the set (often 2x2), and try to reduce the set by
a factor of N, where N is an integer.  Fractional max pooling, as you might
expect from the word 'fractional', means that the overall reduction ratio N
does not have to be an integer.

The sizes of the pooling regions are generated randomly but are fairly uniform.
For example, let's look at the height dimension, and the constraints on the
list of rows that will be pool boundaries.

First we define the following:

1.  input_row_length : the number of rows from the input set
2.  output_row_length : which will be smaller than the input
3.  alpha = input_row_length / output_row_length : our reduction ratio
4.  K = floor(alpha)
5.  row_pooling_sequence : this is the result list of pool boundary rows

Then, row_pooling_sequence should satisfy:

1.  a[0] = 0 : the first value of the sequence is 0
2.  a[end] = input_row_length : the last value of the sequence is the size
3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
4.  length(row_pooling_sequence) = output_row_length+1

For more details on fractional max pooling, see this paper:
[Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) *)
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

(* Computes gradient of the FractionalMaxPool function. *)
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

(* Batch normalization. *)
(* Note that the size of 4D Tensors are defined by either 'NHWC' or 'NCHW'.
The size of 1D Tensors matches the dimension C of the 4D Tensors. *)
val fusedBatchNorm
  :  ?name:string
  -> ?epsilon:float
  -> ?data_format:string
  -> ?is_training:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Gradient for batch normalization. *)
(* Note that the size of 4D Tensors are defined by either 'NHWC' or 'NCHW'.
The size of 1D Tensors matches the dimension C of the 4D Tensors. *)
val fusedBatchNormGrad
  :  ?name:string
  -> ?epsilon:float
  -> ?data_format:string
  -> ?is_training:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Performs a padding as a preprocess during a convolution. *)
(* Similar to FusedResizeAndPadConv2d, this op allows for an optimized
implementation where the spatial padding transformation stage is fused with the
im2col lookup, but in this case without the bilinear filtering required for
resizing. Fusing the padding prevents the need to write out the intermediate
results as whole tensors, reducing memory pressure, and we can get some latency
gains by merging the transformation calculations.
The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
order is used instead.
Internally this op uses a single per-graph scratch buffer, which means that it
will block if multiple versions are being run in parallel. This is because this
operator is primarily an optimization to minimize memory usage. *)
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

(* Performs a resize and padding as a preprocess during a convolution. *)
(* It's often possible to do spatial transformations more efficiently as part of
the packing stage of a convolution, so this op allows for an optimized
implementation where these stages are fused together. This prevents the need to
write out the intermediate results as whole tensors, reducing memory pressure,
and we can get some latency gains by merging the transformation calculations.
The data_format attribute for Conv2D isn't supported by this op, and defaults to
'NHWC' order.
Internally this op uses a single per-graph scratch buffer, which means that it
will block if multiple versions are being run in parallel. This is because this
operator is primarily an optimization to minimize memory usage. *)
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

(* Gather slices from `params` according to `indices`. *)
(* `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

```python
    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]

    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]

    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```

If `indices` is a permutation and `len(indices) == params.shape[0]` then
this operation will permute `params` accordingly.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/Gather.png' alt>
</div> *)
val gather
  :  ?name:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> 'tparams t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 'tparams t

(* Gather values or slices from `params` according to `indices`. *)
(* `params` is a Tensor of rank `P` and `indices` is a Tensor of rank `Q`.

`indices` must be integer tensor, containing indices into `params`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `params`.

Produces an output tensor with shape

```
[d_0, ..., d_{Q-2}, params.shape[K], ..., params.shape[P-1]].
```

Some examples below.

Simple indexing into a matrix:

```python
    indices = [[0, 0], [1, 1]]
    params = [['a', 'b'], ['c', 'd']]
    output = ['a', 'd']
```

Slice indexing into a matrix:

```python
    indices = [[1], [0]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['c', 'd'], ['a', 'b']]
```

Indexing into a 3-tensor:

```python
    indices = [[1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['a1', 'b1'], ['c1', 'd1']]]


    indices = [[0, 1], [1, 0]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['c0', 'd0'], ['a1', 'b1']]


    indices = [[0, 0, 1], [1, 0, 1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = ['b0', 'b1']
```

Batched indexing into a matrix:

```python
    indices = [[[0, 0]], [[0, 1]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['a'], ['b']]
```

Batched slice indexing into a matrix:

```python
    indices = [[[1]], [[0]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [[['c', 'd']], [['a', 'b']]]
```

Batched indexing into a 3-tensor:

```python
    indices = [[[1]], [[0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[[['a1', 'b1'], ['c1', 'd1']]],
              [[['a0', 'b0'], ['c0', 'd0']]]]

    indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['c0', 'd0'], ['a1', 'b1']],
              [['a0', 'b0'], ['c1', 'd1']]]


    indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['b0', 'b1'], ['d0', 'c1']]
``` *)
val gatherNd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'tparams t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 'tparams t

(* Store the input tensor in the state of the current session. *)
val getSessionHandle
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `string ] t

(* Get the value of the tensor specified by its handle. *)
val getSessionTensor
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'dtype t

(* Returns the truth value of (x > y) element-wise. *)
(* *NOTE*: `Greater` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val greater
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

(* Returns the truth value of (x >= y) element-wise. *)
(* *NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val greaterEqual
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

(* Convert one or more images from HSV to RGB. *)
(* Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

See `rgb_to_hsv` for a description of the HSV encoding. *)
val hSVToRGB
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Creates a non-initialized hash table. *)
(* This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable. *)
val hashTable
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?use_node_name_sharing:bool
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Outputs a `Summary` protocol buffer with a histogram. *)
(* The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite. *)
val histogramSummary
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `string ] t

(* Compute the inverse 1-dimensional discrete Fourier Transform over the inner-most *)
(* dimension of `input`. *)
val iFFT
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the inverse 2-dimensional discrete Fourier Transform over the inner-most *)
(* 2 dimensions of `input`. *)
val iFFT2D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the inverse 3-dimensional discrete Fourier Transform over the inner-most *)
(* 3 dimensions of `input`. *)
val iFFT3D
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Return a tensor with the same shape and contents as the input tensor or value. *)
val identity
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* A Reader that outputs the queued work as both the key and value. *)
(* To use, enqueue strings in a Queue.  ReaderRead will take the front
work string and output (work, work). *)
val identityReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Compute the lower regularized incomplete Gamma function `Q(a, x)`. *)
(* The lower regularized incomplete Gamma function is defined as:

```
P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)
```
where
```
gamma(a, x) = int_{0}^{x} t^{a-1} exp(-t) dt
```
is the lower incomplete Gamma function.

Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
Gamma function. *)
val igamma
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Compute the upper regularized incomplete Gamma function `Q(a, x)`. *)
(* The upper regularized incomplete Gamma function is defined as:

```
Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)
```
where
```
Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt
```
is the upper incomplete Gama function.

Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
Gamma function. *)
val igammac
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Returns the imaginary part of a complex number. *)
(* Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the imaginary part of each element in `input`. All
elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
is the real part and *b* is the imaginary part returned by this operation.

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(input) ==> [4.75, 5.75]
``` *)
val imag
  :  ?name:string
  -> type_:([< `float | `double ] as 'tout) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 't) t
  -> ([< `float | `double ] as 'tout) t

(* Outputs a `Summary` protocol buffer with images. *)
(* The summary has up to `max_images` summary values containing images. The
images are built from `tensor` which must be 4-D with shape `[batch_size,
height, width, channels]` and where `channels` can be:

*  1: `tensor` is interpreted as Grayscale.
*  3: `tensor` is interpreted as RGB.
*  4: `tensor` is interpreted as RGBA.

The images have the same number of channels as the input tensor. For float
input, the values are normalized one image at a time to fit in the range
`[0, 255]`.  `uint8` values are unchanged.  The op uses two different
normalization algorithms:

*  If the input values are all positive, they are rescaled so the largest one
   is 255.

*  If any input value is negative, the values are shifted so input value 0.0
   is at 127.  They are then rescaled so that either the smallest value is 0,
   or the largest one is 255.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_images` is 1, the summary value tag is '*tag*/image'.
*  If `max_images` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

The `bad_color` argument is the color to use in the generated images for
non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
Each element must be in the range `[0, 255]` (It represents the value of a
pixel in the output image).  Non-finite values in the input tensor are
replaced by this tensor in the output image.  The default value is the color
red. *)
val imageSummary
  :  ?name:string
  -> ?max_images:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float ] as 't) t
  -> [ `string ] t

(* Returns immutable tensor from memory region. *)
(* The current implementation memmaps the tensor from a file. *)
val immutableConst
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> memory_region_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

(* Says whether the targets are in the top `K` predictions. *)
(* This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
prediction for the target class is among the top `k` predictions among
all predictions for example `i`. Note that the behavior of `InTopK` differs
from the `TopK` op in its handling of ties; if multiple classes have the
same prediction value and straddle the top-`k` boundary, all of those
classes are considered to be in the top `k`.

More formally, let

  \\(predictions_i\\) be the predictions for all classes for example `i`,
  \\(targets_i\\) be the target class for example `i`,
  \\(out_i\\) be the output for example `i`,

$$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$ *)
val inTopK
  :  ?name:string
  -> k:int
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> ([< `int32 | `int64 ] as 't) t
  -> [ `bool ] t

(* Table initializer that takes two tensors for keys and values respectively. *)
val initializeTable
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tkey t
  -> 'tval t
  -> [ `unit ] t

(* Initializes a table from a text file. *)
(* It inserts one key-value pair into the table for each line of the file.
The key and value is extracted from the whole line content, elements from the
split line based on `delimiter` or the line number (starting from zero).
Where to extract the key and value from a line is specified by `key_index` and
`value_index`.

- A value of -1 means use the line number(starting from zero), expects `int64`.
- A value of -2 means use the whole line content, expects `string`.
- A value >= 0 means use the index (starting at zero) of the split line based
  on `delimiter`. *)
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

(* Computes the reciprocal of x element-wise. *)
(* I.e., \\(y = 1 / x\\). *)
val inv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes the gradient for the inverse of `x` wrt its input. *)
(* Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
is the corresponding input gradient. *)
val invGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes the inverse permutation of a tensor. *)
(* This operation computes the inverse of an index permutation. It takes a 1-D
integer tensor `x`, which represents the indices of a zero-based array, and
swaps each value with its index position. In other words, for an output tensor
`y` and an input tensor `x`, this operation computes the following:

`y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

The values must include 0. There can be no duplicate values or negative values.

For example:

```prettyprint
# tensor `x` is [3, 4, 0, 2, 1]
invert_permutation(x) ==> [2, 4, 3, 0, 1]
``` *)
val invertPermutation
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

(* Returns which elements of x are finite. *)
(* @compatibility(numpy)
Equivalent to np.isfinite
@end_compatibility *)
val isFinite
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `bool ] t

(* Returns which elements of x are Inf. *)
(* @compatibility(numpy)
Equivalent to np.isinf
@end_compatibility *)
val isInf
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `bool ] t

(* Returns which elements of x are NaN. *)
(* @compatibility(numpy)
Equivalent to np.isnan
@end_compatibility *)
val isNan
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> [ `bool ] t

(* Checks whether a tensor has been initialized. *)
(* Outputs boolean scalar indicating whether the tensor has been initialized. *)
val isVariableInitialized
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'dtype t
  -> [ `bool ] t

(* L2 Loss. *)
(* Computes half the L2 norm of a tensor without the `sqrt`:

    output = sum(t ** 2) / 2 *)
val l2Loss
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Local Response Normalization. *)
(* The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
dimension), and each vector is normalized independently.  Within a given vector,
each component is divided by the weighted, squared sum of inputs within
`depth_radius`.  In detail,

    sqr_sum[a, b, c, d] =
        sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    output = input / (bias + alpha * sqr_sum) ** beta

For details, see [Krizhevsky et al., ImageNet classification with deep
convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). *)
val lRN
  :  ?name:string
  -> ?depth_radius:int
  -> ?bias:float
  -> ?alpha:float
  -> ?beta:float
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t

(* Gradients for Local Response Normalization. *)
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

(* Generates labels for candidate sampling with a learned unigram distribution. *)
(* See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels. *)
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

(* Returns the truth value of (x < y) element-wise. *)
(* *NOTE*: `Less` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val less
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

(* Returns the truth value of (x <= y) element-wise. *)
(* *NOTE*: `LessEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val lessEqual
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

(* Computes the log of the absolute value of `Gamma(x)` element-wise. *)
val lgamma
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Generates values in an interval. *)
(* A sequence of `num` evenly-spaced values are generated beginning at `start`.
If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
so that the last one is exactly `stop`.

For example:

```
tf.linspace(10.0, 12.0, 3, name='linspace') => [ 10.0  11.0  12.0]
``` *)
val linSpace
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double ] as 't) t

(* Computes the difference between two lists of numbers or strings. *)
(* Given a list `x` and a list `y`, this operation returns a list `out` that
represents all values that are in `x` but not in `y`. The returned list `out`
is sorted in the same order that the numbers appear in `x` (duplicates are
preserved). This operation also returns a list `idx` that represents the
position of each `out` element in `x`. In other words:

`out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

For example, given this input:

```prettyprint
x = [1, 2, 3, 4, 5, 6]
y = [1, 3, 5]
```

This operation would return:

```prettyprint
out ==> [2, 4, 6]
idx ==> [1, 3, 5]
``` *)
val listDiff
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t
  -> 't t * ([< `int32 | `int64 ] as 'out_idx) t

(* Computes natural logarithm of x element-wise. *)
(* I.e., \\(y = \log_e x\\). *)
val log
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes natural logarithm of (1 + x) element-wise. *)
(* I.e., \\(y = \log_e (1 + x)\\). *)
val log1p
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes log softmax activations. *)
(* For each batch `i` and class `j` we have

    logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i]))) *)
val logSoftmax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Generates labels for candidate sampling with a log-uniform distribution. *)
(* See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels. *)
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

(* Returns the truth value of x AND y element-wise. *)
(* *NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val logicalAnd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> [ `bool ] t
  -> [ `bool ] t

(* Returns the truth value of NOT x element-wise. *)
val logicalNot
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> [ `bool ] t

(* Returns the truth value of x OR y element-wise. *)
(* *NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val logicalOr
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> [ `bool ] t
  -> [ `bool ] t

(* Outputs all keys and values in the table. *)
val lookupTableExport
  :  ?name:string
  -> type_:'tkeys Type.t
  -> type_1:'tvalues Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tkeys t * 'tvalues t

(* Looks up keys in a table, outputs the corresponding values. *)
(* The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values. *)
val lookupTableFind
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tin t
  -> 'tout t
  -> 'tout t

(* Replaces the contents of the table with the specified keys and values. *)
(* The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values. *)
val lookupTableImport
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tin t
  -> 'tout t
  -> [ `unit ] t

(* Updates the table to associates keys with values. *)
(* The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values. *)
val lookupTableInsert
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'tin t
  -> 'tout t
  -> [ `unit ] t

(* Computes the number of elements in the given table. *)
val lookupTableSize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

(* Forwards the input to the output. *)
(* This operator represents the loop termination condition used by the
'pivot' switches of a loop. *)
val loopCond
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> [ `bool ] t

(* Multiply the matrix 'a' by the matrix 'b'. *)
(* The inputs must be two-dimensional matrices and the inner dimension of
'a' (after being transposed if transpose_a is true) must match the
outer dimension of 'b' (after being transposed if transposed_b is
true).

*Note*: The default kernel implementation for MatMul on GPUs uses
cublas. *)
val matMul
  :  ?name:string
  -> ?transpose_a:bool
  -> ?transpose_b:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t

(* Returns the set of files matching a pattern. *)
(* Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion. *)
val matchingFiles
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

(* Copy a tensor setting everything outside a central band in each innermost matrix *)
(* to zero.

The `band` part is computed as follows:
Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor with the same shape where

`band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

The indicator function

`in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
                 (num_upper < 0 || (n-m) <= num_upper)`.

For example:

```prettyprint
# if 'input' is [[ 0,  1,  2, 3]
                 [-1,  0,  1, 2]
                 [-2, -1,  0, 1]
                 [-3, -2, -1, 0]],

tf.matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                       [-1,  0,  1, 2]
                                       [ 0, -1,  0, 1]
                                       [ 0,  0, -1, 0]],

tf.matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                      [-1,  0,  1, 0]
                                      [-2, -1,  0, 1]
                                      [ 0, -2, -1, 0]]
```

Useful special cases:

```prettyprint
 tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
 tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
 tf.matrix_band_part(input, 0, 0) ==> Diagonal.
``` *)
val matrixBandPart
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> 't t

(* Computes the determinant of one ore more square matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor containing the determinants
for all input submatrices `[..., :, :]`. *)
val matrixDeterminant
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Returns a batched diagonal tensor with a given batched diagonal values. *)
(* Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:

`output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.

For example:

```prettyprint
# 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]

and diagonal.shape = (2, 4)

tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                     [0, 2, 0, 0]
                                     [0, 0, 3, 0]
                                     [0, 0, 0, 4]],
                                    [[5, 0, 0, 0]
                                     [0, 6, 0, 0]
                                     [0, 0, 7, 0]
                                     [0, 0, 0, 8]]]

which has shape (2, 4, 4)
``` *)
val matrixDiag
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Returns the batched diagonal part of a batched tensor. *)
(* This operation returns a tensor with the `diagonal` part
of the batched `input`. The `diagonal` part is computed as follows:

Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:

`diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.

The input must be at least a matrix.

For example:

```prettyprint
# 'input' is [[[1, 0, 0, 0]
               [0, 2, 0, 0]
               [0, 0, 3, 0]
               [0, 0, 0, 4]],
              [[5, 0, 0, 0]
               [0, 6, 0, 0]
               [0, 0, 7, 0]
               [0, 0, 0, 8]]]

and input.shape = (2, 4, 4)

tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]

which has shape (2, 4)
``` *)
val matrixDiagPart
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Computes the inverse of one or more square invertible matrices or their *)
(* adjoints (conjugate transposes).

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.

The op uses LU decomposition with partial pivoting to compute the inverses.

If a matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result. *)
val matrixInverse
  :  ?name:string
  -> ?adjoint:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Returns a batched matrix tensor with new batched diagonal values. *)
(* Given `input` and `diagonal`, this operation returns a tensor with the
same shape and values as `input`, except for the main diagonal of the
innermost matrices.  These will be overwritten by the values in `diagonal`.

The output is computed as follows:

Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
`k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:

  * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
  * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`. *)
val matrixSetDiag
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t
  -> 't t

(* Solves systems of linear equations. *)
(* `Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `True` then each output matrix satisfies
`adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`. *)
val matrixSolve
  :  ?name:string
  -> ?adjoint:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t

(* Solves one or more linear least-squares problems. *)
(* `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
form matrices of size `[M, N]`. Rhs is a tensor of shape `[..., M, K]`.
The output is a tensor shape `[..., N, K]` where each output matrix solves
each of the equations matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]
in the least squares sense.

matrix and right-hand sides in the batch:

`matrix`=\\(A \in \Re^{m \times n}\\),
`rhs`=\\(B  \in \Re^{m \times k}\\),
`output`=\\(X  \in \Re^{n \times k}\\),
`l2_regularizer`=\\(\lambda\\).

If `fast` is `True`, then the solution is computed by solving the normal
equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
\\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
\lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
\\(X = A^T (A A^T + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is the
minimum-norm solution to the under-determined linear system, i.e.
\\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\), subject to
\\(A Z = B\\). Notice that the fast path is only numerically stable when
\\(A\\) is numerically full rank and has a condition number
\\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\) or\\(\lambda\\) is
sufficiently large.

If `fast` is `False` an algorithm based on the numerically robust complete
orthogonal decomposition is used. This computes the minimum-norm
least-squares solution, even when \\(A\\) is rank deficient. This path is
typically 6-7 times slower than the fast path. If `fast` is `False` then
`l2_regularizer` is ignored. *)
val matrixSolveLs
  :  ?name:string
  -> ?fast:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> [ `double ] t
  -> ([< `double | `float ] as 't) t

(* Solves systems of linear equations with upper or lower triangular matrices by *)
(* backsubstitution.

`matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
square matrices. If `lower` is `True` then the strictly upper triangular part
of each inner-most matrix is assumed to be zero and not accessed.
If `lower` is False then the strictly lower triangular part of each inner-most
matrix is assumed to be zero and not accessed.
`rhs` is a tensor of shape `[..., M, K]`.

The output is a tensor of shape `[..., M, K]`. If `adjoint` is
`True` then the innermost matrices in output` satisfy matrix equations
`matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `False` then the strictly then the  innermost matrices in
`output` satisfy matrix equations
`adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`. *)
val matrixTriangularSolve
  :  ?name:string
  -> ?lower:bool
  -> ?adjoint:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Computes the maximum of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val max
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Performs max pooling on the input. *)
val maxPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t

(* Performs 3D max pooling on the input. *)
val maxPool3D
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes gradients of max pooling function. *)
val maxPool3DGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes gradients of the maxpooling function. *)
val maxPoolGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t

(* Computes gradients of the maxpooling function. *)
val maxPoolGradWithArgmax
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t
  -> ([< `int32 | `int64 ] as 'targmax) t
  -> ([< `float ] as 't) t

(* Performs max pooling on the input and outputs both max values and indices. *)
(* The indices in `argmax` are flattened, so that a maximum value at position
`[b, y, x, c]` becomes flattened index
`((b * height + y) * width + x) * channels + c`. *)
val maxPoolWithArgmax
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'targmax) Type.t
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> ([< `float ] as 't) t
  -> ([< `float ] as 't) t * ([< `int32 | `int64 ] as 'targmax) t

(* Returns the max of x and y (i.e. x > y ? x : y) element-wise. *)
(* *NOTE*: `Maximum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val maximum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes the mean of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val mean
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Forwards the value of an available tensor from `inputs` to `output`. *)
(* `Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.

`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`. *)
val merge
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t list
  -> 't t * [ `int32 ] t

(* Merges summaries. *)
(* This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag. *)
val mergeSummary
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t list
  -> [ `string ] t

(* V2 format specific: merges the metadata files of sharded checkpoints.  The *)
(* result is one logical checkpoint, with one physical metadata file and renamed
data files.

Intended for 'grouping' multiple checkpoints in a sharded checkpoint setup.

If delete_old_dirs is true, attempts to delete recursively the dirname of each
path in the input checkpoint_prefixes.  This is useful when those paths are non
user-facing temporary locations. *)
val mergeV2Checkpoints
  :  ?name:string
  -> ?delete_old_dirs:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `unit ] t

(* Computes the minimum of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val min
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Returns the min of x and y (i.e. x < y ? x : y) element-wise. *)
(* *NOTE*: `Minimum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val minimum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Pads a tensor with mirrored values. *)
(* This operation pads a `input` with mirrored values according to the `paddings`
you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many values to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many values to add after the contents of `input`
in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
(if false, respectively).

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

```prettyprint
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1]], [2, 2]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
                      [2, 1, 1, 2, 3, 3, 2]
                      [5, 4, 4, 5, 6, 6, 5]
                      [5, 4, 4, 5, 6, 6, 5]]
``` *)
val mirrorPad
  :  ?name:string
  -> mode:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

(* Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor. *)
(* This operation folds the padded areas of `input` by `MirrorPad` according to the
`paddings` you specify. `paddings` must be the same as `paddings` argument
given to the corresponding `MirrorPad` op.

The folded size of each dimension D of the output is:

`input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`

For example:

```prettyprint
# 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
# 'paddings' is [[0, 1]], [0, 1]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[ 1,  5]
                      [11, 28]]
``` *)
val mirrorPadGrad
  :  ?name:string
  -> mode:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

(* Returns element-wise remainder of division. *)
(* *NOTE*: `Mod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val mod_
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

(* Returns x * y element-wise. *)
(* *NOTE*: `Mul` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val mul
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Draws samples from a multinomial distribution. *)
val multinomial
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> [ `int64 ] t

(* Creates an empty hash table that uses tensors as the backing store. It uses *)
(* 'open addressing' with quadratic reprobing to resolve collisions.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation. *)
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

(* Creates an empty hash table. *)
(* This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation. *)
val mutableHashTable
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?use_node_name_sharing:bool
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Creates an empty hash table. *)
(* This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation. *)
val mutableHashTableOfTensors
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?use_node_name_sharing:bool
  -> ?value_shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Computes numerical negative value element-wise. *)
(* I.e., \\(y = -x\\). *)
val neg
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Training via negative sampling. *)
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

(* Makes its input available to the next iteration. *)
val nextIteration
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Does nothing. Only useful as a placeholder for control edges. *)
val noOp
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `unit ] t

(* Greedily selects a subset of bounding boxes in descending order of score, *)
(* pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system.  Note that this
algorithm is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.

The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:

  selected_indices = tf.image.non_max_suppression(
      boxes, scores, max_output_size, iou_threshold)
  selected_boxes = tf.gather(boxes, selected_indices) *)
val nonMaxSuppression
  :  ?name:string
  -> ?iou_threshold:float
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `int32 ] t

(* Returns the truth value of (x != y) element-wise. *)
(* *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val notEqual
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> [ `bool ] t

(* Returns a one-hot tensor. *)
(* The locations represented by indices in `indices` take value `on_value`,
while all other locations take value `off_value`.

If the input `indices` is rank `N`, the output will have rank `N+1`,
The new axis is created at dimension `axis` (default: the new axis is
appended at the end).

If `indices` is a scalar the output shape will be a vector of length `depth`.

If `indices` is a vector of length `features`, the output shape will be:
```
  features x depth if axis == -1
  depth x features if axis == 0
```

If `indices` is a matrix (batch) with shape `[batch, features]`,
the output shape will be:
```
  batch x features x depth if axis == -1
  batch x depth x features if axis == 1
  depth x batch x features if axis == 0
```


Examples
=========

Suppose that

```
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 5.0
  off_value = 0.0
  axis = -1
```

Then output is `[4 x 3]`:

    ```output =
      [5.0 0.0 0.0]  // one_hot(0)
      [0.0 0.0 5.0]  // one_hot(2)
      [0.0 0.0 0.0]  // one_hot(-1)
      [0.0 5.0 0.0]  // one_hot(1)
    ```

Suppose that

```
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 0.0
  off_value = 3.0
  axis = 0
```

Then output is `[3 x 4]`:

    ```output =
      [0.0 3.0 3.0 3.0]
      [3.0 3.0 3.0 0.0]
      [3.0 3.0 3.0 3.0]
      [3.0 0.0 3.0 3.0]
    //  ^                one_hot(0)
    //      ^            one_hot(2)
    //          ^        one_hot(-1)
    //              ^    one_hot(1)
    ```
Suppose that

```
  indices = [[0, 2], [1, -1]]
  depth = 3
  on_value = 1.0
  off_value = 0.0
  axis = -1
```

Then output is `[2 x 2 x 3]`:

    ```output =
      [
        [1.0, 0.0, 0.0]  // one_hot(0)
        [0.0, 0.0, 1.0]  // one_hot(2)
      ][
        [0.0, 1.0, 0.0]  // one_hot(1)
        [0.0, 0.0, 0.0]  // one_hot(-1)
      ]``` *)
val oneHot
  :  ?name:string
  -> ?axis:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tI) t
  -> [ `int32 ] t
  -> 't t
  -> 't t
  -> 't t

(* Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor. *)
(* Packs the `N` tensors in `values` into a tensor with rank one higher than each
tensor in `values`, by packing them along the `axis` dimension.
Given a list of tensors of shape `(A, B, C)`;

if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
Etc.

For example:

```prettyprint
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```

This is the opposite of `unpack`. *)
val pack
  :  ?name:string
  -> ?axis:int
  -> ?control_inputs:Node.p list
  -> 't t list
  -> 't t

(* Pads a tensor with zeros. *)
(* This operation pads a `input` with zeros according to the `paddings` you
specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many zeros to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many zeros to add after the contents of `input`
in that dimension.

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

```prettyprint
# 't' is [[1, 1], [2, 2]]
# 'paddings' is [[1, 1], [2, 2]]
# rank of 't' is 2
pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                      [0, 0, 1, 1, 0, 0]
                      [0, 0, 2, 2, 0, 0]
                      [0, 0, 0, 0, 0, 0]]
``` *)
val pad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

(* A queue that produces elements in first-in first-out order. *)
(* Variable-size shapes are allowed by setting the corresponding shape dimensions
to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
size of any given element in the minibatch.  See below for details. *)
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

(* Concatenates a list of `N` tensors along the first dimension. *)
(* The input tensors are all required to have size 1 in the first dimension.

For example:

```prettyprint
# 'x' is [[1, 4]]
# 'y' is [[2, 5]]
# 'z' is [[3, 6]]
parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
```

The difference between concat and parallel_concat is that concat requires all
of the inputs be computed before the operation will begin but doesn't require
that the input shapes be known during graph construction.  Parallel concat
will copy pieces of the input into the output as they become available, in
some situations this can provide a performance benefit. *)
val parallelConcat
  :  ?name:string
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> 't t list
  -> 't t

(* Outputs random values from a normal distribution. The parameters may each be a *)
(* scalar which applies to the entire output, or a vector of length shape[0] which
stores the parameters for each batch. *)
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

(* Transforms a serialized tensorflow.TensorProto proto into a Tensor. *)
val parseTensor
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'out_type t

(* A placeholder op for a value that will be fed into the computation. *)
(* N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime. *)
val placeholder
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

(* A placeholder op for a value that will be fed into the computation. *)
(* N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime. *)
val placeholderV2
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

(* A placeholder op that passes through `input` when its output is not fed. *)
val placeholderWithDefault
  :  ?name:string
  -> shape:Dim.t list
  -> ?control_inputs:Node.p list
  -> 'dtype t
  -> 'dtype t

(* Compute the polygamma function \\(\psi^{(n)}(x)\\). *)
(* The polygamma function is defined as:

```
\psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)
```
where \\(\psi(x)\\) is the digamma function. *)
val polygamma
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the power of one value to another. *)
(* Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

```
# tensor 'x' is [[2, 2]], [3, 3]]
# tensor 'y' is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]
``` *)
val pow
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* An identity op that triggers an error if a gradient is requested. *)
(* When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, the TensorFlow gradient system
will return an error when trying to lookup the gradient of this op,
because no gradient must ever be registered for this function.  This
op exists to prevent subtle bugs from silently returning unimplemented
gradients in some corner cases. *)
val preventGradient
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* A queue that produces elements sorted by the first component value. *)
(* Note that the PriorityQueue requires the first component of any element
to be a scalar int64, in addition to the other elements declared by
component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
entry in their input (resp. output) lists. *)
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

(* Computes the product of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val prod
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the QR decompositions of one or more matrices. *)
(* Computes the QR decomposition of each inner matrix in `tensor` such that
`tensor[..., :, :] = q[..., :, :] * r[..., :,:])`

```prettyprint
# a is a tensor.
# q is a tensor of orthonormal matrices.
# r is a tensor of upper triangular matrices.
q, r = qr(a)
q_full, r_full = qr(a, full_matrices=True)
``` *)
val qr
  :  ?name:string
  -> ?full_matrices:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t * ([< `double | `float | `complex64 ] as 't) t

(* Quantizes then dequantizes a tensor. *)
(* This op simulates the precision loss from the quantized forward pass by:
1. Quantizing the tensor to fixed point numbers, which should match the target
   quantization method when it is used in inference.
2. Dequantizing it back to floating point numbers for the following ops, most
   likely matmul.

There are different ways to quantize. This version does not use the full range
of the output type, choosing to elide the lowest possible value for symmetry
(e.g., output range is -127 to 127, not -128 to 127 for signed 8 bit
quantization), so that 0.0 maps to 0.

To perform this op, we first find the range of values in our tensor. The range
we use is always centered on 0, so we find m such that

1. m = max(abs(input_min), abs(input_max)) if range_given is true,
2. m = max(max(abs(min_elem(input)), abs(max_elem(input))) otherwise.

Our input tensor range is then [-m, m].

Next, we choose our fixed-point quantization buckets, [min_fixed, max_fixed].
If signed_input is true, this is

  [min_fixed, max_fixed ] =
      [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1].

Otherwise, if signed_input is false, the fixed-point range is

  [min_fixed, max_fixed] = [0, (1 << num_bits) - 1].

From this we compute our scaling factor, s:

  s = (max_fixed - min_fixed) / (2 * m).

Now we can quantize and dequantize the elements of our tensor.  An element e
is transformed into e':

  e' = (e * s).round_to_nearest() / s.

Note that we have a different number of buckets in the signed vs. unsigned
cases.  For example, if num_bits == 8, we get 254 buckets in the signed case
vs. 255 in the unsigned case.

For example, suppose num_bits = 8 and m = 1.  Then

  [min_fixed, max_fixed] = [-127, 127], and
  s = (127 + 127) / 2 = 127.

Given the vector {-1, -0.5, 0, 0.3}, this is quantized to
{-127, -63, 0, 38}, and dequantized to {-1, -63.0/127, 0, 38.0/127}. *)
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

(* Convert the quantized 'input' tensor into a lower-precision 'output', using the *)
(* actual distribution of the values to maximize the usage of the lower bit depth
and adjusting the output min and max ranges accordingly.

[input_min, input_max] are scalar floats that specify the range for the float
interpretation of the 'input' data. For example, if input_min is -1.0f and
input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

This operator tries to squeeze as much precision as possible into an output with
a lower bit depth by calculating the actual min and max values found in the
data. For example, maybe that quint16 input has no values lower than 16,384 and
none higher than 49,152. That means only half the range is actually needed, all
the float interpretations are between -0.5f and 0.5f, so if we want to compress
the data into a quint8 output, we can use that range rather than the theoretical
-1.0f to 1.0f that is suggested by the input min and max.

In practice, this is most useful for taking output from operations like
QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
may have large potential output ranges, but in practice have a distribution of
input values that only uses a small fraction of the possible range. By feeding
that output into this operator, we can reduce it from 32 bits down to 8 with
minimal loss of accuracy. *)
val quantizeDownAndShrinkRange
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

(* Quantize the 'input' tensor of type float to 'output' tensor of type 'T'. *)
(* [min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
if T == qint8, out[i] -= (range(T) + 1) / 2.0
```
here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

*MIN_COMBINED Mode Example*

Assume the input is type float and has a possible range of [0.0, 6.0] and the
output type is quint8 ([0, 255]). The min_range and max_range values should be
specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
value of the input by 255/6 and cast to quint8.

If the output type was qint8 ([-128, 127]), the operation will additionally
subtract each value by 128 prior to casting, so that the range of values aligns
with the range of qint8.

If the mode is 'MIN_FIRST', then this approach is used:

```
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = number_of_steps / range
quantized = round(input * range_scale) - round(range_min * range_scale) +
  numeric_limits<T>::min()
quantized = max(quantized, numeric_limits<T>::min())
quantized = min(quantized, numeric_limits<T>::max())
```

The biggest difference between this and MIN_COMBINED is that the minimum range
is rounded first, before it's subtracted from the rounded value. With
MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
and dequantizing will introduce a larger and larger error.

One thing to watch out for is that the operator may choose to adjust the
requested minimum and maximum values slightly during the quantization process,
so you should always use the output ports as the range for further calculations.
For example, if the requested minimum and maximum values are close to equal,
they will be separated by a small epsilon value to prevent ill-formed quantized
buffers from being created. Otherwise, you can end up with buffers where all the
quantized values map to the same float value, which causes problems for
operations that have to perform further calculations on them. *)
val quantizeV2
  :  ?name:string
  -> type_:'t Type.t
  -> ?mode:string
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 't t * [ `float ] t * [ `float ] t

(* Produces the average pool of the input tensor for quantized types. *)
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

(* Quantized Batch normalization. *)
(* This op is deprecated and will be removed in the future. Prefer
`tf.nn.batch_normalization`. *)
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

(* Adds Tensor 'bias' to Tensor 'input' for Quantized types. *)
(* Broadcasts the values of bias on dimensions 0..N-2 of 'input'. *)
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

(* Concatenates quantized tensors along one dimension. *)
val quantizedConcat
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 't t list
  -> [ `float ] t list
  -> [ `float ] t list
  -> 't t * [ `float ] t * [ `float ] t

(* Computes a 2D convolution given quantized 4D input and filter tensors. *)
(* The inputs are quantized tensors where the lowest value represents the real
number of the associated minimum, and the highest represents the maximum.
This means that you can only interpret the quantized output in the same way, by
taking the returned minimum and maximum values into account. *)
val quantizedConv2D
  :  ?name:string
  -> type_:'out_type Type.t
  -> strides:int list
  -> padding:string
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> 'tfilter t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

(* Quantized Instance normalization. *)
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

(* Perform a quantized matrix multiplication of  `a` by the matrix `b`. *)
(* The inputs must be two-dimensional matrices and the inner dimension of
`a` (after being transposed if `transpose_a` is non-zero) must match the
outer dimension of `b` (after being transposed if `transposed_b` is
non-zero). *)
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

(* Produces the max pool of the input tensor for quantized types. *)
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

(* Computes Quantized Rectified Linear: `max(features, 0)` *)
val quantizedRelu
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

(* Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)` *)
val quantizedRelu6
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

(* Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)` *)
val quantizedReluX
  :  ?name:string
  -> type_:'out_type Type.t
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> 'out_type t * [ `float ] t * [ `float ] t

(* Reshapes a quantized tensor as per the Reshape op. *)
(* ``` *)
val quantizedReshape
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tshape) t
  -> [ `float ] t
  -> [ `float ] t
  -> 't t * [ `float ] t * [ `float ] t

(* Closes the given queue. *)
(* This operation signals that no more elements will be enqueued in the
given queue. Subsequent Enqueue(Many) operations will fail.
Subsequent Dequeue(Many) operations will continue to succeed if
sufficient elements remain in the queue. Subsequent Dequeue(Many)
operations that would block will fail immediately. *)
val queueClose
  :  ?name:string
  -> ?cancel_pending_enqueues:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

(* Computes the number of elements in the given queue. *)
val queueSize
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t

(* Converts one or more images from RGB to HSV. *)
(* Outputs a tensor of the same shape as the `images` tensor, containing the HSV
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
`output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue. *)
val rGBToHSV
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Randomly crop `image`. *)
(* `size` is a 1-D int64 tensor with 2 elements representing the crop height and
width.  The values must be non negative.

This Op picks a random location in `image` and crops a `height` by `width`
rectangle from that location.  The random location is picked so the cropped
area will fit inside the original image. *)
val randomCrop
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int64 ] t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

(* Outputs random values from the Gamma distribution(s) described by alpha. *)
(* This op uses the algorithm by Marsaglia et al. to acquire samples via
transformation-rejection from pairs of uniform and normal random variables.
See http://dl.acm.org/citation.cfm?id=358414 *)
val randomGamma
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 's) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Randomly shuffles a tensor along its first dimension. *)
(*   The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:

```prettyprint
[[1, 2],       [[5, 6],
 [3, 4],  ==>   [1, 2],
 [5, 6]]        [3, 4]]
``` *)
val randomShuffle
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* A queue that randomizes the order of elements. *)
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

(* Outputs random values from a normal distribution. *)
(* The generated values will have mean 0 and standard deviation 1. *)
val randomStandardNormal
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `float | `double ] as 'dtype) t

(* Outputs random values from a uniform distribution. *)
(* The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded. *)
val randomUniform
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `float | `double ] as 'dtype) t

(* Outputs random integers from a uniform distribution. *)
(* The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.

The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`). *)
val randomUniformInt
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tout) t
  -> ([< `int32 | `int64 ] as 'tout) t
  -> ([< `int32 | `int64 ] as 'tout) t

(* Creates a sequence of numbers. *)
(* This operation creates a sequence of numbers that begins at `start` and
extends by increments of `delta` up to but not including `limit`.

For example:

```
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
``` *)
val range
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int32 | `int64 ] as 'tidx) t

(* Returns the rank of a tensor. *)
(* This operation returns an integer representing the rank of `input`.

For example:

```prettyprint
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
```

**Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
of a tensor is the number of indices required to uniquely select each element
of the tensor. Rank is also known as 'order', 'degree', or 'ndims.' *)
val rank
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int32 ] t

(* Reads and outputs the entire contents of the input filename. *)
val readFile
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

(* Returns the number of records this Reader has produced. *)
(* This is the same as the number of ReaderRead executions that have
succeeded. *)
val readerNumRecordsProduced
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

(* Returns the number of work units this Reader has finished processing. *)
val readerNumWorkUnitsCompleted
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

(* Returns the next record (key, value pair) produced by a Reader. *)
(* Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file). *)
val readerRead
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t * [ `string ] t

(* Returns up to `num_records` (key, value) pairs produced by a Reader. *)
(* Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).
It may return less than `num_records` even before the last batch. *)
val readerReadUpTo
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `string ] t * [ `string ] t

(* Restore a Reader to its initial clean state. *)
val readerReset
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

(* Restore a reader to a previously saved state. *)
(* Not all Readers support being restored, so this can produce an
Unimplemented error. *)
val readerRestoreState
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `unit ] t

(* Produce a string tensor that encodes the state of a Reader. *)
(* Not all Readers support being serialized, so this can produce an
Unimplemented error. *)
val readerSerializeState
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t

(* Returns the real part of a complex number. *)
(* Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the real part of each element in `input`. All elements in
`input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
 part returned by this operation and *b* is the imaginary part.

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.real(input) ==> [-2.25, 3.25]
``` *)
val real
  :  ?name:string
  -> type_:([< `float | `double ] as 'tout) Type.t
  -> ?control_inputs:Node.p list
  -> ([< `complex64 ] as 't) t
  -> ([< `float | `double ] as 'tout) t

(* Returns x / y element-wise for real types. *)
(* If `x` and `y` are reals, this will return the floating-point division.

*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val realDiv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes the reciprocal of x element-wise. *)
(* I.e., \\(y = 1 / x\\). *)
val reciprocal
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes the gradient for the inverse of `x` wrt its input. *)
(* Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
is the corresponding input gradient. *)
val reciprocalGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Joins a string Tensor across the given dimensions. *)
(* Computes the string join across dimensions in the given string Tensor of shape
`[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
strings with the given separator (default: empty string).  Negative indices are
counted backwards from the end, with `-1` being equivalent to `n - 1`.

For example:

```
# tensor `a` is [['a', 'b'], ['c', 'd']]
tf.reduce_join(a, 0) ==> ['ac', 'bd']
tf.reduce_join(a, 1) ==> ['ab', 'cd']
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ['ac', 'bd']
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ['ab', 'cd']
tf.reduce_join(a, 0, keep_dims=True) ==> [['ac', 'bd']]
tf.reduce_join(a, 1, keep_dims=True) ==> [['ab'], ['cd']]
tf.reduce_join(a, 0, separator='.') ==> ['a.c', 'b.d']
tf.reduce_join(a, [0, 1]) ==> ['acbd']
tf.reduce_join(a, [1, 0]) ==> ['abcd']
tf.reduce_join(a, []) ==> ['abcd']
``` *)
val reduceJoin
  :  ?name:string
  -> ?keep_dims:bool
  -> ?separator:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `string ] t

(* Creates or finds a child frame, and makes `data` available to the child frame. *)
(* The unique `frame_name` is used by the `Executor` to identify frames. If
`is_constant` is true, `output` is a constant in the child frame; otherwise
it may be changed in the child frame. At most `parallel_iterations` iterations
are run in parallel in the child frame. *)
val refEnter
  :  ?name:string
  -> frame_name:string
  -> ?is_constant:bool
  -> ?parallel_iterations:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Exits the current frame to its parent frame. *)
(* Exit makes its input `data` available to the parent frame. *)
val refExit
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Return the same ref tensor as the input ref tensor. *)
val refIdentity
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Forwards the value of an available tensor from `inputs` to `output`. *)
(* `Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.

`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`. *)
val refMerge
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t list
  -> 't t * [ `int32 ] t

(* Makes its input available to the next iteration. *)
val refNextIteration
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Forwards the `index`th element of `inputs` to `output`. *)
val refSelect
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 't t list
  -> 't t

(* Forwards the ref tensor `data` to the output port determined by `pred`. *)
(* If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.

See also `Switch` and `Merge`. *)
val refSwitch
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `bool ] t
  -> 't t * 't t

(* Computes rectified linear: `max(features, 0)`. *)
val relu
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes rectified linear 6: `min(max(features, 0), 6)`. *)
val relu6
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes rectified linear 6 gradients for a Relu6 operation. *)
val relu6Grad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes rectified linear gradients for a Relu operation. *)
val reluGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Given a quantized tensor described by (input, input_min, input_max), outputs a *)
(* range that covers the actual values present in that tensor.  This op is
typically used to produce the requested_output_min and requested_output_max for
Requantize. *)
val requantizationRange
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 'tinput t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t * [ `float ] t

(* Convert the quantized 'input' tensor into a lower-precision 'output', using the *)
(* output range specified with 'requested_output_min' and 'requested_output_max'.

[input_min, input_max] are scalar floats that specify the range for the float
interpretation of the 'input' data. For example, if input_min is -1.0f and
input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f. *)
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

(* Reshapes a tensor. *)
(* Given `tensor`, this operation returns a tensor that has the same values
as `tensor` with shape `shape`.

If one component of `shape` is the special value -1, the size of that dimension
is computed so that the total size remains constant.  In particular, a `shape`
of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.

If `shape` is 1-D or higher, then the operation returns a tensor with shape
`shape` filled with the values of `tensor`. In this case, the number of elements
implied by `shape` must be the same as the number of elements in `tensor`.

For example:

```prettyprint
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]

# tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                        [3, 3, 4, 4]]

# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

# -1 can also be used to infer the shape

# -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 2:
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 3:
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]],
                             [[4, 4, 4],
                              [5, 5, 5],
                              [6, 6, 6]]]

# tensor 't' is [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
``` *)
val reshape
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tshape) t
  -> 't t

(* Resize `images` to `size` using area interpolation. *)
(* Input images can be of different types but output images are always float. *)
val resizeArea
  :  ?name:string
  -> ?align_corners:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t

(* Resize `images` to `size` using bicubic interpolation. *)
(* Input images can be of different types but output images are always float. *)
val resizeBicubic
  :  ?name:string
  -> ?align_corners:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t

(* Resize `images` to `size` using bilinear interpolation. *)
(* Input images can be of different types but output images are always float. *)
val resizeBilinear
  :  ?name:string
  -> ?align_corners:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t

(* Computes the gradient of bilinear interpolation. *)
val resizeBilinearGrad
  :  ?name:string
  -> ?align_corners:bool
  -> ?control_inputs:Node.p list
  -> [ `float ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Resize `images` to `size` using nearest neighbor interpolation. *)
val resizeNearestNeighbor
  :  ?name:string
  -> ?align_corners:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

(* Computes the gradient of nearest neighbor interpolation. *)
val resizeNearestNeighborGrad
  :  ?name:string
  -> ?align_corners:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `int32 | `float | `double ] as 't) t

(* Restores a tensor from checkpoint files. *)
(* Reads a tensor stored in one or several files. If there are several files (for
instance because a tensor was saved as slices), `file_pattern` may contain
wildcard symbols (`*` and `?`) in the filename portion only, not in the
directory portion.

If a `file_pattern` matches several files, `preferred_shard` can be used to hint
in which file the requested tensor is likely to be found. This op will first
open the file at index `preferred_shard` in the list of matching files and try
to restore tensors from that file.  Only if some tensors or tensor slices are
not found in that first file, then the Op opens all the files. Setting
`preferred_shard` to match the value passed as the `shard` input
of a matching `Save` Op may speed up Restore.  This attribute only affects
performance, not correctness.  The default value -1 means files are processed in
order.

See also `RestoreSlice`. *)
val restore
  :  ?name:string
  -> type_:'dt Type.t
  -> ?preferred_shard:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> 'dt t

(* Restores a tensor from checkpoint files. *)
(* This is like `Restore` except that restored tensor can be listed as filling
only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
larger tensor and the slice that the restored tensor covers.

The `shape_and_slice` input has the same format as the
elements of the `shapes_and_slices` input of the `SaveSlices` op. *)
val restoreSlice
  :  ?name:string
  -> type_:'dt Type.t
  -> ?preferred_shard:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t
  -> 'dt t

(* Reverses specific dimensions of a tensor. *)
(* Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
of `tensor`, this operation reverses each dimension i of `tensor` where
`dims[i]` is `True`.

`tensor` can have up to 8 dimensions. The number of dimensions
of `tensor` must equal the number of elements in `dims`. In other words:

`rank(tensor) = size(dims)`

For example:

```prettyprint
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]

# 'dims' is [False, False, False, True]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]

# 'dims' is [False, True, False, False]
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]

# 'dims' is [False, False, True, False]
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
``` *)
val reverse
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `bool | `float | `double | `complex64 ] as 't) t
  -> [ `bool ] t
  -> ([< `int32 | `int64 | `bool | `float | `double | `complex64 ] as 't) t

(* Reverses variable length slices. *)
(* This op first slices `input` along the dimension `batch_dim`, and for each
slice `i`, reverses the first `seq_lengths[i]` elements along
the dimension `seq_dim`.

The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

The output slice `i` along dimension `batch_dim` is then given by input
slice `i`, with the first `seq_lengths[i]` slices along dimension
`seq_dim` reversed.

For example:

```prettyprint
# Given this:
batch_dim = 0
seq_dim = 1
input.dims = (4, 8, ...)
seq_lengths = [7, 2, 3, 5]

# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

# while entries past seq_lens are copied through:
output[0, 7:, :, ...] = input[0, 7:, :, ...]
output[1, 2:, :, ...] = input[1, 2:, :, ...]
output[2, 3:, :, ...] = input[2, 3:, :, ...]
output[3, 2:, :, ...] = input[3, 2:, :, ...]
```

In contrast, if:

```prettyprint
# Given this:
batch_dim = 2
seq_dim = 0
input.dims = (8, ?, 4, ...)
seq_lengths = [7, 2, 3, 5]

# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]

# while entries past seq_lens are copied through:
output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
``` *)
val reverseSequence
  :  ?name:string
  -> seq_dim:int
  -> ?batch_dim:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tlen) t
  -> 't t

(* Reverses specific dimensions of a tensor. *)
(* NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
`tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.

Given a `tensor`, and a `int32` tensor `axis` representing the set of
dimensions of `tensor` to reverse. This operation reverses each dimension
`i` for which there exists `j` s.t. `axis[j] == i`.

`tensor` can have up to 8 dimensions. The number of dimensions specified
in `axis` may be 0 or more entries. If an index is specified more than
once, a InvalidArgument error is raised.

For example:

```prettyprint
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]

# 'dims' is [3] or 'dims' is -1
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]

# 'dims' is '[1]' (or 'dims' is '[-3]')
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]

# 'dims' is '[2]' (or 'dims' is '[-2]')
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
``` *)
val reverseV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `bool | `float | `double | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `int32 | `int64 | `bool | `float | `double | `complex64 ] as 't) t

(* Returns element-wise integer closest to x. *)
(* If the result is midway between two representable values,
the even representable is chosen.
For example:

```
rint(-1.5) ==> -2.0
rint(0.5000001) ==> 1.0
rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
``` *)
val rint
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Rounds the values of a tensor to the nearest integer, element-wise. *)
(* Rounds half to even.  Also known as bankers rounding. If you want to round
according to the current system rounding mode use std::cint. *)
val round
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes reciprocal of square root of x element-wise. *)
(* I.e., \\(y = 1 / \sqrt{x}\\). *)
val rsqrt
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes the gradient for the rsqrt of `x` wrt its input. *)
(* Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
is the corresponding input gradient. *)
val rsqrtGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Generate a single randomly distorted bounding box for an image. *)
(* Bounding box annotations are often supplied in addition to ground-truth labels
in image recognition or object localization tasks. A common technique for
training such a system is to randomly distort an image while preserving
its content, i.e. *data augmentation*. This Op outputs a randomly distorted
localization of an object, i.e. bounding box, given an `image_size`,
`bounding_boxes` and a series of constraints.

The output of this Op is a single bounding box that may be used to crop the
original image. The output is returned as 3 tensors: `begin`, `size` and
`bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
what the bounding box looks like.

Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example,

```python
    # Generate a single distorted bounding box.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bounding_boxes)

    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox_for_draw)
    tf.image_summary('images_with_box', image_with_box)

    # Employ the bounding box to distort the image.
    distorted_image = tf.slice(image, begin, size)
```

Note that if no bounding box information is available, setting
`use_image_if_no_bounding_boxes = true` will assume there is a single implicit
bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
false and no bounding boxes are supplied, an error is raised. *)
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

(* Outputs a `Summary` protocol buffer with scalar values. *)
(* The input `tags` and `values` must have the same shape.  The generated summary
has a summary value for each tag-value pair in `tags` and `values`. *)
val scalarSummary
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `string ] t

(* Adds sparse updates to a variable reference. *)
(* This operation computes

    # Scalar indices
    ref[indices, ...] += updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] += updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/ScatterAdd.png' alt>
</div> *)
val scatterAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Divides a variable reference by sparse updates. *)
(* This operation computes

    # Scalar indices
    ref[indices, ...] /= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] /= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions divide.

Requires `updates.shape = indices.shape + ref.shape[1:]`. *)
val scatterDiv
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Multiplies sparse updates into a variable reference. *)
(* This operation computes

    # Scalar indices
    ref[indices, ...] *= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] *= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] *= updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions multiply.

Requires `updates.shape = indices.shape + ref.shape[1:]`. *)
val scatterMul
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Creates a new tensor by applying sparse `updates` to individual *)
(* values or slices within a zero tensor of the given `shape` tensor according to
indices.  This operator is the inverse of the [tf.gather_nd](#gather_nd)
operator which extracts values or slices from a given tensor.

TODO(simister): Add a link to Variable.__getitem__ documentation on slice
syntax.

`shape` is a `TensorShape` with rank `P` and `indices` is a `Tensor` of rank
`Q`.

`indices` must be integer tensor, containing indices into `shape`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `shape`.

`updates` is Tensor of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, shape[K], ..., shape[P-1]].
```

The simplest form of scatter is to insert individual elements in a tensor by
index. For example, say we want to insert 4 scattered elements in a rank-1
tensor with 8 elements.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/ScatterNd1.png' alt>
</div>

In Python, this scatter operation would look like this:

    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
      print sess.run(scatter)

The resulting tensor would look like this:

    [0, 11, 0, 10, 9, 0, 0, 12]

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/ScatterNd2.png' alt>
</div>

In Python, this scatter operation would look like this:

    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
      print sess.run(scatter)

The resulting tensor would look like this:

    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]] *)
val scatterNd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t

(* Applies sparse addition between `updates` and individual values or slices *)
(* within a given variable according to `indices`.

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```

For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
elements. In Python, that addition would look like this:

    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    add = tf.scatter_nd_add(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(add)

The resulting update to ref would look like this:

    [1, 13, 3, 14, 14, 6, 7, 20]

See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
slices. *)
val scatterNdAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Applies sparse subtraction between `updates` and individual values or slices *)
(* within a given variable according to `indices`.

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```

For example, say we want to subtract 4 scattered elements from a rank-1 tensor
with 8 elements. In Python, that subtraction would look like this:

    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    sub = tf.scatter_nd_sub(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(sub)

The resulting update to ref would look like this:

    [1, -9, 3, -6, -4, 6, 7, -4]

See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
slices. *)
val scatterNdSub
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Applies sparse `updates` to individual values or slices within a given *)
(* variable according to `indices`.

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```

For example, say we want to update 4 scattered elements to a rank-1 tensor to
8 elements. In Python, that update would look like this:

    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1] ,[7]])
    updates = tf.constant([9, 10, 11, 12])
    update = tf.scatter_nd_update(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(update)

The resulting update to ref would look like this:

    [1, 11, 3, 10, 9, 6, 7, 12]

See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
slices. *)
val scatterNdUpdate
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t

(* Subtracts sparse updates to a variable reference. *)
(*     # Scalar indices
    ref[indices, ...] -= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] -= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their (negated) contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/ScatterSub.png' alt>
</div> *)
val scatterSub
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Applies sparse updates to a variable reference. *)
(* This operation computes

    # Scalar indices
    ref[indices, ...] = updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] = updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

If values in `ref` is to be updated more than once, because there are
duplicate entries in `indices`, the order at which the updates happen
for each value is undefined.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/ScatterUpdate.png' alt>
</div> *)
val scatterUpdate
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t

(* Computes fingerprints of the input strings. *)
val sdcaFprint
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

(* Applies L1 regularization shrink step on the parameters. *)
val sdcaShrinkL1
  :  ?name:string
  -> l1:float
  -> l2:float
  -> ?control_inputs:Node.p list
  -> [ `float ] t list
  -> [ `unit ] t

(* Computes the maximum along segments of a tensor. *)
(* Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \max_j(data_j)\\) where `max` is over `j` such
that `segment_ids[j] == i`.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/SegmentMax.png' alt>
</div> *)
val segmentMax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes the mean along segments of a tensor. *)
(* Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
over `j` such that `segment_ids[j] == i` and `N` is the total number of
values summed.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/SegmentMean.png' alt>
</div> *)
val segmentMean
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes the minimum along segments of a tensor. *)
(* Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \min_j(data_j)\\) where `min` is over `j` such
that `segment_ids[j] == i`.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/SegmentMin.png' alt>
</div> *)
val segmentMin
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes the product along segments of a tensor. *)
(* Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \prod_j data_j\\) where the product is over `j` such
that `segment_ids[j] == i`.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/SegmentProd.png' alt>
</div> *)
val segmentProd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the sum along segments of a tensor. *)
(* Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/SegmentSum.png' alt>
</div> *)
val segmentSum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Selects elements from `t` or `e`, depending on `condition`. *)
(* The `t`, and `e` tensors must all have the same shape, and the
output will also have that shape.

The `condition` tensor must be a scalar if `t` and `e` are scalars.
If `t` and `e` are vectors or higher rank, then `condition` must be either a
scalar, a vector with size matching the first dimension of `t`, or must have
the same shape as `t`.

The `condition` tensor acts as a mask that chooses, based on the value at each
element, whether the corresponding element / row in the output should be
taken from `t` (if true) or `e` (if false).

If `condition` is a vector and `t` and `e` are higher rank matrices, then
it chooses which row (outer dimension) to copy from `t` and `e`.
If `condition` has the same shape as `t` and `e`, then it chooses which
element to copy from `t` and `e`.

For example:

```prettyprint
# 'condition' tensor is [[True,  False]
#                        [False, True]]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 6],
                             [7, 4]]


# 'condition' tensor is [True, False]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 2],
                             [7, 8]]

``` *)
val select
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> 't t
  -> 't t
  -> 't t

(* Computes the Eigen Decomposition of a batch of square self-adjoint matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix
SelfAdjointEig.

The result is a [..., M+1, M] matrix with [..., 0,:] containing the
eigenvalues, and subsequent [...,1:, :] containing the eigenvectors. *)
val selfAdjointEig
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Computes the eigen decomposition of one or more square self-adjoint matrices. *)
(* Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
`input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`.

```prettyprint
# a is a tensor.
# e is a tensor of eigenvalues.
# v is a tensor of eigenvectors.
e, v = self_adjoint_eig(a)
e = self_adjoint_eig(a, compute_v=False)
``` *)
val selfAdjointEigV2
  :  ?name:string
  -> ?compute_v:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t * ([< `double | `float ] as 't) t

(* Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`. *)
(* The `SparseTensor` must have rank `R` greater than 1, and the first dimension
is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The serialized
`SparseTensor` objects going into each row of `serialized_sparse` will have
rank `R-1`.

The minibatch size `N` is extracted from `sparse_shape[0]`. *)
val serializeManySparse
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `string ] t

(* Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object. *)
val serializeSparse
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `string ] t

(* Number of unique elements along last dimension of input `set`. *)
(* Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
and `set_shape`. The last dimension contains values in a set, duplicates are
allowed but ignored.

If `validate_indices` is `True`, this op validates the order and range of `set`
indices. *)
val setSize
  :  ?name:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `int32 | `int64 | `string ] as 't) t
  -> [ `int64 ] t
  -> [ `int32 ] t

(* Returns the shape of a tensor. *)
(* This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```prettyprint
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
``` *)
val shape
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'out_type) t

(* Returns shape of tensors. *)
(* This operation returns N 1-D integer tensors representing shape of `input[i]s`. *)
val shapeN
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> 't t list
  -> ([< `int32 | `int64 ] as 'out_type) t list

(* Generate a sharded filename. The filename is printf formatted as *)
(*    %s-%05d-of-%05d, basename, shard, num_shards. *)
val shardedFilename
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `string ] t

(* Generate a glob pattern matching all sharded file names. *)
val shardedFilespec
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `string ] t

(* Computes sigmoid of `x` element-wise. *)
(* Specifically, `y = 1 / (1 + exp(-x))`. *)
val sigmoid
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes the gradient of the sigmoid of `x` wrt its input. *)
(* Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
`dy` is the corresponding input gradient. *)
val sigmoidGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Returns an element-wise indication of the sign of a number. *)
(* `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`. *)
val sign
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes sin of x element-wise. *)
val sin
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Returns the size of a tensor. *)
(* This operation returns an integer representing the number of elements in
`input`.

For example:

```prettyprint
# 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
size(t) ==> 12
``` *)
val size
  :  ?name:string
  -> type_:([< `int32 | `int64 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'out_type) t

(* Parses a text file and creates a batch of examples. *)
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

(* Return a slice from 'input'. *)
(* The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.

*Requirements*:
  0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n) *)
val slice
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> 't t

(* Computes softmax activations. *)
(* For each batch `i` and class `j` we have

    softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j])) *)
val softmax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes softmax cross entropy cost and gradients to backpropagate. *)
(* Inputs are the logits, not probabilities. *)
val softmaxCrossEntropyWithLogits
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t

(* Computes softplus: `log(exp(features) + 1)`. *)
val softplus
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes softplus gradients for a softplus operation. *)
val softplusGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes softsign: `features / (abs(features) + 1)`. *)
val softsign
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes softsign gradients for a softsign operation. *)
val softsignGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* SpaceToBatch for 4-D tensors of type T. *)
(* This is a legacy version of the more general SpaceToBatchND.

Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size. *)
val spaceToBatch
  :  ?name:string
  -> block_size:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

(* SpaceToBatch for N-D tensors of type T. *)
(* This operation divides 'spatial' dimensions `[1, ..., M]` of the input into a
grid of blocks of shape `block_shape`, and interleaves these blocks with the
'batch' dimension (0) such that in the output, the spatial dimensions
`[1, ..., M]` correspond to the position within the grid, and the batch
dimension combines both the position within a spatial block and the original
batch position.  Prior to division into blocks, the spatial dimensions of the
input are optionally zero padded according to `paddings`.  See below for a
precise description. *)
val spaceToBatchND
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tblock_shape) t
  -> ([< `int32 | `int64 ] as 'tpaddings) t
  -> 't t

(* SpaceToDepth for tensors of type T. *)
(* Rearranges blocks of spatial data, into depth. More specifically,
this op outputs a copy of the input tensor where values from the `height`
and `width` dimensions are moved to the `depth` dimension.
The attr `block_size` indicates the input block size and how the data is moved.

  * Non-overlapping blocks of size `block_size x block size` are rearranged
    into depth at each location.
  * The depth of the output tensor is `input_depth * block_size * block_size`.
  * The input tensor's height and width must be divisible by block_size.

That is, assuming the input is in the shape:
`[batch, height, width, depth]`,
the shape of the output will be:
`[batch, height/block_size, width/block_size, depth*block_size*block_size]`

This operation requires that the input tensor be of rank 4, and that
`block_size` be >=1 and a divisor of both the input `height` and `width`.

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given this input of shape `[1, 2, 2, 1]`, and block_size of 2:

```prettyprint
x = [[[[1], [2]],
      [[3], [4]]]]
```

This operation will output a tensor of shape `[1, 1, 1, 4]`:

```prettyprint
[[[[1, 2, 3, 4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
the corresponding output will have a single element (i.e. width and height are
both 1) and will have a depth of 4 channels (1 * block_size * block_size).
The output element shape is `[1, 1, 4]`.

For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.

```prettyprint
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

This operation, for block_size of 2, will return the following tensor of shape
`[1, 1, 1, 12]`

```prettyprint
[[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:

```prettyprint
x = [[[[1],   [2],  [5],  [6]],
      [[3],   [4],  [7],  [8]],
      [[9],  [10], [13],  [14]],
      [[11], [12], [15],  [16]]]]
```

the operator will return the following tensor of shape `[1 2 2 4]`:

```prettyprint
x = [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
``` *)
val spaceToDepth
  :  ?name:string
  -> block_size:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Applies a sparse gradient to a given accumulator. Does not add if local_step is *)
(* lesser than the accumulator's global_step. *)
val sparseAccumulatorApplyGradient
  :  ?name:string
  -> has_known_shape:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 'dtype) t
  -> [ `int64 ] t
  -> [ `unit ] t

(* Extracts the average sparse gradient in the given SparseConditionalAccumulator, *)
(* provided that sufficient (i.e., more than num_required) gradients have been
accumulated. The op will blocks until sufficient gradients have been
accumulated. If the accumulator has already aggregated more than num_required
gradients, it will return its average of the accumulated gradients.
Also automatically increments the recorded global_step in the accumulator by 1,
and resets the aggregate to 0. *)
val sparseAccumulatorTakeGradient
  :  ?name:string
  -> type_1:([< `float | `double | `int64 | `int32 | `complex64 ] as 'dtype) Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `int64 ] t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 'dtype) t * [ `int64 ] t

(* Adds two `SparseTensor` objects to produce another `SparseTensor`. *)
(* The input `SparseTensor` objects' indices are assumed ordered in standard
lexicographic order.  If this is not the case, before this step run
`SparseReorder` to restore index ordering.

By default, if two values sum to zero at some index, the output `SparseTensor`
would still include that particular location in its index, storing a zero in the
corresponding value slot.  To override this, callers can specify `thresh`,
indicating that if the sum has a magnitude strictly smaller than `thresh`, its
corresponding value and index would then not be included.  In particular,
`thresh == 0` (default) means everything is kept and actual thresholding happens
only for a positive value.

In the following shapes, `nnz` is the count after taking `thresh` into account. *)
val sparseAdd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 'treal) t
  -> [ `int64 ] t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * [ `int64 ] t

(* The gradient operator for the SparseAdd op. *)
(* The SparseAdd op calculates A + B, where A, B, and the sum are all represented
as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
values of A and B. *)
val sparseAddGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* var: Should be from a Variable(). *)
val sparseApplyAdadelta
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update relevant entries in '*var' and '*accum' according to the adagrad scheme. *)
(* That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum)) *)
val sparseApplyAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update entries in '*var' and '*accum' according to the proximal adagrad scheme. *)
val sparseApplyAdagradDA
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the centered RMSProp algorithm. *)
(* The centered RMSProp algorithm uses an estimate of the centered second moment
(i.e., the variance) for normalization, as opposed to regular RMSProp, which
uses the (uncentered) second moment. This often helps with training, but is
slightly more expensive in terms of computation and memory.

Note that in dense implementation of this algorithm, mg, ms, and mom will
update even if the grad is zero, but in this sparse implementation, mg, ms,
and mom will not update in iterations during which the grad is zero.

mean_square = decay * mean_square + (1-decay) * gradient ** 2
mean_grad = decay * mean_grad + (1-decay) * gradient
Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom *)
val sparseApplyCenteredRMSProp
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update relevant entries in '*var' according to the Ftrl-proximal scheme. *)
(* That is for rows we have grad for, we update var, accum and linear as follows:
accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new *)
val sparseApplyFtrl
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update relevant entries in '*var' and '*accum' according to the momentum scheme. *)
(* Set use_nesterov = True if you want to use Nesterov momentum.

That is for rows we have grad for, we update var and accum as follows:

accum = accum * momentum + grad
var -= lr * accum *)
val sparseApplyMomentum
  :  ?name:string
  -> ?use_locking:bool
  -> ?use_nesterov:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Sparse update entries in '*var' and '*accum' according to FOBOS algorithm. *)
(* That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
prox_v = var
prox_v -= lr * grad * (1 / sqrt(accum))
var = sign(prox_v)/(1+lr*l2) * max{ |prox_v|-lr*l1,0} *)
val sparseApplyProximalAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Sparse update '*var' as FOBOS algorithm with fixed learning rate. *)
(* That is for rows we have grad for, we update var as follows:
prox_v = var - alpha * grad
var = sign(prox_v)/(1+alpha*l2) * max{ |prox_v|-alpha*l1,0} *)
val sparseApplyProximalGradientDescent
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the RMSProp algorithm. *)
(* Note that in dense implementation of this algorithm, ms and mom will
update even if the grad is zero, but in this sparse implementation, ms
and mom will not update in iterations during which the grad is zero.

mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom *)
val sparseApplyRMSProp
  :  ?name:string
  -> ?use_locking:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Concatenates a list of `SparseTensor` along the specified dimension. *)
(* Concatenation is with respect to the dense versions of these sparse tensors.
It is assumed that each input is a `SparseTensor` whose elements are ordered
along increasing dimension number.

All inputs' shapes must match, except for the concat dimension.  The
`indices`, `values`, and `shapes` lists must have the same length.

The output shape is identical to the inputs', except along the concat
dimension, where it is the sum of the inputs' sizes along that dimension.

The output elements will be resorted to preserve the sort order along
increasing dimension number.

This op runs in `O(M log M)` time, where `M` is the total number of non-empty
values across all inputs. This is due to the need for an internal sort in
order to concatenate efficiently across an arbitrary dimension.

For example, if `concat_dim = 1` and the inputs are

    sp_inputs[0]: shape = [2, 3]
    [0, 2]: 'a'
    [1, 0]: 'b'
    [1, 1]: 'c'

    sp_inputs[1]: shape = [2, 4]
    [0, 1]: 'd'
    [0, 2]: 'e'

then the output will be

    shape = [2, 7]
    [0, 2]: 'a'
    [0, 4]: 'd'
    [0, 5]: 'e'
    [1, 0]: 'b'
    [1, 1]: 'c'

Graphically this is equivalent to doing

    [    a] concat [  d e  ] = [    a   d e  ]
    [b c  ]        [       ]   [b c          ] *)
val sparseConcat
  :  ?name:string
  -> concat_dim:int
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t list
  -> 't t list
  -> [ `int64 ] t list
  -> [ `int64 ] t * 't t * [ `int64 ] t

(* A conditional accumulator for aggregating sparse gradients. The accumulator *)
(* accepts gradients marked with local_step greater or equal to the most recent
global_step known to the accumulator. The average can be extracted from the
accumulator, provided sufficient gradients have been accumulated. Extracting the
average automatically resets the aggregate to 0, and increments the global_step
recorded by the accumulator. *)
val sparseConditionalAccumulator
  :  ?name:string
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Adds up a SparseTensor and a dense Tensor, using these special rules: *)
(* (1) Broadcasts the dense side to have the same shape as the sparse side, if
    eligible;
(2) Then, only the dense values pointed to by the indices of the SparseTensor
    participate in the cwise addition.

By these rules, the result is a logical SparseTensor with exactly the same
indices and shape, but possibly with different non-zero values.  The output of
this Op is the resultant non-zero values. *)
val sparseDenseCwiseAdd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Component-wise divides a SparseTensor by a dense Tensor. *)
(* *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction. *)
val sparseDenseCwiseDiv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Component-wise multiplies a SparseTensor by a dense Tensor. *)
(* The output locations corresponding to the implicitly zero elements in the sparse
tensor will be zero (i.e., will not take up storage space), regardless of the
contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).

*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction. *)
val sparseDenseCwiseMul
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Multiply matrix 'a' by matrix 'b'. *)
(* The inputs must be two-dimensional matrices and the inner dimension of 'a' must
match the outer dimension of 'b'. This op is optimized for the case where at
least one of 'a' or 'b' is sparse. The breakeven for using this versus a dense
matrix multiply on one platform was 30% zero values in the sparse matrix. *)
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

(* Computes the sum of elements across dimensions of a SparseTensor. *)
(* This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
instead of a sparse one.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python. *)
val sparseReduceSum
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the sum of elements across dimensions of a SparseTensor. *)
(* This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
SparseTensor.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python. *)
val sparseReduceSumSparse
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> [ `int64 ] t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * [ `int64 ] t

(* Reorders a SparseTensor into the canonical, row-major ordering. *)
(* Note that by convention, all sparse ops preserve the canonical ordering along
increasing dimension number. The only time ordering can be violated is during
manual manipulation of the indices and values vectors to add entries.

Reordering does not affect the shape of the SparseTensor.

If the tensor has rank `R` and `N` non-empty values, `input_indices` has
shape `[N, R]`, input_values has length `N`, and input_shape has length `R`. *)
val sparseReorder
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t * 't t

(* Reshapes a SparseTensor to represent values in a new dense shape. *)
(* This operation has the same semantics as reshape on the represented dense
tensor.  The `input_indices` are recomputed based on the requested `new_shape`.

If one component of `new_shape` is the special value -1, the size of that
dimension is computed so that the total dense size remains constant.  At
most one component of `new_shape` can be -1.  The number of dense elements
implied by `new_shape` must be the same as the number of dense elements
originally implied by `input_shape`.

Reshaping does not affect the order of values in the SparseTensor.

If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
has length `R_out`, then `input_indices` has shape `[N, R_in]`,
`input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
`output_shape` has length `R_out`. *)
val sparseReshape
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `int64 ] t

(* Computes the mean along sparse segments of a tensor. *)
(* Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`. *)
val sparseSegmentMean
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

(* Computes gradients for SparseSegmentMean. *)
(* Returns tensor 'output' with same shape as grad, except for dimension 0 whose
value is output_dim0. *)
val sparseSegmentMeanGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

(* Computes the sum along sparse segments of a tensor divided by the sqrt of N. *)
(* N is the size of the segment being reduced.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments. *)
val sparseSegmentSqrtN
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

(* Computes gradients for SparseSegmentSqrtN. *)
(* Returns tensor 'output' with same shape as grad, except for dimension 0 whose
value is output_dim0. *)
val sparseSegmentSqrtNGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

(* Computes the sum along sparse segments of a tensor. *)
(* Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`.

For example:

```prettyprint
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

# Select two rows, one segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
  ==> [[0 0 0 0]]

# Select two rows, two segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
  ==> [[ 1  2  3  4]
       [-1 -2 -3 -4]]

# Select all rows, two segments.
tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
  ==> [[0 0 0 0]
       [5 6 7 8]]

# Which is equivalent to:
tf.segment_sum(c, tf.constant([0, 0, 1]))
``` *)
val sparseSegmentSum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Applies softmax to a batched N-D `SparseTensor`. *)
(* The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
(where `N >= 2`), and with indices sorted in the canonical lexicographic order.

This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
zero elements do not participate*.  Specifically, the algorithm is equivalent
to the following:

  (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
      with shape `[B, C]`, along the size-C dimension;
  (2) Masks out the original implicitly-zero locations;
  (3) Renormalizes the remaining elements.

Hence, the `SparseTensor` result has exactly the same non-zero indices and
shape. *)
val sparseSoftmax
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double ] as 't) t

(* Computes softmax cross entropy cost and gradients to backpropagate. *)
(* Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
a matrix of label probabilities, but rather a single label per row
of features.  This label is considered to have probability 1.0 for the
given row.

Inputs are the logits, not probabilities. *)
val sparseSoftmaxCrossEntropyWithLogits
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tlabels) t
  -> ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t

(* Returns the element-wise max of two SparseTensors. *)
(* Assumes the two SparseTensors have the same shape, i.e., no broadcasting. *)
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

(* Returns the element-wise min of two SparseTensors. *)
(* Assumes the two SparseTensors have the same shape, i.e., no broadcasting. *)
val sparseSparseMinimum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`. *)
(* This Op does not require `a_indices` be sorted in standard lexicographic order. *)
val sparseTensorDenseAdd
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Multiply SparseTensor (of rank 2) 'A' by dense matrix 'B'. *)
(* No validity checking is performed on the indices of A.  However, the following
input format is recommended for optimal behavior:

if adjoint_a == false:
  A should be sorted in lexicographically increasing order.  Use SparseReorder
  if you're not sure.
if adjoint_a == true:
  A should be sorted in order of increasing dimension 1 (i.e., 'column major'
  order instead of 'row major' order). *)
val sparseTensorDenseMatMul
  :  ?name:string
  -> ?adjoint_a:bool
  -> ?adjoint_b:bool
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> 't t
  -> 't t

(* Converts a sparse representation into a dense tensor. *)
(* Builds an array `dense` with shape `output_shape` such that

```prettyprint
# If sparse_indices is scalar
dense[i] = (i == sparse_indices ? sparse_values : default_value)

# If sparse_indices is a vector, then for each i
dense[sparse_indices[i]] = sparse_values[i]

# If sparse_indices is an n by d matrix, then for each i in [0, n)
dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
```

All other values in `dense` are set to `default_value`.  If `sparse_values` is a
scalar, all sparse indices are set to this single value.

Indices should be sorted in lexicographic order, and indices must not
contain any repeats. If `validate_indices` is true, these properties
are checked during execution. *)
val sparseToDense
  :  ?name:string
  -> ?validate_indices:bool
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t
  -> 't t

(* Applies set operation along last dimension of 2 `SparseTensor` inputs. *)
(* See SetOperationOp::SetOperationFromContext for values of `set_operation`.

If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
order and range of `set1` and `set2` indices.

Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

If `validate_indices` is `True`, this op validates the order and range of `set1`
and `set2` indices.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`. *)
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

(* Splits a tensor into `num_split` tensors along one dimension. *)
val split
  :  ?name:string
  -> num_split:int
  -> ?control_inputs:Node.p list
  -> [ `int32 ] t
  -> 't t
  -> 't t list

(* Splits a tensor into `num_split` tensors along one dimension. *)
val splitV
  :  ?name:string
  -> num_split:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tlen) t
  -> [ `int32 ] t
  -> 't t list

(* Computes square root of x element-wise. *)
(* I.e., \\(y = \sqrt{x} = x^{1/2}\\). *)
val sqrt
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes the gradient for the sqrt of `x` wrt its input. *)
(* Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
is the corresponding input gradient. *)
val sqrtGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes square of x element-wise. *)
(* I.e., \\(y = x * x = x^2\\). *)
val square
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Returns (x - y)(x - y) element-wise. *)
(* *NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val squaredDifference
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Removes dimensions of size 1 from the shape of a tensor. *)
(* Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`squeeze_dims`.

For example:

```prettyprint
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
```

Or, to remove specific size 1 dimensions:

```prettyprint
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
``` *)
val squeeze
  :  ?name:string
  -> ?squeeze_dims:int list
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* A stack that produces elements in first-in last-out order. *)
val stack
  :  ?name:string
  -> ?stack_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Delete the stack from its resource container. *)
val stackClose
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `unit ] t

(* Pop the element at the top of the stack. *)
val stackPop
  :  ?name:string
  -> type_:'elem_type Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 'elem_type t

(* Push an element onto the stack. *)
val stackPush
  :  ?name:string
  -> ?swap_memory:bool
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> 't t
  -> 't t

(* Stops gradient computation. *)
(* When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, this op prevents the contribution of
its inputs to be taken into account.  Normally, the gradient generator adds ops
to a graph to compute the derivatives of a specified 'loss' by recursively
finding out inputs that contributed to its computation.  If you insert this op
in the graph it inputs are masked from the gradient generator.  They are not
taken into account for computing gradients.

This is useful any time you want to compute a value with TensorFlow but need
to pretend that the value was a constant. Some examples include:

*  The *EM* algorithm where the *M-step* should not involve backpropagation
   through the output of the *E-step*.
*  Contrastive divergence training of Boltzmann machines where, when
   differentiating the energy function, the training must not backpropagate
   through the graph that generated the samples from the model.
*  Adversarial training, where no backprop should happen through the adversarial
   example generation process. *)
val stopGradient
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Return a strided slice from `input`. *)
(* Note, most python users will want to use the Python `Tensor.__getitem__`
or `Variable.__getitem__` rather than this op directly.

The goal of this op is to produce a new tensor with a subset of
the elements from the `n` dimensional `input` tensor. The subset is chosen using
a sequence of `m` sparse range specifications encoded into the arguments
of this function. Note, in some cases
`m` could be equal to `n`, but this need not be the case. Each
range specification entry can be one of the following:

- An ellipsis (...). Ellipses are used to imply zero or more
  dimensions of full-dimension selection and are produced using
  `ellipsis_mask`. For example, `foo[...]` is the identity slice.

- A new axis. This is used to insert a new shape=1 dimension and is
  produced using `new_axis_mask`. For example, `foo[:, ...]` where
  `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.


- A range `begin:end:stride`. This is used to specify how much to choose from
  a given dimension. `stride` can be any integer but 0.  `begin` is an integer
  which represents the index of the first value to select while `end` represents
  the index of the last value to select. The number of values selected in each
  dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
  `begin` and `end` can be negative where `-1` is the last element, `-2` is
  the second to last. `begin_mask` controls whether to replace the explicitly
  given `begin` with an implicit effective value of `0` if `stride > 0` and
  `-1` if `stride < 0`. `end_mask` is analogous but produces the number
  required to create the largest open interval. For example, given a shape
  `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
  not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
  and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
  first dimension of a tensor while dropping the last two (in the original
  order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.

- A single index. This is used to keep only elements that have a given
  index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
  shape `(6,)` tensor. This is encoded in `begin` and `end` and
  `shrink_axis_mask`.

Each conceptual range specification is encoded in the op's argument. This
encoding is best understand by considering a non-trivial example. In
particular,
`foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as

```prettyprint
begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
end = [2, 4, x, x, -3, x]
strides = [1, 1, x, x, -1, 1]
begin_mask = 1<<4 | 1 << 5 = 48
end_mask = 1<<5 = 32
ellipsis_mask = 1<<3 = 8
new_axis_mask = 1<<2 4
shrink_axis_mask = 1<<0
```

In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
the slice becomes (2, 1, 5, 5, 2, 5).
Let us walk step by step through each argument specification.

1.  The first argument in the example slice is turned into `begin = 1` and
`end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
also set the appropriate bit in `shrink_axis_mask`.

2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
zero bits contributed.

3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
dimension in the final shape. Dummy values are contributed to begin,
end and stride, while the new_axis_mask bit is set.

4. `...` grab the full ranges from as many dimensions as needed to
fully specify a slice for every dimension of the input shape.

5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
with a dimension that has shape `s` is converted to a positive index
`s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
is done internally so begin, end and strides receive x, -3, and -1.
The appropriate begin_mask bit is set to indicate the start range is the
full range (ignoring the x).

6. `:` indicates that the entire contents of the corresponding dimension
is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
`end_mask` are also set.

*Requirements*:
  `0 != strides[i] for i in [0, m)`
  `ellipsis_mask must be a power of two (only one ellipsis)` *)
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

(* Assign `value` to the sliced l-value reference of `ref`. *)
(* The values of `value` are assigned to the positions in the variable
`ref` that are selected by the slice parameters. The slice parameters
`begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.

NOTE this op currently does not support broadcasting and so `value`'s
shape must be exactly the shape produced by the slice of `ref`. *)
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

(* Returns the gradient of `StridedSlice`. *)
(* Since `StridedSlice` cuts out pieces of its `input` which is size
`shape`, its gradient will have the same shape (which is passed here
as `shape`). The gradient will be zero in any element that the slice
does not select.

Arguments are the same as StridedSliceGrad with the exception that
`dy` is the input gradient to be propagated and `shape` is the
shape of `StridedSlice`'s `input`. *)
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

(* Joins the strings in the given list of string tensors into one tensor; *)
(* with the given separator (default is an empty separator). *)
val stringJoin
  :  ?name:string
  -> ?separator:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t list
  -> [ `string ] t

(* Split elements of `input` based on `delimiter` into a `SparseTensor`. *)
(* Let N be the size of source (typically N will be the batch size). Split each
element of `input` based on `delimiter` and return a `SparseTensor`
containing the splitted tokens. Empty tokens are ignored.

`delimiter` can be empty, or a string of split characters. If `delimiter` is an
 empty string, each element of `input` is split into individual single-byte
 character strings, including splitting of UTF-8 multibyte sequences. Otherwise
 every character of `delimiter` is a potential split point.

For example:
  N = 2, input[0] is 'hello world' and input[1] is 'a b c', then the output
  will be

  indices = [0, 0;
             0, 1;
             1, 0;
             1, 1;
             1, 2]
  shape = [2, 3]
  values = ['hello', 'world', 'a', 'b', 'c'] *)
val stringSplit
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `int64 ] t * [ `string ] t * [ `int64 ] t

(* Converts each string in the input Tensor to its hash mod by a number of buckets. *)
(* The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time.
This functionality will be deprecated and it's recommended to use
`tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`. *)
val stringToHashBucket
  :  ?name:string
  -> num_buckets:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

(* Converts each string in the input Tensor to its hash mod by a number of buckets. *)
(* The hash function is deterministic on the content of the string within the
process and will never change. However, it is not suitable for cryptography.
This function may be used when CPU time is scarce and inputs are trusted or
unimportant. There is a risk of adversaries constructing inputs that all hash
to the same bucket. To prevent this problem, use a strong hash function with
`tf.string_to_hash_bucket_strong`. *)
val stringToHashBucketFast
  :  ?name:string
  -> num_buckets:int
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

(* Converts each string in the input Tensor to its hash mod by a number of buckets. *)
(* The hash function is deterministic on the content of the string within the
process. The hash function is a keyed hash function, where attribute `key`
defines the key of the hash function. `key` is an array of 2 elements.

A strong hash is important when inputs may be malicious, e.g. URLs with
additional components. Adversaries could try to make their inputs hash to the
same bucket for a denial-of-service attack or to skew the results. A strong
hash prevents this by making it dificult, if not infeasible, to compute inputs
that hash to the same bucket. This comes at a cost of roughly 4x higher compute
time than `tf.string_to_hash_bucket_fast`. *)
val stringToHashBucketStrong
  :  ?name:string
  -> num_buckets:int
  -> key:int list
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int64 ] t

(* Converts each string in the input Tensor to the specified numeric type. *)
(* (Note that int32 overflow results in an error while float overflow
results in a rounded value.) *)
val stringToNumber
  :  ?name:string
  -> type_:([< `float | `int32 ] as 'out_type) Type.t
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `float | `int32 ] as 'out_type) t

(* Returns x - y element-wise. *)
(* *NOTE*: `Sub` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val sub
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Return substrings from `Tensor` of strings. *)
(* For each string in the input `Tensor`, creates a substring starting at index
`pos` with a total length of `len`.

If `len` defines a substring that would extend beyond the length of the input
string, then as many characters as possible are used.

If `pos` is negative or specifies a character index larger than any of the input
strings, then an `InvalidArgumentError` is thrown.

`pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
Op creation.

*NOTE*: `Substr` supports broadcasting up to two dimensions. More about
broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

---

Examples

Using scalar `pos` and `len`:

```
input = [b'Hello', b'World']
position = 1
length = 3

output = [b'ell', b'orl']
```

Using `pos` and `len` with same shape as `input`:

```
input = [[b'ten', b'eleven', b'twelve'],
         [b'thirteen', b'fourteen', b'fifteen'],
         [b'sixteen', b'seventeen', b'eighteen']]
position = [[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]]
length =   [[2, 3, 4],
            [4, 3, 2],
            [5, 5, 5]]

output = [[b'en', b'eve', b'lve'],
          [b'hirt', b'urt', b'te'],
          [b'ixtee', b'vente', b'hteen']]
```

Broadcasting `pos` and `len` onto `input`:

```
input = [[b'ten', b'eleven', b'twelve'],
         [b'thirteen', b'fourteen', b'fifteen'],
         [b'sixteen', b'seventeen', b'eighteen'],
         [b'nineteen', b'twenty', b'twentyone']]
position = [1, 2, 3]
length =   [1, 2, 3]

output = [[b'e', b'ev', b'lve'],
          [b'h', b'ur', b'tee'],
          [b'i', b've', b'hte'],
          [b'i', b'en', b'nty']]
```

Broadcasting `input` onto `pos` and `len`:

```
input = b'thirteen'
position = [1, 5, 7]
length =   [3, 2, 1]

output = [b'hir', b'ee', b'n']
``` *)
val substr
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t
  -> [ `string ] t

(* Computes the sum of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val sum
  :  ?name:string
  -> ?keep_dims:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tidx) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the singular value decompositions of one or more matrices. *)
(* Computes the SVD of each inner matrix in `input` such that
`input[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :, :])`

```prettyprint
# a is a tensor containing a batch of matrices.
# s is a tensor of singular values for each matrix.
# u is the tensor containing of left singular vectors for each matrix.
# v is the tensor containing of right singular vectors for each matrix.
s, u, v = svd(a)
s, _, _ = svd(a, compute_uv=False)
``` *)
val svd
  :  ?name:string
  -> ?compute_uv:bool
  -> ?full_matrices:bool
  -> ?control_inputs:Node.p list
  -> ([< `double | `float | `complex64 ] as 't) t
  -> ([< `double | `float | `complex64 ] as 't) t * ([< `double | `float | `complex64 ] as 't) t * ([< `double | `float | `complex64 ] as 't) t

(* Forwards `data` to the output port determined by `pred`. *)
(* If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.

See also `RefSwitch` and `Merge`. *)
val switch
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `bool ] t
  -> 't t * 't t

(* A Reader that outputs the records from a TensorFlow Records file. *)
val tFRecordReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?compression_type:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Read `SparseTensors` from a `SparseTensorsMap` and concatenate them. *)
(* The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
`N` is the minibatch size and the rows correspond to the output handles of
`AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
original `SparseTensor` objects that went into the given input ops must all
match.  When the final `SparseTensor` is created, it has rank one
higher than the ranks of the incoming `SparseTensor` objects
(they have been concatenated along a new row dimension on the left).

The output `SparseTensor` object's shape values for all dimensions but the
first are the max across the input `SparseTensor` objects' shape values
for the corresponding dimensions.  Its first shape value is `N`, the minibatch
size.

The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.

For example, if the handles represent an input, which is a `[2, 3]` matrix
representing two original `SparseTensor` objects:

```
    index = [ 0]
            [10]
            [20]
    values = [1, 2, 3]
    shape = [50]
```

and

```
    index = [ 2]
            [10]
    values = [4, 5]
    shape = [30]
```

then the final `SparseTensor` will be:

```
    index = [0  0]
            [0 10]
            [0 20]
            [1  2]
            [1 10]
    values = [1, 2, 3, 4, 5]
    shape = [2 50]
``` *)
val takeManySparseFromTensorsMap
  :  ?name:string
  -> type_1:'dtype Type.t
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> [ `int64 ] t
  -> [ `int64 ] t * 'dtype t * [ `int64 ] t

(* Computes tan of x element-wise. *)
val tan
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes hyperbolic tangent of `x` element-wise. *)
val tanh
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes the gradient for the tanh of `x` wrt its input. *)
(* Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
is the corresponding input gradient. *)
val tanhGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Returns a tensor that may be mutated, but only persists within a single step. *)
(* This is an experimental op for internal use only and it is possible to use this
op in unsafe ways.  DO NOT USE unless you fully understand the risks.

It is the caller's responsibility to ensure that 'ref' is eventually passed to a
matching 'DestroyTemporaryVariable' op after all other uses have completed.

Outputs a ref to the tensor state so it may be read or modified.

  E.g.
      var = state_ops._temporary_variable([1, 2], types.float_)
      var_name = var.op.name
      var = state_ops.assign(var, [[4.0, 5.0]])
      var = state_ops.assign_add(var, [[6.0, 7.0]])
      final = state_ops._destroy_temporary_variable(var, var_name=var_name) *)
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

(* Deprecated. Use TensorArrayCloseV3 *)
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

(* Deprecated. Use TensorArrayConcatV3 *)
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

(* Deprecated. Use TensorArrayGatherV3 *)
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

(* Deprecated. Use TensorArrayGradV3 *)
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

(* Deprecated. Use TensorArrayReadV3 *)
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

(* Deprecated. Use TensorArrayScatterV3 *)
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

(* Deprecated. Use TensorArraySizeV3 *)
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

(* Deprecated. Use TensorArraySplitV3 *)
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

(* Deprecated. Use TensorArrayV3 *)
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

(* Deprecated. Use TensorArrayGradV3 *)
val tensorArrayWriteV2
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `int32 ] t
  -> 't t
  -> [ `float ] t
  -> [ `float ] t

(* Outputs a `Summary` protocol buffer with a tensor. *)
val tensorSummary
  :  ?name:string
  -> ?description:string
  -> ?display_name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `string ] t

(* A Reader that outputs the lines of a file delimited by '\n'. *)
val textLineReader
  :  ?name:string
  -> ?skip_header_lines:int
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Generates labels for candidate sampling with a learned unigram distribution. *)
(* See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels. *)
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

(* Constructs a tensor by tiling a given tensor. *)
(* This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`. *)
val tile
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tmultiples) t
  -> 't t

(* Returns the gradient of `Tile`. *)
(* Since `Tile` takes an input and repeats the input `multiples` times
along each dimension, `TileGrad` takes in `multiples` and aggregates
each repeated tile of `input` into `output`. *)
val tileGrad
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> [ `int32 ] t
  -> 't t

(* Finds values and indices of the `k` largest elements for the last dimension. *)
(* If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

    values.shape = indices.shape = input.shape[:-1] + [k]

If two elements are equal, the lower-index element appears first.

If `k` varies dynamically, use `TopKV2` below. *)
val topK
  :  ?name:string
  -> k:int
  -> ?sorted:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t * [ `int32 ] t

(* Finds values and indices of the `k` largest elements for the last dimension. *)
(* If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

    values.shape = indices.shape = input.shape[:-1] + [k]

If two elements are equal, the lower-index element appears first.

This is the same as `TopK`, but takes `k` as in input rather than an attr. *)
val topKV2
  :  ?name:string
  -> ?sorted:bool
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t * [ `int32 ] t

(* Shuffle dimensions of x according to a permutation. *)
(* The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]` *)
val transpose
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> ([< `int32 | `int64 ] as 'tperm) t
  -> 't t

(* Returns x / y element-wise for integer types. *)
(* Truncation designates that negative numbers will round fractional quantities
toward zero. I.e. -7 / 5 = 1. This matches C semantics but it is different
than Python semantics. See `FloorDiv` for a division function that matches
Python Semantics.

*NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val truncateDiv
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Returns element-wise remainder of division. This emulates C semantics where *)
(* true, this follows C semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

*NOTE*: `Mod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) *)
val truncateMod
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

(* Outputs random values from a truncated normal distribution. *)
(* The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked. *)
val truncatedNormal
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ?control_inputs:Node.p list
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `float | `double ] as 'dtype) t

(* Generates labels for candidate sampling with a uniform distribution. *)
(* See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels. *)
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

(* Finds unique elements in a 1-D tensor. *)
(* This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

```prettyprint
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
``` *)
val unique
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t * ([< `int32 | `int64 ] as 'out_idx) t

(* Finds unique elements in a 1-D tensor. *)
(* This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. Finally, it returns a third tensor `count` that
contains the count of each element of `y` in `x`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

```prettyprint
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx, count = unique_with_counts(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
count ==> [2, 1, 3, 1, 2]
``` *)
val uniqueWithCounts
  :  ?name:string
  -> type_1:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> type_2:([< `int32 | `int64 ] as 'out_idx) Type.t
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t * ([< `int32 | `int64 ] as 'out_idx) t * ([< `int32 | `int64 ] as 'out_idx) t

(* Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors. *)
(* Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
For example, given a tensor of shape `(A, B, C, D)`;

If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
  and each tensor in `output` will have shape `(B, C, D)`. (Note that the
  dimension unpacked along is gone, unlike `split`).

If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
  and each tensor in `output` will have shape `(A, C, D)`.
Etc.

This is the opposite of `pack`. *)
val unpack
  :  ?name:string
  -> num:int
  -> ?axis:int
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t list

(* Computes the sum along segments of a tensor. *)
(* Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
`(output[i] = sum_{j...} data[j...]` where the sum is over tuples `j...` such
that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
need not be sorted and need not cover all values in the full
range of valid values.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.

`num_segments` should equal the number of distinct segment IDs.

<div style='width:70%; margin:auto; margin-bottom:10px; margin-top:20px;'>
<img style='width:100%' src='../../images/UnsortedSegmentSum.png' alt>
</div> *)
val unsortedSegmentSum
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Use VariableV2 instead. *)
val variable
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

(* Holds state in the form of a tensor that persists across steps. *)
(* Outputs a ref to the tensor state so it may be read or modified.
TODO(zhifengc/mrry): Adds a pointer to a more detail document
about sharing states in tensorflow. *)
val variableV2
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> 'dtype t

(* Returns locations of true values in a boolean tensor. *)
(* This operation returns the coordinates of true elements in `input`. The
coordinates are returned in a 2-D tensor where the first dimension (rows)
represents the number of true elements, and the second dimension (columns)
represents the coordinates of the true elements. Keep in mind, the shape of
the output tensor can vary depending on how many true values there are in
`input`. Indices are output in row-major order.

For example:

```prettyprint
# 'input' tensor is [[True, False]
#                    [True, False]]
# 'input' has two true values, so output has two coordinates.
# 'input' has rank of 2, so coordinates have two indices.
where(input) ==> [[0, 0],
                  [1, 0]]

# `input` tensor is [[[True, False]
#                     [True, False]]
#                    [[False, True]
#                     [False, True]]
#                    [[False, False]
#                     [False, True]]]
# 'input' has 5 true values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
``` *)
val where
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `bool ] t
  -> [ `int64 ] t

(* A Reader that outputs the entire contents of a file as a value. *)
(* To use, enqueue filenames in a Queue.  The output of ReaderRead will
be a filename (key) and the contents of that file (value). *)
val wholeFileReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> ?control_inputs:Node.p list
  -> unit
  -> [ `string ] t

(* Writes contents to the file at input filename. Creates file if not existing. *)
val writeFile
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> [ `string ] t
  -> [ `string ] t
  -> [ `unit ] t

(* Returns a tensor of zeros with the same shape and type as x. *)
val zerosLike
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> 't t
  -> 't t

(* Compute the Hurwitz zeta function \\(\zeta(x, q)\\). *)
(* The Hurwitz zeta function is defined as:

```
\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}
``` *)
val zeta
  :  ?name:string
  -> ?control_inputs:Node.p list
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

