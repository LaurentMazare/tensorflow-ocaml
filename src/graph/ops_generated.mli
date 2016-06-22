(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)
open Node

module Op_names : sig
  val abort : Op_name.t
  val abs : Op_name.t
  val add : Op_name.t
  val addN : Op_name.t
  val adjustContrast : Op_name.t
  val adjustContrastv2 : Op_name.t
  val all : Op_name.t
  val allCandidateSampler : Op_name.t
  val any : Op_name.t
  val applyAdadelta : Op_name.t
  val applyAdagrad : Op_name.t
  val applyAdam : Op_name.t
  val applyFtrl : Op_name.t
  val applyGradientDescent : Op_name.t
  val applyMomentum : Op_name.t
  val applyRMSProp : Op_name.t
  val argMax : Op_name.t
  val argMin : Op_name.t
  val assign : Op_name.t
  val assignAdd : Op_name.t
  val assignSub : Op_name.t
  val audioSummary : Op_name.t
  val avgPool : Op_name.t
  val avgPool3D : Op_name.t
  val avgPool3DGrad : Op_name.t
  val avgPoolGrad : Op_name.t
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
  val batchMatrixSolve : Op_name.t
  val batchMatrixSolveLs : Op_name.t
  val batchMatrixTriangularSolve : Op_name.t
  val batchNormWithGlobalNormalization : Op_name.t
  val batchNormWithGlobalNormalizationGrad : Op_name.t
  val batchSelfAdjointEig : Op_name.t
  val batchToSpace : Op_name.t
  val biasAdd : Op_name.t
  val biasAddGrad : Op_name.t
  val biasAddV1 : Op_name.t
  val bitcast : Op_name.t
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
  val conj : Op_name.t
  val controlTrigger : Op_name.t
  val conv2D : Op_name.t
  val conv2DBackpropFilter : Op_name.t
  val conv2DBackpropInput : Op_name.t
  val conv3D : Op_name.t
  val conv3DBackpropFilter : Op_name.t
  val conv3DBackpropInput : Op_name.t
  val cos : Op_name.t
  val countUpTo : Op_name.t
  val cross : Op_name.t
  val decodeJSONExample : Op_name.t
  val decodePng : Op_name.t
  val decodeRaw : Op_name.t
  val deleteSessionTensor : Op_name.t
  val depthToSpace : Op_name.t
  val depthwiseConv2dNative : Op_name.t
  val depthwiseConv2dNativeBackpropFilter : Op_name.t
  val depthwiseConv2dNativeBackpropInput : Op_name.t
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
  val encodePng : Op_name.t
  val enter : Op_name.t
  val equal : Op_name.t
  val erf : Op_name.t
  val erfc : Op_name.t
  val exit : Op_name.t
  val exp : Op_name.t
  val expandDims : Op_name.t
  val extractGlimpse : Op_name.t
  val extractImagePatches : Op_name.t
  val fFT : Op_name.t
  val fFT2D : Op_name.t
  val fFT3D : Op_name.t
  val fIFOQueue : Op_name.t
  val fact : Op_name.t
  val fill : Op_name.t
  val fixedLengthRecordReader : Op_name.t
  val fixedUnigramCandidateSampler : Op_name.t
  val floor : Op_name.t
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
  val inv : Op_name.t
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
  val logSoftmax : Op_name.t
  val logUniformCandidateSampler : Op_name.t
  val logicalAnd : Op_name.t
  val logicalNot : Op_name.t
  val logicalOr : Op_name.t
  val lookupTableFind : Op_name.t
  val lookupTableSize : Op_name.t
  val loopCond : Op_name.t
  val matMul : Op_name.t
  val matchingFiles : Op_name.t
  val matrixDeterminant : Op_name.t
  val matrixInverse : Op_name.t
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
  val min : Op_name.t
  val minimum : Op_name.t
  val mirrorPad : Op_name.t
  val mirrorPadGrad : Op_name.t
  val mod_ : Op_name.t
  val mul : Op_name.t
  val multinomial : Op_name.t
  val neg : Op_name.t
  val negTrain : Op_name.t
  val nextIteration : Op_name.t
  val noOp : Op_name.t
  val notEqual : Op_name.t
  val oneHot : Op_name.t
  val pack : Op_name.t
  val pad : Op_name.t
  val paddingFIFOQueue : Op_name.t
  val placeholder : Op_name.t
  val placeholderWithDefault : Op_name.t
  val polygamma : Op_name.t
  val pow : Op_name.t
  val prod : Op_name.t
  val queueClose : Op_name.t
  val queueSize : Op_name.t
  val rGBToHSV : Op_name.t
  val randomCrop : Op_name.t
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
  val readerReset : Op_name.t
  val readerRestoreState : Op_name.t
  val readerSerializeState : Op_name.t
  val real : Op_name.t
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
  val rsqrt : Op_name.t
  val sampleDistortedBoundingBox : Op_name.t
  val scalarSummary : Op_name.t
  val scatterAdd : Op_name.t
  val scatterSub : Op_name.t
  val scatterUpdate : Op_name.t
  val segmentMax : Op_name.t
  val segmentMean : Op_name.t
  val segmentMin : Op_name.t
  val segmentProd : Op_name.t
  val segmentSum : Op_name.t
  val select : Op_name.t
  val selfAdjointEig : Op_name.t
  val serializeManySparse : Op_name.t
  val serializeSparse : Op_name.t
  val shape : Op_name.t
  val shapeN : Op_name.t
  val shardedFilename : Op_name.t
  val shardedFilespec : Op_name.t
  val sigmoid : Op_name.t
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
  val spaceToDepth : Op_name.t
  val sparseAdd : Op_name.t
  val sparseAddGrad : Op_name.t
  val sparseApplyAdadelta : Op_name.t
  val sparseApplyAdagrad : Op_name.t
  val sparseApplyFtrl : Op_name.t
  val sparseApplyMomentum : Op_name.t
  val sparseConcat : Op_name.t
  val sparseDenseCwiseAdd : Op_name.t
  val sparseDenseCwiseDiv : Op_name.t
  val sparseDenseCwiseMul : Op_name.t
  val sparseMatMul : Op_name.t
  val sparseReduceSum : Op_name.t
  val sparseReorder : Op_name.t
  val sparseSegmentMean : Op_name.t
  val sparseSegmentMeanGrad : Op_name.t
  val sparseSegmentSqrtN : Op_name.t
  val sparseSegmentSqrtNGrad : Op_name.t
  val sparseSegmentSum : Op_name.t
  val sparseSoftmax : Op_name.t
  val sparseSoftmaxCrossEntropyWithLogits : Op_name.t
  val sparseTensorDenseAdd : Op_name.t
  val sparseTensorDenseMatMul : Op_name.t
  val sparseToDense : Op_name.t
  val split : Op_name.t
  val sqrt : Op_name.t
  val square : Op_name.t
  val squaredDifference : Op_name.t
  val squeeze : Op_name.t
  val stack : Op_name.t
  val stackClose : Op_name.t
  val stackPop : Op_name.t
  val stackPush : Op_name.t
  val stopGradient : Op_name.t
  val stringToHashBucket : Op_name.t
  val stringToHashBucketFast : Op_name.t
  val stringToHashBucketStrong : Op_name.t
  val stringToNumber : Op_name.t
  val sub : Op_name.t
  val sum : Op_name.t
  val switch : Op_name.t
  val tFRecordReader : Op_name.t
  val tanh : Op_name.t
  val temporaryVariable : Op_name.t
  val tensorArray : Op_name.t
  val tensorArrayClose : Op_name.t
  val tensorArrayConcat : Op_name.t
  val tensorArrayGrad : Op_name.t
  val tensorArrayPack : Op_name.t
  val tensorArrayRead : Op_name.t
  val tensorArraySize : Op_name.t
  val tensorArraySplit : Op_name.t
  val tensorArrayUnpack : Op_name.t
  val tensorArrayWrite : Op_name.t
  val textLineReader : Op_name.t
  val threadUnsafeUnigramCandidateSampler : Op_name.t
  val tile : Op_name.t
  val tileGrad : Op_name.t
  val topK : Op_name.t
  val topKV2 : Op_name.t
  val transpose : Op_name.t
  val truncatedNormal : Op_name.t
  val uniformCandidateSampler : Op_name.t
  val unique : Op_name.t
  val uniqueWithCounts : Op_name.t
  val unpack : Op_name.t
  val unsortedSegmentSum : Op_name.t
  val variable : Op_name.t
  val where : Op_name.t
  val wholeFileReader : Op_name.t
  val zerosLike : Op_name.t
  val zeta : Op_name.t
end

(* Raise a exception to abort the process when called. *)
(* Returns nothing but an exception. *)
val abort
  :  ?name:string
  -> ?error_msg:string
  -> unit
  -> [ `unit ] t

(* Computes the absolute value of a tensor. *)
(* Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\). *)
val abs
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Returns x + y element-wise. *)
(* *NOTE*: Add supports broadcasting. AddN does not. *)
val add
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) t

(* Add all input tensors element wise. *)
val addN
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t list
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Deprecated. Disallowed in GraphDef version >= 2. *)
val adjustContrast
  :  ?name:string
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
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Computes the "logical and" of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val all
  :  ?name:string
  -> ?keep_dims:bool
  -> [ `bool ] t
  -> [ `int32 ] t
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
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

(* Computes the "logical or" of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val any
  :  ?name:string
  -> ?keep_dims:bool
  -> [ `bool ] t
  -> [ `int32 ] t
  -> [ `bool ] t

(* Update '*var' according to the adadelta scheme. *)
(* accum = rho() * accum + (1 - rho()) * grad.square();
update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
update_accum = rho() * update_accum + (1 - rho()) * update.square();
var -= update; *)
val applyAdadelta
  :  ?name:string
  -> ?use_locking:bool
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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the Adam algorithm. *)
(* lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon) *)
val applyAdam
  :  ?name:string
  -> ?use_locking:bool
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

(* Update '*var' according to the Ftrl-proximal scheme. *)
(* accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new *)
val applyFtrl
  :  ?name:string
  -> ?use_locking:bool
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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the momentum scheme. *)
(* accum = accum * momentum + grad
var -= lr * accum *)
val applyMomentum
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update '*var' according to the RMSProp algorithm. *)
(* mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom *)
val applyRMSProp
  :  ?name:string
  -> ?use_locking:bool
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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int32 ] t
  -> [ `int64 ] t

(* Returns the index with the smallest value across dimensions of a tensor. *)
val argMin
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int32 ] t
  -> [ `int64 ] t

(* Update 'ref' by assigning 'value' to it. *)
(* This operation outputs "ref" after the assignment is done.
This makes it easier to chain operations that need to use the reset value. *)
val assign
  :  ?name:string
  -> ?validate_shape:bool
  -> ?use_locking:bool
  -> 't t
  -> 't t
  -> 't t

(* Update 'ref' by adding 'value' to it. *)
(* This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value. *)
val assignAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Update 'ref' by subtracting 'value' from it. *)
(* This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value. *)
val assignSub
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

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
  -> [ `string ] t
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
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Performs 3D average pooling on the input. *)
val avgPool3D
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes gradients of average pooling function. *)
val avgPool3DGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
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
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Calculates the Cholesky decomposition of a batch of square matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix Cholesky
decomposition above. The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`. *)
val batchCholesky
  :  ?name:string
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Calculates the reverse mode backpropagated gradient of the Cholesky algorithm. *)
(* For an explanation see "Differentiation of the Cholesky algorithm" by
Iain Murray http://arxiv.org/abs/1602.07527. *)
val batchCholeskyGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Compute the 1-dimensional discrete Fourier Transform over the inner-most *)
(* dimension of `input`. *)
val batchFFT
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the 2-dimensional discrete Fourier Transform over the inner-most *)
(* 2 dimensions of `input`. *)
val batchFFT2D
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the 3-dimensional discrete Fourier Transform over the inner-most 3 *)
(* dimensions of `input`. *)
val batchFFT3D
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the inverse 1-dimensional discrete Fourier Transform over the inner-most *)
(* dimension of `input`. *)
val batchIFFT
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the inverse 2-dimensional discrete Fourier Transform over the inner-most *)
(* 2 dimensions of `input`. *)
val batchIFFT2D
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the inverse 3-dimensional discrete Fourier Transform over the inner-most *)
(* 3 dimensions of `input`. *)
val batchIFFT3D
  :  ?name:string
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
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t

(* Copy a tensor setting everything outside a central band in each innermost matrix *)
(* to zero.

The `band` part is computed as follows:
Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor with the same shape where

`band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

The indicator function 'in_band(m, n)` is one if
`(num_lower < 0 || (m-n) <= num_lower)) &&
(num_upper < 0 || (n-m) <= num_upper)`, and zero otherwise.

For example:

```prettyprint
# if 'input' is [[ 0,  1,  2, 3]
                 [-1,  0,  1, 2]
                 [-2, -1,  0, 1]
                 [-3, -2, -1, 0]],

tf.batch_matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                             [-1,  0,  1, 2]
                                             [ 0, -1,  0, 1]
                                             [ 0,  0, -1, 0]],

tf.batch_matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                            [-1,  0,  1, 0]
                                            [-2, -1,  0, 1]
                                            [ 0, -2, -1, 0]]
```

Useful special cases:

```prettyprint
 tf.batch_matrix_band_part(input, 0, -1) ==> Upper triangular part.
 tf.batch_matrix_band_part(input, -1, 0) ==> Lower triangular part.
 tf.batch_matrix_band_part(input, 0, 0) ==> Diagonal.
``` *)
val batchMatrixBandPart
  :  ?name:string
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> 't t

(* Calculates the determinants for a batch of square matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a 1-D tensor containing the determinants
for all input submatrices `[..., :, :]`. *)
val batchMatrixDeterminant
  :  ?name:string
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

tf.batch_matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                     [0, 2, 0, 0]
                                     [0, 0, 3, 0]
                                     [0, 0, 0, 4]],
                                    [[5, 0, 0, 0]
                                     [0, 6, 0, 0]
                                     [0, 0, 7, 0]
                                     [0, 0, 0, 8]]]

which has shape (2, 4, 4)
``` *)
val batchMatrixDiag
  :  ?name:string
  -> 't t
  -> 't t

(* Returns the batched diagonal part of a batched tensor. *)
(* This operation returns a tensor with the `diagonal` part
of the batched `input`. The `diagonal` part is computed as follows:

Assume `input` has `k` dimensions `[I, J, K, ..., N, N]`, then the output is a
tensor of rank `k - 1` with dimensions `[I, J, K, ..., N]` where:

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

tf.batch_matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]

which has shape (2, 4)
``` *)
val batchMatrixDiagPart
  :  ?name:string
  -> 't t
  -> 't t

(* Calculates the inverse of square invertible matrices or their adjoints *)
(* (conjugate transposes).

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.

The op uses LU decomposition with partial pivoting to compute the inverses.

If a matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result. *)
val batchMatrixInverse
  :  ?name:string
  -> ?adjoint:bool
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Solves systems of linear equations. Checks for invertibility. *)
(* Matrix is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. Rhs is a tensor of shape
`[..., M, K]`. The output is a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output
matrix satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `True` then each output
matrix satisfies `adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`. *)
val batchMatrixSolve
  :  ?name:string
  -> ?adjoint:bool
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Solves multiple linear least-squares problems. *)
(* `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
form square matrices. Rhs is a tensor of shape `[..., M, K]`. The output
is a tensor shape `[..., N, K]` where each output matrix solves each of
the equations matrix[..., :, :] * output[..., :, :] = rhs[..., :, :] in the
least squares sense.

Below we will use the following notation for each pair of
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
val batchMatrixSolveLs
  :  ?name:string
  -> ?fast:bool
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
`rhs` is a tensor of shape [..., M, K]`.

The output is a tensor of shape `[..., M, K]`. If `adjoint` is `True` then the
innermost matrices in output` satisfy matrix equations
`matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `False` then the strictly then the  innermost matrices in
`output` satisfy matrix equations
`adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`. *)
val batchMatrixTriangularSolve
  :  ?name:string
  -> ?lower:bool
  -> ?adjoint:bool
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Batch normalization. *)
(* This op is deprecated. Prefer `tf.nn.batch_normalization`. *)
val batchNormWithGlobalNormalization
  :  ?name:string
  -> variance_epsilon:float
  -> scale_after_normalization:bool
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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Calculates the Eigen Decomposition of a batch of square self-adjoint matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix
SelfAdjointEig.

The result is a '[..., M+1, M] matrix with [..., 0,:] containing the
eigenvalues, and subsequent [...,1:, :] containing the eigenvectors. *)
val batchSelfAdjointEig
  :  ?name:string
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* BatchToSpace for 4-D tensors of type T. *)
(* Rearranges (permutes) data from batch into blocks of spatial data, followed by
cropping. This is the reverse transformation of SpaceToBatch. More specifically,
this op outputs a copy of the input tensor where values from the `batch`
dimension are moved in spatial blocks to the `height` and `width` dimensions,
followed by cropping along the `height` and `width` dimensions. *)
val batchToSpace
  :  ?name:string
  -> block_size:int
  -> 't t
  -> [ `int32 ] t
  -> 't t

(* Adds `bias` to `value`. *)
(* This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions. *)
val biasAdd
  :  ?name:string
  -> ?data_format:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* The backward operation for "BiasAdd" on the "bias" tensor. *)
(* It accumulates all the values from out_backprop into the feature dimension.
For NHWC data format, the feature dimension is the last. For NCHW data format,
the feature dimension is the third-to-last. *)
val biasAddGrad
  :  ?name:string
  -> ?data_format:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Adds `bias` to `value`. *)
(* This is a deprecated version of BiasAdd and will be soon removed.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions. *)
val biasAddV1
  :  ?name:string
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
[..., sizeof(`type`)/sizeof(`T`)] to [...]. *)
val bitcast
  :  ?name:string
  -> type_:([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) Type.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) t

(* Return the reduction indices for computing gradients of s0 op s1 with broadcast. *)
(* This is typically used by gradient computations for a broadcasting operation. *)
val broadcastGradientArgs
  :  ?name:string
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `int32 ] t * [ `int32 ] t

(* Performs greedy decoding on the logits given in inputs. *)
(* A note about the attribute merge_repeated: if enabled, when
consecutive logits' maximum indices are the same, only the first of
these is emitted.  Labeling the blank '*', the sequence "A B B * B B"
becomes "A B" if merge_repeated = True and "A B B B B" if
merge_repeated = False.

Regardless of the value of merge_repeated, if the maximum index of a given
time and batch corresponds to the blank, index `(num_classes - 1)`, no new
element is emitted. *)
val cTCGreedyDecoder
  :  ?name:string
  -> ?merge_repeated:bool
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
  -> [ `float ] t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `float ] t * [ `float ] t

(* Cast x of type SrcT to y of DstT. *)
val cast
  :  ?name:string
  -> type_:'dstT Type.t
  -> 'srcT t
  -> 'dstT t

(* Returns element-wise smallest integer in not less than x. *)
val ceil
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Checks a tensor for NaN and Inf values. *)
(* When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is. *)
val checkNumerics
  :  ?name:string
  -> message:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Calculates the Cholesky decomposition of a square matrix. *)
(* The input has to be symmetric and positive definite. Only the lower-triangular
part of the input will be used for this operation. The upper-triangular part
will not be read.

The result is the lower-triangular matrix of the Cholesky decomposition of the
input, `L`, so that `input = L L^*`. *)
val cholesky
  :  ?name:string
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Calculates the reverse mode backpropagated gradient of the Cholesky algorithm. *)
(* For an explanation see "Differentiation of the Cholesky algorithm" by
Iain Murray http://arxiv.org/abs/1602.07527. *)
val choleskyGrad
  :  ?name:string
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
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `complex64 ] as 'tout) t

(* Computes the complex absolute value of a tensor. *)
(* Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` or `double` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
value is computed as \\( \sqrt{a^2 + b^2}\\).

For example:

```
# tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
tf.complex_abs(x) ==> [5.25594902, 6.60492229]
``` *)
val complexAbs
  :  ?name:string
  -> type_:([< `float | `double ] as 'tout) Type.t
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
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int32 ] t * [ `int64 ] t * [ `float ] t

(* Concatenates tensors along one dimension. *)
val concat
  :  ?name:string
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
  -> [ `int32 ] t
  -> [ `int32 ] t list
  -> [ `int32 ] t list

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
  -> ([< `complex64 ] as 't) t
  -> ([< `complex64 ] as 't) t

(* Does nothing. Serves as a control trigger for scheduling. Only useful as a *)
(* placeholder for control edges. *)
val controlTrigger
  :  ?name:string
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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the gradients of 3D convolution with respect to the filter. *)
val conv3DBackpropFilter
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes the gradients of 3D convolution with respect to the input. *)
val conv3DBackpropInput
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes cos of x element-wise. *)
val cos
  :  ?name:string
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Increments 'ref' until it reaches 'limit'. *)
(* This operation outputs "ref" after the update is done.  This makes it
easier to chain operations that need to use the updated value. *)
val countUpTo
  :  ?name:string
  -> limit:int
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 't) t

(* Compute the pairwise cross product. *)
(* `a` and `b` must be the same shape; they can either be simple 3-element vectors,
or any shape where the innermost dimension is 3. In the latter case, each pair
of corresponding 3-element vectors is cross-multiplied independently. *)
val cross
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Convert JSON-encoded Example records to binary protocol buffer strings. *)
(* This op translates a tensor containing Example records, encoded using
the [standard JSON
mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
into a tensor containing the same records encoded as binary protocol
buffers. The resulting tensor can then be fed to any of the other
Example-parsing ops. *)
val decodeJSONExample
  :  ?name:string
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
  -> [ `string ] t
  -> 'dtype t

(* Reinterpret the bytes of a string as a vector of numbers. *)
val decodeRaw
  :  ?name:string
  -> type_:([< `float | `double | `int32 | `int64 ] as 'out_type) Type.t
  -> ?little_endian:bool
  -> [ `string ] t
  -> ([< `float | `double | `int32 | `int64 ] as 'out_type) t

(* Delete the tensor specified by its handle in the session. *)
val deleteSessionTensor
  :  ?name:string
  -> [ `string ] t
  -> [ `unit ] t

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
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the gradients of depthwise convolution with respect to the filter. *)
val depthwiseConv2dNativeBackpropFilter
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the gradients of depthwise convolution with respect to the input. *)
val depthwiseConv2dNativeBackpropInput
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

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
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes Psi, the derivative of Lgamma (the log of the absolute value of *)
(* `Gamma(x)`), element-wise. *)
val digamma
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors. *)
(* The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
`filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
input channel is processed independently of the others with its own structuring
function. The `output` tensor has shape
`[batch, out_height, out_width, depth]`. The spatial dimensions of the output
tensor depend on the `padding` algorithm. We currently only support the default
"NHWC" `data_format`.

In detail, the grayscale morphological 2-D dilation is the max-sum correlation
(for consistency with `conv2d`, we use unmirrored filters):

    output[b, y, x, c] =
       max_{dy, dx} input[b,
                          strides[1] * y + rates[1] * dy,
                          strides[2] * x + rates[2] * dx,
                          c] +
                    filter[dy, dx, c]

Max-pooling is a special case when the filter has size equal to the pooling
kernel size and contains all zeros. *)
val dilation2D
  :  ?name:string
  -> strides:int list
  -> rates:int list
  -> padding:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes the gradient of morphological 2-D dilation with respect to the filter. *)
val dilation2DBackpropFilter
  :  ?name:string
  -> strides:int list
  -> rates:int list
  -> padding:string
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
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Returns x / y element-wise. *)
val div
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Draw bounding boxes on a batch of images. *)
(* Outputs a copy of `images` but draws on top of the pixels zero or more bounding
boxes specified by the locations in `boxes`. The coordinates of the each
bounding box in `boxes are encoded as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example, if an image is 100 x 200 pixels and the bounding box is
`[0.1, 0.5, 0.2, 0.9]`, the bottom-left and upper-right coordinates of the
bounding box will be `(10, 40)` to `(50, 180)`.

Parts of the bounding box may fall outside the image. *)
val drawBoundingBoxes
  :  ?name:string
  -> ([< `float ] as 't) t
  -> [ `float ] t
  -> ([< `float ] as 't) t

(* Partitions `data` into `num_partitions` tensors using indices from `partitions`. *)
(* For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
are placed in `outputs[i]` in lexicographic order of `js`, and the first
dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
In detail,

    outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

    outputs[i] = pack([data[js, ...] for js if partitions[js] == i])

`data.shape` must start with `partitions.shape`.

For example:

    # Scalar partitions
    partitions = 1
    num_partitions = 2
    data = [10, 20]
    outputs[0] = []  # Empty with shape [0, 2]
    outputs[1] = [[10, 20]]

    # Vector partitions
    partitions = [0, 0, 1, 1, 0]
    num_partitions = 2
    data = [10, 20, 30, 40, 50]
    outputs[0] = [10, 20, 50]
    outputs[1] = [30, 40]

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/DynamicPartition.png" alt>
</div> *)
val dynamicPartition
  :  ?name:string
  -> num_partitions:int
  -> 't t
  -> [ `int32 ] t
  -> 't t list

(* Interleave the values from the `data` tensors into a single tensor. *)
(* Builds a merged tensor such that

    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]

For example, if each `indices[m]` is scalar or vector, we have

    # Scalar indices
    merged[indices[m], ...] = data[m][...]

    # Vector indices
    merged[indices[m][i], ...] = data[m][i, ...]

Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is

    merged.shape = [max(indices)] + constant

Values are merged in order, so if an index appears in both `indices[m][i]` and
`indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
merged result.

For example:

    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/DynamicStitch.png" alt>
</div> *)
val dynamicStitch
  :  ?name:string
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
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes gradients for the exponential linear (Elu) operation. *)
val eluGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

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
  -> 't t
  -> 't t

(* Returns the truth value of (x == y) element-wise. *)
val equal
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string | `bool ] as 't) t
  -> [ `bool ] t

(* Computes the Gauss error function of `x` element-wise. *)
val erf
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes the complementary error function of `x` element-wise. *)
val erfc
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Exits the current frame to its parent frame. *)
(* Exit makes its input `data` available to the parent frame. *)
val exit
  :  ?name:string
  -> 't t
  -> 't t

(* Computes exponential of x element-wise.  \\(y = e^x\\). *)
val exp
  :  ?name:string
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
  -> 't t
  -> [ `int32 ] t
  -> 't t

(* Extracts a glimpse from the input tensor. *)
(* Returns a set of windows called glimpses extracted at location
`offsets` from the input tensor. If the windows only partially
overlaps the inputs, the non overlapping areas will be filled with
random noise.

The result is a 4-D tensor of shape `[batch_size, glimpse_height,
glimpse_width, channels]`. The channels and batch dimensions are the
same as that of the input tensor. The height and width of the output
windows are specified in the `size` parameter.

The argument `normalized` and `centered` controls how the windows are *)
val extractGlimpse
  :  ?name:string
  -> ?centered:bool
  -> ?normalized:bool
  -> ?uniform_noise:bool
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `float ] t

(* Extract `patches` from `images` and puth them in the "depth" output dimension. *)
val extractImagePatches
  :  ?name:string
  -> ?ksizes:int list
  -> ?strides:int list
  -> ?rates:int list
  -> padding:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Compute the 1-dimensional discrete Fourier Transform. *)
val fFT
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the 2-dimensional discrete Fourier Transform. *)
val fFT2D
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the 3-dimensional discrete Fourier Transform. *)
val fFT3D
  :  ?name:string
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
  -> unit
  -> [ `string ] t

(* Output a fact about factorials. *)
val fact
  :  ?name:string
  -> unit
  -> [ `string ] t

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
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

(* Returns element-wise largest integer not greater than x. *)
val floor
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Gather slices from `params` according to `indices`. *)
(* `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]

    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]

    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]

If `indices` is a permutation and `len(indices) == params.shape[0]` then
this operation will permute `params` accordingly.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/Gather.png" alt>
</div> *)
val gather
  :  ?name:string
  -> ?validate_indices:bool
  -> 'tparams t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 'tparams t

(* Gather values from `params` according to `indices`. *)
(* `indices` must be integer tensor, containing indices into `params`.
It must be shape `[d_0, ..., d_N, R]` where `R` is the rank of `params`.
The innermost dimension of `indices` (with length `R`) corresponds to the
indices of `params`.

Produces an output tensor with shape `[d_0, ..., d_{n-1}]` where:

    output[i, j, k, ...] = params[indices[i, j, k, ..., :]]

e.g. for `indices` a matrix:

    output[i] = params[indices[i, :]] *)
val gatherNd
  :  ?name:string
  -> 'tparams t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 'tparams t

(* Store the input tensor in the state of the current session. *)
val getSessionHandle
  :  ?name:string
  -> 't t
  -> [ `string ] t

(* Get the value of the tensor specified by its handle. *)
val getSessionTensor
  :  ?name:string
  -> type_:'dtype Type.t
  -> [ `string ] t
  -> 'dtype t

(* Returns the truth value of (x > y) element-wise. *)
val greater
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

(* Returns the truth value of (x >= y) element-wise. *)
val greaterEqual
  :  ?name:string
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
  -> [ `float ] t
  -> [ `float ] t

(* Creates a non-initialized hash table. *)
(* This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable. *)
val hashTable
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] t

(* Outputs a `Summary` protocol buffer with a histogram. *)
(* The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite. *)
val histogramSummary
  :  ?name:string
  -> [ `string ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `string ] t

(* Compute the inverse 1-dimensional discrete Fourier Transform. *)
val iFFT
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the inverse 2-dimensional discrete Fourier Transform. *)
val iFFT2D
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Compute the inverse 3-dimensional discrete Fourier Transform. *)
val iFFT3D
  :  ?name:string
  -> [ `complex64 ] t
  -> [ `complex64 ] t

(* Return a tensor with the same shape and contents as the input tensor or value. *)
val identity
  :  ?name:string
  -> 't t
  -> 't t

(* A Reader that outputs the queued work as both the key and value. *)
(* To use, enqueue strings in a Queue.  ReaderRead will take the front
work string and output (work, work). *)
val identityReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] t

(* Compute the lower regularized incomplete Gamma function `Q(a, x)`. *)
(* The lower regularized incomplete Gamma function is defined as:

```
P(a, x) = gamma(a, x) / Gamma(x) = 1 - Q(a, x)
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
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Compute the upper regularized incomplete Gamma function `Q(a, x)`. *)
(* The upper regularized incomplete Gamma function is defined as:

```
Q(a, x) = Gamma(a, x) / Gamma(x) = 1 - P(a, x)
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
  -> [ `float ] t
  -> ([< `int32 | `int64 ] as 't) t
  -> [ `bool ] t

(* Table initializer that takes two tensors for keys and values respectively. *)
val initializeTable
  :  ?name:string
  -> [ `string ] t
  -> 'tkey t
  -> 'tval t
  -> [ `unit ] t

(* Computes the reciprocal of x element-wise. *)
(* I.e., \\(y = 1 / x\\). *)
val inv
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

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
  -> [ `int32 ] t
  -> [ `int32 ] t

(* Returns which elements of x are finite. *)
val isFinite
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> [ `bool ] t

(* Returns which elements of x are Inf. *)
val isInf
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> [ `bool ] t

(* Returns which elements of x are NaN. *)
val isNan
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> [ `bool ] t

(* Checks whether a tensor has been initialized. *)
(* Outputs boolean scalar indicating whether the tensor has been initialized. *)
val isVariableInitialized
  :  ?name:string
  -> 'dtype t
  -> [ `bool ] t

(* L2 Loss. *)
(* Computes half the L2 norm of a tensor without the `sqrt`:

    output = sum(t ** 2) / 2 *)
val l2Loss
  :  ?name:string
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
convolutional neural networks (NIPS 2012)]
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). *)
val lRN
  :  ?name:string
  -> ?depth_radius:int
  -> ?bias:float
  -> ?alpha:float
  -> ?beta:float
  -> [ `float ] t
  -> [ `float ] t

(* Gradients for Local Response Normalization. *)
val lRNGrad
  :  ?name:string
  -> ?depth_radius:int
  -> ?bias:float
  -> ?alpha:float
  -> ?beta:float
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

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
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

(* Returns the truth value of (x < y) element-wise. *)
val less
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

(* Returns the truth value of (x <= y) element-wise. *)
val lessEqual
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `bool ] t

(* Computes the log of the absolute value of `Gamma(x)` element-wise. *)
val lgamma
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Generates values in an interval. *)
(* A sequence of `num` evenly-spaced values are generated beginning at `start`.
If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
so that the last one is exactly `stop`.

For example:

```
tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
``` *)
val linSpace
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
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
  -> 't t
  -> 't t
  -> 't t * [ `int32 ] t

(* Computes natural logarithm of x element-wise. *)
(* I.e., \\(y = \log_e x\\). *)
val log
  :  ?name:string
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes log softmax activations. *)
(* For each batch `i` and class `j` we have

    logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i]))) *)
val logSoftmax
  :  ?name:string
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
  -> [ `int64 ] t
  -> [ `int64 ] t * [ `float ] t * [ `float ] t

(* Returns the truth value of x AND y element-wise. *)
val logicalAnd
  :  ?name:string
  -> [ `bool ] t
  -> [ `bool ] t
  -> [ `bool ] t

(* Returns the truth value of NOT x element-wise. *)
val logicalNot
  :  ?name:string
  -> [ `bool ] t
  -> [ `bool ] t

(* Returns the truth value of x OR y element-wise. *)
val logicalOr
  :  ?name:string
  -> [ `bool ] t
  -> [ `bool ] t
  -> [ `bool ] t

(* Looks up keys in a table, outputs the corresponding values. *)
(* The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values. *)
val lookupTableFind
  :  ?name:string
  -> [ `string ] t
  -> 'tin t
  -> 'tout t
  -> 'tout t

(* Computes the number of elements in the given table. *)
val lookupTableSize
  :  ?name:string
  -> [ `string ] t
  -> [ `int64 ] t

(* Forwards the input to the output. *)
(* This operator represents the loop termination condition used by the
"pivot" switches of a loop. *)
val loopCond
  :  ?name:string
  -> [ `bool ] t
  -> [ `bool ] t

(* Multiply the matrix "a" by the matrix "b". *)
(* The inputs must be two-dimensional matrices and the inner dimension of
"a" (after being transposed if transpose_a is true) must match the
outer dimension of "b" (after being transposed if transposed_b is
true).

*Note*: The default kernel implementation for MatMul on GPUs uses
cublas. *)
val matMul
  :  ?name:string
  -> ?transpose_a:bool
  -> ?transpose_b:bool
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) t

(* Returns the set of files matching a pattern. *)
(* Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion. *)
val matchingFiles
  :  ?name:string
  -> [ `string ] t
  -> [ `string ] t

(* Calculates the determinant of a square matrix. *)
val matrixDeterminant
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Calculates the inverse of a square invertible matrix or its adjoint (conjugate *)
(* transpose).

The op uses LU decomposition with partial pivoting to compute the inverse.

If the matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result. *)
val matrixInverse
  :  ?name:string
  -> ?adjoint:bool
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Solves a system of linear equations. Checks for invertibility. *)
val matrixSolve
  :  ?name:string
  -> ?adjoint:bool
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Solves a linear least-squares problem. *)
(* Below we will use the following notation
`matrix`=\\(A \in \Re^{m \times n}\\),
`rhs`=\\(B  \in \Re^{m \times k}\\),
`output`=\\(X  \in \Re^{n \times k}\\),
`l2_regularizer`=\\(\lambda\\).

If `fast` is `True`, then the solution is computed by solving the normal
equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
\\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
\lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
\\(X = A^T (A A^T + \lambda I)^{-1} B\\),
which (for \\(\lambda = 0\\)) is the minimum-norm solution to the
under-determined linear system, i.e.
\\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\),
subject to \\(A Z = B\\).
Notice that the fast path is only numerically stable when \\(A\\) is
numerically full rank and has a condition number
\\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\)
or \\(\lambda\\) is sufficiently large.

If `fast` is `False` an algorithm based on the numerically robust complete
orthogonal decomposition is used. This computes the minimum-norm
least-squares solution, even when \\(A\\) is rank deficient. This path is
typically 6-7 times slower than the fast path. If `fast` is `False` then
`l2_regularizer` is ignored. *)
val matrixSolveLs
  :  ?name:string
  -> ?fast:bool
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t
  -> [ `double ] t
  -> ([< `double | `float ] as 't) t

(* Solves a system of linear equations with an upper or lower triangular matrix by *)
(* backsubstitution.

`matrix` is a matrix of shape `[M, M]`. If `lower` is `True` then the strictly
upper triangular part of `matrix` is assumed to be zero and not accessed.
If `lower` is False then the strictly lower triangular part of `matrix` is
assumed to be zero and not accessed.
`rhs` is a matrix of shape [M, K]`.

The output is a matrix of shape `[M, K]`. If `adjoint` is `False` the output
satisfies the matrix equation `matrix` * `output` = `rhs`.
If `adjoint` is `False` then `output` satisfies the matrix equation
`matrix` * `output` = `rhs`.
If `adjoint` is `True` then `output` satisfies the matrix equation
`adjoint(matrix)` * `output` = `rhs`. *)
val matrixTriangularSolve
  :  ?name:string
  -> ?lower:bool
  -> ?adjoint:bool
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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Performs max pooling on the input. *)
val maxPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> [ `float ] t
  -> [ `float ] t

(* Performs 3D max pooling on the input. *)
val maxPool3D
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Computes gradients of max pooling function. *)
val maxPool3DGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
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
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t
  -> [ `float ] t

(* Computes gradients of the maxpooling function. *)
val maxPoolGradWithArgmax
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> [ `float ] t
  -> [ `float ] t
  -> ([< `int32 | `int64 ] as 'targmax) t
  -> [ `float ] t

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
  -> [ `float ] t
  -> [ `float ] t * ([< `int32 | `int64 ] as 'targmax) t

(* Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts. *)
val maximum
  :  ?name:string
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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Forwards the value of an available tensor from `inputs` to `output`. *)
(* `Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.

`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`.

It is an error if more than one tensor in `inputs` is available. *)
val merge
  :  ?name:string
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
  -> [ `string ] t list
  -> [ `string ] t

(* Computes the minimum of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val min
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts. *)
val minimum
  :  ?name:string
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
  -> 't t
  -> [ `int32 ] t
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
  -> 't t
  -> [ `int32 ] t
  -> 't t

(* Returns element-wise remainder of division. *)
val mod_
  :  ?name:string
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

(* Returns x * y element-wise. *)
val mul
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Draws samples from a multinomial distribution. *)
val multinomial
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> [ `int64 ] t

(* Computes numerical negative value element-wise. *)
(* I.e., \\(y = -x\\). *)
val neg
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Training via negative sampling. *)
val negTrain
  :  ?name:string
  -> vocab_count:int list
  -> num_negative_samples:int
  -> [ `float ] t
  -> [ `float ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> [ `unit ] t

(* Makes its input available to the next iteration. *)
val nextIteration
  :  ?name:string
  -> 't t
  -> 't t

(* Does nothing. Only useful as a placeholder for control edges. *)
val noOp
  :  ?name:string
  -> unit
  -> [ `unit ] t

(* Returns the truth value of (x != y) element-wise. *)
val notEqual
  :  ?name:string
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
  -> ([< `int32 | `int64 ] as 'tI) t
  -> [ `int32 ] t
  -> 't t
  -> 't t
  -> 't t

(* Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor. *)
(* Packs the `N` tensors in `values` into a tensor with rank one higher than each
tensor in `values` and shape `[N] + values[0].shape`. The output satisfies
`output[i, ...] = values[i][...]`.

This is the opposite of `unpack`. *)
val pack
  :  ?name:string
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
  -> 't t
  -> [ `int32 ] t
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
  -> unit
  -> [ `string ] t

(* A placeholder op for a value that will be fed into the computation. *)
(* N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime. *)
val placeholder
  :  ?name:string
  -> type_:'dtype Type.t
  -> ?shape:Dim.t list
  -> unit
  -> 'dtype t

(* A placeholder op that passes though `input` when its output is not fed. *)
val placeholderWithDefault
  :  ?name:string
  -> shape:Dim.t list
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
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes the product of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val prod
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Closes the given queue. *)
(* This operation signals that no more elements will be enqueued in the
given queue. Subsequent Enqueue(Many) operations will fail.
Subsequent Dequeue(Many) operations will continue to succeed if
sufficient elements remain in the queue. Subsequent Dequeue(Many)
operations that would block will fail immediately. *)
val queueClose
  :  ?name:string
  -> ?cancel_pending_enqueues:bool
  -> [ `string ] t
  -> [ `unit ] t

(* Computes the number of elements in the given queue. *)
val queueSize
  :  ?name:string
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
  -> [ `float ] t
  -> [ `float ] t

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
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int64 ] t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

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
  -> unit
  -> [ `string ] t

(* Outputs random values from a normal distribution. *)
(* The generated values will have mean 0 and standard deviation 1. *)
val randomStandardNormal
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
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
  -> ([< `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tout) t
  -> ([< `int32 | `int64 ] as 'tout) t
  -> ([< `int32 | `int64 ] as 'tout) t

(* Creates a sequence of integers. *)
(* This operation creates a sequence of integers that begins at `start` and
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
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `int32 ] t

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
of the tensor. Rank is also known as "order", "degree", or "ndims." *)
val rank
  :  ?name:string
  -> 't t
  -> [ `int32 ] t

(* Reads and outputs the entire contents of the input filename. *)
val readFile
  :  ?name:string
  -> [ `string ] t
  -> [ `string ] t

(* Returns the number of records this Reader has produced. *)
(* This is the same as the number of ReaderRead executions that have
succeeded. *)
val readerNumRecordsProduced
  :  ?name:string
  -> [ `string ] t
  -> [ `int64 ] t

(* Returns the number of work units this Reader has finished processing. *)
val readerNumWorkUnitsCompleted
  :  ?name:string
  -> [ `string ] t
  -> [ `int64 ] t

(* Returns the next record (key, value pair) produced by a Reader. *)
(* Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file). *)
val readerRead
  :  ?name:string
  -> [ `string ] t
  -> [ `string ] t
  -> [ `string ] t * [ `string ] t

(* Restore a Reader to its initial clean state. *)
val readerReset
  :  ?name:string
  -> [ `string ] t
  -> [ `unit ] t

(* Restore a reader to a previously saved state. *)
(* Not all Readers support being restored, so this can produce an
Unimplemented error. *)
val readerRestoreState
  :  ?name:string
  -> [ `string ] t
  -> [ `string ] t
  -> [ `unit ] t

(* Produce a string tensor that encodes the state of a Reader. *)
(* Not all Readers support being serialized, so this can produce an
Unimplemented error. *)
val readerSerializeState
  :  ?name:string
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
  -> ([< `complex64 ] as 't) t
  -> ([< `float | `double ] as 'tout) t

(* Joins a string Tensor across the given dimensions. *)
(* Computes the string join across dimensions in the given string Tensor of shape
`[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
strings with the given separator (default: empty string).  Negative indices are
counted backwards from the end, with `-1` being equivalent to `n - 1`.  Passing
an empty `reduction_indices` joins all strings in linear index order and outputs
a scalar string.


For example:
```
# tensor `a` is [["a", "b"], ["c", "d"]]
tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) ==> ["acbd"]
tf.reduce_join(a, [1, 0]) ==> ["abcd"]
tf.reduce_join(a, []) ==> ["abcd"]
``` *)
val reduceJoin
  :  ?name:string
  -> ?keep_dims:bool
  -> ?separator:string
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
  -> 't t
  -> 't t

(* Exits the current frame to its parent frame. *)
(* Exit makes its input `data` available to the parent frame. *)
val refExit
  :  ?name:string
  -> 't t
  -> 't t

(* Return the same ref tensor as the input ref tensor. *)
val refIdentity
  :  ?name:string
  -> 't t
  -> 't t

(* Forwards the value of an available tensor from `inputs` to `output`. *)
(* `Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.

`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`.

It is an error if more than one tensor in `inputs` is available. *)
val refMerge
  :  ?name:string
  -> 't t list
  -> 't t * [ `int32 ] t

(* Makes its input available to the next iteration. *)
val refNextIteration
  :  ?name:string
  -> 't t
  -> 't t

(* Forwards the `index`th element of `inputs` to `output`. *)
val refSelect
  :  ?name:string
  -> [ `int32 ] t
  -> 't t list
  -> 't t

(* Forwards the ref tensor `data` to the output port determined by `pred`. *)
(* If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.

See also `Switch` and `Merge`. *)
val refSwitch
  :  ?name:string
  -> 't t
  -> [ `bool ] t
  -> 't t * 't t

(* Computes rectified linear: `max(features, 0)`. *)
val relu
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes rectified linear 6: `min(max(features, 0), 6)`. *)
val relu6
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes rectified linear 6 gradients for a Relu6 operation. *)
val relu6Grad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes rectified linear gradients for a Relu operation. *)
val reluGrad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

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
  -> 't t
  -> [ `int32 ] t
  -> 't t

(* Resize `images` to `size` using area interpolation. *)
(* Input images can be of different types but output images are always float. *)
val resizeArea
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t

(* Resize `images` to `size` using bicubic interpolation. *)
(* Input images can be of different types but output images are always float. *)
val resizeBicubic
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t

(* Resize `images` to `size` using bilinear interpolation. *)
(* Input images can be of different types but output images are always float. *)
val resizeBilinear
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `float ] t

(* Computes the gradient of bilinear interpolation. *)
val resizeBilinearGrad
  :  ?name:string
  -> ?align_corners:bool
  -> [ `float ] t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Resize `images` to `size` using nearest neighbor interpolation. *)
val resizeNearestNeighbor
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) t
  -> [ `int32 ] t
  -> ([< `int32 | `int64 | `float | `double ] as 't) t

(* Computes the gradient of nearest neighbor interpolation. *)
val resizeNearestNeighborGrad
  :  ?name:string
  -> ?align_corners:bool
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
  -> ([< `int32 | `bool | `float | `double ] as 't) t
  -> [ `bool ] t
  -> ([< `int32 | `bool | `float | `double ] as 't) t

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
  -> 't t
  -> [ `int64 ] t
  -> 't t

(* Computes reciprocal of square root of x element-wise. *)
(* I.e., \\(y = 1 / \sqrt{x}\\). *)
val rsqrt
  :  ?name:string
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
image. The latter may be supplied to `tf.image.draw_bounding_box` to visualize
what the bounding box looks like.

Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example,

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
  -> ([< `int32 | `int64 ] as 't) t
  -> [ `float ] t
  -> ([< `int32 | `int64 ] as 't) t * ([< `int32 | `int64 ] as 't) t * [ `float ] t

(* Outputs a `Summary` protocol buffer with scalar values. *)
(* The input `tags` and `values` must have the same shape.  The generated summary
has a summary value for each tag-value pair in `tags` and `values`. *)
val scalarSummary
  :  ?name:string
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

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterAdd.png" alt>
</div> *)
val scatterAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

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

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterSub.png" alt>
</div> *)
val scatterSub
  :  ?name:string
  -> ?use_locking:bool
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
duplicate entires in `indices`, the order at which the updates happen
for each value is undefined.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterUpdate.png" alt>
</div> *)
val scatterUpdate
  :  ?name:string
  -> ?use_locking:bool
  -> 't t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t

(* Computes the maximum along segments of a tensor. *)
(* Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \max_j(data_j)\\) where `max` is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentMax.png" alt>
</div> *)
val segmentMax
  :  ?name:string
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

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentMean.png" alt>
</div> *)
val segmentMean
  :  ?name:string
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

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentMin.png" alt>
</div> *)
val segmentMin
  :  ?name:string
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

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentProd.png" alt>
</div> *)
val segmentProd
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes the sum along segments of a tensor. *)
(* Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentSum.png" alt>
</div> *)
val segmentSum
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Selects elements from `t` or `e`, depending on `condition`. *)
(* The `t`, and `e` tensors must all have the same shape,
and the output will also have that shape.  The `condition` tensor
must be a scalar if `t` and `e` are scalars.  If `t` and `e` are vectors
or higher rank, then `condition` must be either a vector with size
matching the first dimension of `t`, or must have the same shape as `t`.

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
  -> [ `bool ] t
  -> 't t
  -> 't t
  -> 't t

(* Calculates the Eigen Decomposition of a square Self-Adjoint matrix. *)
(* Only the lower-triangular part of the input will be used in this case. The
upper-triangular part will not be read.

The result is a M+1 x M matrix whose first row is the eigenvalues, and
subsequent rows are eigenvectors. *)
val selfAdjointEig
  :  ?name:string
  -> ([< `double | `float ] as 't) t
  -> ([< `double | `float ] as 't) t

(* Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`. *)
(* The `SparseTensor` must have rank `R` greater than 1, and the first dimension
is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The serialized
`SparseTensor` objects going into each row of `serialized_sparse` will have
rank `R-1`.

The minibatch size `N` is extracted from `sparse_shape[0]`. *)
val serializeManySparse
  :  ?name:string
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `string ] t

(* Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object. *)
val serializeSparse
  :  ?name:string
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `string ] t

(* Returns the shape of a tensor. *)
(* This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```prettyprint
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
``` *)
val shape
  :  ?name:string
  -> 't t
  -> [ `int32 ] t

(* Returns shape of tensors. *)
(* This operation returns N 1-D integer tensors representing shape of `input[i]s`. *)
val shapeN
  :  ?name:string
  -> 't t list
  -> [ `int32 ] t list

(* Generate a sharded filename. The filename is printf formatted as *)
(*    %s-%05d-of-%05d, basename, shard, num_shards. *)
val shardedFilename
  :  ?name:string
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> [ `string ] t

(* Generate a glob pattern matching all sharded file names. *)
val shardedFilespec
  :  ?name:string
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `string ] t

(* Computes sigmoid of `x` element-wise. *)
(* Specifically, `y = 1 / (1 + exp(-x))`. *)
val sigmoid
  :  ?name:string
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Returns an element-wise indication of the sign of a number. *)
(* `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`. *)
val sign
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes sin of x element-wise. *)
val sin
  :  ?name:string
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
  -> 't t
  -> [ `int32 ] t

(* Parses a text file and creates a batch of examples. *)
val skipgram
  :  ?name:string
  -> filename:string
  -> batch_size:int
  -> ?window_size:int
  -> ?min_count:int
  -> ?subsample:float
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
  -> 't t
  -> ([< `int32 | `int64 ] as 'index) t
  -> ([< `int32 | `int64 ] as 'index) t
  -> 't t

(* Computes softmax activations. *)
(* For each batch `i` and class `j` we have

    softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i])) *)
val softmax
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

(* Computes softmax cross entropy cost and gradients to backpropagate. *)
(* Inputs are the logits, not probabilities. *)
val softmaxCrossEntropyWithLogits
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t

(* Computes softplus: `log(exp(features) + 1)`. *)
val softplus
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes softplus gradients for a softplus operation. *)
val softplusGrad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes softsign: `features / (abs(features) + 1)`. *)
val softsign
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Computes softsign gradients for a softsign operation. *)
val softsignGrad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* SpaceToBatch for 4-D tensors of type T. *)
(* Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size. *)
val spaceToBatch
  :  ?name:string
  -> block_size:int
  -> 't t
  -> [ `int32 ] t
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
  -> 't t
  -> 't t

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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t * ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* var: Should be from a Variable(). *)
val sparseApplyAdadelta
  :  ?name:string
  -> ?use_locking:bool
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
(* That is for rows we have grad for, we update var and accum as follows:

accum = accum * momentum + grad
var -= lr * accum *)
val sparseApplyMomentum
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
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
    [0, 2]: "a"
    [1, 0]: "b"
    [1, 1]: "c"

    sp_inputs[1]: shape = [2, 4]
    [0, 1]: "d"
    [0, 2]: "e"

then the output will be

    shape = [2, 7]
    [0, 2]: "a"
    [0, 4]: "d"
    [0, 5]: "e"
    [1, 0]: "b"
    [1, 1]: "c"

Graphically this is equivalent to doing

    [    a] concat [  d e  ] = [    a   d e  ]
    [b c  ]        [       ]   [b c          ] *)
val sparseConcat
  :  ?name:string
  -> concat_dim:int
  -> [ `int64 ] t list
  -> 't t list
  -> [ `int64 ] t list
  -> [ `int64 ] t * 't t * [ `int64 ] t

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
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Multiply matrix "a" by matrix "b". *)
(* The inputs must be two-dimensional matrices and the inner dimension of "a" must
match the outer dimension of "b". This op is optimized for the case where at
least one of "a" or "b" is sparse. The breakeven for using this versus a dense
matrix multiply on one platform was 30% zero values in the sparse matrix. *)
val sparseMatMul
  :  ?name:string
  -> ?transpose_a:bool
  -> ?transpose_b:bool
  -> ?a_is_sparse:bool
  -> ?b_is_sparse:bool
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
  -> [ `int64 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int64 ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Reorders a SparseTensor into the canonical, row-major ordering. *)
(* Note that by convention, all sparse ops preserve the canonical ordering along
increasing dimension number. The only time ordering can be violated is during
manual manipulation of the indices and values vectors to add entries.

Reordering does not affect the shape of the SparseTensor.

If the tensor has rank `R` and `N` non-empty values, `input_indices` has
shape `[N, R]`, input_values has length `N`, and input_shape has length `R`. *)
val sparseReorder
  :  ?name:string
  -> [ `int64 ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `int64 ] t * 't t

(* Computes the mean along sparse segments of a tensor. *)
(* Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`. *)
val sparseSegmentMean
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

(* Computes gradients for SparseSegmentMean. *)
(* Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0. *)
val sparseSegmentMeanGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
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
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double ] as 't) t

(* Computes gradients for SparseSegmentSqrtN. *)
(* Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0. *)
val sparseSegmentSqrtNGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> [ `int32 ] t
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
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Applies softmax to a batched N-D `SparseTensor`. *)
(* The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
(where `N >= 2`), and with indices sorted in the canonical lexicographic order.

This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
zero elements do not participate*.  Specifically, the algorithm is equivalent *)
val sparseSoftmax
  :  ?name:string
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
  -> ([< `float | `double ] as 't) t
  -> ([< `int32 | `int64 ] as 'tlabels) t
  -> ([< `float | `double ] as 't) t * ([< `float | `double ] as 't) t

(* Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`. *)
(* This Op does not require `a_indices` be sorted in standard lexicographic order. *)
val sparseTensorDenseAdd
  :  ?name:string
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Multiply SparseTensor (of rank 2) "A" by dense matrix "B". *)
(* No validity checking is performed on the indices of A.  However, the following
input format is recommended for optimal behavior:

if adjoint_a == false:
  A should be sorted in lexicographically increasing order.  Use SparseReorder
  if you're not sure.
if adjoint_a == true:
  A should be sorted in order of increasing dimension 1 (i.e., "column major"
  order instead of "row major" order). *)
val sparseTensorDenseMatMul
  :  ?name:string
  -> ?adjoint_a:bool
  -> ?adjoint_b:bool
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
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> 't t
  -> 't t
  -> 't t

(* Splits a tensor into `num_split` tensors along one dimension. *)
val split
  :  ?name:string
  -> num_split:int
  -> [ `int32 ] t
  -> 't t
  -> 't t list

(* Computes square root of x element-wise. *)
(* I.e., \\(y = \sqrt{x} = x^{1/2}\\). *)
val sqrt
  :  ?name:string
  -> ([< `float | `double | `complex64 ] as 't) t
  -> ([< `float | `double | `complex64 ] as 't) t

(* Computes square of x element-wise. *)
(* I.e., \\(y = x * x = x^2\\). *)
val square
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Returns (x - y)(x - y) element-wise. *)
val squaredDifference
  :  ?name:string
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
  -> 't t
  -> 't t

(* A stack that produces elements in first-in last-out order. *)
val stack
  :  ?name:string
  -> ?stack_name:string
  -> unit
  -> [ `string ] t

(* Delete the stack from its resource container. *)
val stackClose
  :  ?name:string
  -> [ `string ] t
  -> [ `unit ] t

(* Pop the element at the top of the stack. *)
val stackPop
  :  ?name:string
  -> type_:'elem_type Type.t
  -> [ `string ] t
  -> 'elem_type t

(* Push an element onto the stack. *)
val stackPush
  :  ?name:string
  -> ?swap_memory:bool
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
  -> 't t
  -> 't t

(* Converts each string in the input Tensor to its hash mod by a number of buckets. *)
(* The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time. *)
val stringToHashBucket
  :  ?name:string
  -> num_buckets:int
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
time than tf.string_to_hash_bucket_fast. *)
val stringToHashBucketStrong
  :  ?name:string
  -> num_buckets:int
  -> key:int list
  -> [ `string ] t
  -> [ `int64 ] t

(* Converts each string in the input Tensor to the specified numeric type. *)
(* (Note that int32 overflow results in an error while float overflow
results in a rounded value.) *)
val stringToNumber
  :  ?name:string
  -> type_:([< `float | `int32 ] as 'out_type) Type.t
  -> [ `string ] t
  -> ([< `float | `int32 ] as 'out_type) t

(* Returns x - y element-wise. *)
val sub
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) t

(* Computes the sum of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val sum
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) t

(* Forwards `data` to the output port determined by `pred`. *)
(* If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.

See also `RefSwitch` and `Merge`. *)
val switch
  :  ?name:string
  -> 't t
  -> [ `bool ] t
  -> 't t * 't t

(* A Reader that outputs the records from a TensorFlow Records file. *)
val tFRecordReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] t

(* Computes hyperbolic tangent of `x` element-wise. *)
val tanh
  :  ?name:string
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
  -> unit
  -> 'dtype t

(* An array of Tensors of given size, with data written via Write and read *)
(* via Read or Pack. *)
val tensorArray
  :  ?name:string
  -> ?dynamic_size:bool
  -> ?clear_after_read:bool
  -> ?tensor_array_name:string
  -> [ `int32 ] t
  -> [ `string ] t

(* Delete the TensorArray from its resource container.  This enables *)
(* the user to close and release the resource in the middle of a step/run. *)
val tensorArrayClose
  :  ?name:string
  -> [ `string ] t
  -> [ `unit ] t

(* Concat the elements from the TensorArray into value `value`. *)
(* Takes `T` elements of shapes

  ```
  (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
  ```

and concatenates them into a Tensor of shape:

  ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```

All elements must have the same shape (excepting the first dimension). *)
val tensorArrayConcat
  :  ?name:string
  -> type_:'dtype Type.t
  -> [ `string ] t
  -> [ `float ] t
  -> 'dtype t * [ `int64 ] t

(* Creates a TensorArray for storing the gradients of values in the given handle. *)
(* If the given TensorArray gradient already exists, returns a reference to it.

Locks the size of the original TensorArray by disabling its dynamic size flag.

**A note about the input flow_in:**

The handle flow_in forces the execution of the gradient lookup to occur
only after certain other operations have occurred.  For example, when
the forward TensorArray is dynamically sized, writes to this TensorArray
may resize the object.  The gradient TensorArray is statically sized based
on the size of the forward TensorArray when this operation executes.
Furthermore, the size of the forward TensorArray is frozen by this call.
As a result, the flow is used to ensure that the call to generate the gradient
TensorArray only happens after all writes are executed.

In terms of e.g. python TensorArray sugar wrappers when using dynamically sized *)
val tensorArrayGrad
  :  ?name:string
  -> source:string
  -> [ `string ] t
  -> [ `float ] t
  -> [ `string ] t

(* Pack the elements from the TensorArray into output `value`. *)
(* All elements must have the same shape. *)
val tensorArrayPack
  :  ?name:string
  -> type_:'dtype Type.t
  -> [ `string ] t
  -> [ `float ] t
  -> 'dtype t

(* Read an element from the TensorArray into output `value`. *)
val tensorArrayRead
  :  ?name:string
  -> type_:'dtype Type.t
  -> [ `string ] t
  -> [ `int32 ] t
  -> [ `float ] t
  -> 'dtype t

(* Get the current size of the TensorArray. *)
val tensorArraySize
  :  ?name:string
  -> [ `string ] t
  -> [ `float ] t
  -> [ `int32 ] t

(* Split the data from the input value into TensorArray elements. *)
(* Assuming that `lengths` takes on values

  ```(n0, n1, ..., n(T-1))```

and that `value` has shape

  ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```,

this splits values into a TensorArray with T tensors.

TensorArray index t will be the subtensor of values with starting position

  ```(n0 + n1 + ... + n(t-1), 0, 0, ...)```

and having size

  ```nt x d0 x d1 x ...``` *)
val tensorArraySplit
  :  ?name:string
  -> [ `string ] t
  -> 't t
  -> [ `int64 ] t
  -> [ `float ] t
  -> [ `float ] t

(* Unpack the data from the input value into TensorArray elements. *)
val tensorArrayUnpack
  :  ?name:string
  -> [ `string ] t
  -> 't t
  -> [ `float ] t
  -> [ `float ] t

(* Push an element onto the tensor_array. *)
val tensorArrayWrite
  :  ?name:string
  -> [ `string ] t
  -> [ `int32 ] t
  -> 't t
  -> [ `float ] t
  -> [ `float ] t

(* A Reader that outputs the lines of a file delimited by '\n'. *)
val textLineReader
  :  ?name:string
  -> ?skip_header_lines:int
  -> ?container:string
  -> ?shared_name:string
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
  -> 't t
  -> [ `int32 ] t
  -> 't t

(* Returns the gradient of `Tile`. *)
(* Since `Tile` takes an input and repeats the input `multiples` times
along each dimension, `TileGrad` takes in `multiples` and aggregates
each repeated tile of `input` into `output`. *)
val tileGrad
  :  ?name:string
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
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t * [ `int32 ] t

(* Shuffle dimensions of x according to a permutation. *)
(* The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]` *)
val transpose
  :  ?name:string
  -> 't t
  -> [ `int32 ] t
  -> 't t

(* Outputs random values from a truncated normal distribution. *)
(* The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked. *)
val truncatedNormal
  :  ?name:string
  -> type_:([< `float | `double ] as 'dtype) Type.t
  -> ?seed:int
  -> ?seed2:int
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
  -> 't t
  -> 't t * [ `int32 ] t

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
  -> 't t
  -> 't t * [ `int32 ] t * [ `int32 ] t

(* Unpacks the outer dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors. *)
(* Unpacks `num` tensors from `value` by chipping it along the first dimension.
The i'th tensor in `output` is the slice `value[i, ...]`. Each tensor in
`output` has shape `value.shape[1:]`.

This is the opposite of `pack`. *)
val unpack
  :  ?name:string
  -> num:int
  -> 't t
  -> 't t list

(* Computes the sum along segments of a tensor. *)
(* Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`. Unlike `SegmentSum`, `segment_ids`
need not be sorted and need not cover all values in the full
  range of valid values.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.

`num_segments` should equal the number of distinct segment IDs.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/UnsortedSegmentSum.png" alt>
</div> *)
val unsortedSegmentSum
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) t
  -> ([< `int32 | `int64 ] as 'tindices) t
  -> [ `int32 ] t
  -> ([< `float | `double | `int32 | `int64 ] as 't) t

(* Holds state in the form of a tensor that persists across steps. *)
(* Outputs a ref to the tensor state so it may be read or modified.
TODO(zhifengc/mrry): Adds a pointer to a more detail document
about sharing states in tensorflow. *)
val variable
  :  ?name:string
  -> type_:'dtype Type.t
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
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
  -> [ `bool ] t
  -> [ `int64 ] t

(* A Reader that outputs the entire contents of a file as a value. *)
(* To use, enqueue filenames in a Queue.  The output of ReaderRead will
be a filename (key) and the contents of that file (value). *)
val wholeFileReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] t

(* Returns a tensor of zeros with the same shape and type as x. *)
val zerosLike
  :  ?name:string
  -> 't t
  -> 't t

(* Compute the Hurwitz zeta function \\(\zeta(x, q)\\). *)
(* The Hurwitz zeta function is defined as:

```
\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}
``` *)
val zeta
  :  ?name:string
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t
  -> ([< `float | `double ] as 't) t

