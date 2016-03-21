(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)
open Node

val abs
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val add
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t

val addN
  :  ?name:string
  -> n:int
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val adjustContrast
  :  ?name:string
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val adjustContrastv2
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val all
  :  ?name:string
  -> ?keep_dims:bool
  -> [ `bool ] Node.t
  -> [ `int32 ] Node.t
  -> [ `bool ] Node.t

val any
  :  ?name:string
  -> ?keep_dims:bool
  -> [ `bool ] Node.t
  -> [ `int32 ] Node.t
  -> [ `bool ] Node.t

val applyAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val applyAdam
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val applyFtrl
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val applyGradientDescent
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val applyMomentum
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val applyRMSProp
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val argMax
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int64 ] Node.t

val argMin
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int64 ] Node.t

val assign
  :  ?name:string
  -> ?validate_shape:bool
  -> ?use_locking:bool
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

val assignAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val assignSub
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val avgPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val avgPoolGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchCholesky
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

val batchMatMul
  :  ?name:string
  -> ?adj_x:bool
  -> ?adj_y:bool
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t

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
  -> ?fast:bool
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> [ `double ] Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchMatrixTriangularSolve
  :  ?name:string
  -> ?lower:bool
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val batchNormWithGlobalNormalization
  :  ?name:string
  -> variance_epsilon:float
  -> scale_after_normalization:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val batchSelfAdjointEig
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

val biasAdd
  :  ?name:string
  -> ?data_format:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val biasAddGrad
  :  ?name:string
  -> ?data_format:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val biasAddV1
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val bitcast
  :  ?name:string
  -> type_ : ([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) Node.Type.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) Node.t

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
  -> message:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val cholesky
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

val complex
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `complex64 ] Node.t

val complexAbs
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `float ] Node.t

val concat
  :  ?name:string
  -> n:int
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

val concatOffset
  :  ?name:string
  -> n:int
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t

val conj
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `complex64 ] Node.t

val controlTrigger
  :  ?name:string
  -> unit
  -> [ `unit ] Node.t

val conv2D
  :  ?name:string
  -> strides:int list
  -> ?use_cudnn_on_gpu:bool
  -> padding:string
  -> ?data_format:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val conv2DBackpropFilter
  :  ?name:string
  -> strides:int list
  -> ?use_cudnn_on_gpu:bool
  -> padding:string
  -> ?data_format:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val conv2DBackpropInput
  :  ?name:string
  -> strides:int list
  -> ?use_cudnn_on_gpu:bool
  -> padding:string
  -> ?data_format:string
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val cos
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val countUpTo
  :  ?name:string
  -> limit:int
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 't) Node.t

val cross
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val decodeJSONExample
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t

val decodePng
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> ?channels:int
  -> [ `string ] Node.t
  -> 'dtype Node.t

val decodeRaw
  :  ?name:string
  -> type_ : ([< `float | `double | `int32 | `int64 ] as 'out_type) Node.Type.t
  -> ?little_endian:bool
  -> [ `string ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 'out_type) Node.t

val depthToSpace
  :  ?name:string
  -> block_size:int
  -> 't Node.t
  -> 't Node.t

val depthwiseConv2dNative
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val depthwiseConv2dNativeBackpropFilter
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val depthwiseConv2dNativeBackpropInput
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val destroyTemporaryVariable
  :  ?name:string
  -> var_name:string
  -> 't Node.t
  -> 't Node.t

val diag
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val diagPart
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val digamma
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val div
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t

val drawBoundingBoxes
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val dynamicPartition
  :  ?name:string
  -> num_partitions:int
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

val dynamicStitch
  :  ?name:string
  -> n:int
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

val editDistance
  :  ?name:string
  -> ?normalize:bool
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
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

val encodePng
  :  ?name:string
  -> ?compression:int
  -> 't Node.t
  -> [ `string ] Node.t

val enter
  :  ?name:string
  -> frame_name:string
  -> ?is_constant:bool
  -> ?parallel_iterations:int
  -> 't Node.t
  -> 't Node.t

val equal
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> [ `bool ] Node.t

val erf
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val erfc
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val exit
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val exp
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val expandDims
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

val extractGlimpse
  :  ?name:string
  -> ?centered:bool
  -> ?normalized:bool
  -> ?uniform_noise:bool
  -> [ `float ] Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val fFT2D
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `complex64 ] Node.t

val fIFOQueue
  :  ?name:string
  -> component_types:Type.p list
  -> ?shapes:Dim.t list list
  -> ?capacity:int
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

val fact
  :  ?name:string
  -> unit
  -> [ `string ] Node.t

val fill
  :  ?name:string
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

val fixedLengthRecordReader
  :  ?name:string
  -> ?header_bytes:int
  -> record_bytes:int
  -> ?footer_bytes:int
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

val floor
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val gather
  :  ?name:string
  -> ?validate_indices:bool
  -> 'tparams Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> 'tparams Node.t

val greater
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

val greaterEqual
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

val hSVToRGB
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val hashTable
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

val histogramSummary
  :  ?name:string
  -> [ `string ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `string ] Node.t

val iFFT2D
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `complex64 ] Node.t

val identity
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val identityReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

val imag
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `float ] Node.t

val imageSummary
  :  ?name:string
  -> ?max_images:int
  -> [ `string ] Node.t
  -> ([< `float ] as 't) Node.t
  -> [ `string ] Node.t

val inTopK
  :  ?name:string
  -> k:int
  -> [ `float ] Node.t
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

val initializeTable
  :  ?name:string
  -> [ `string ] Node.t
  -> 'tkey Node.t
  -> 'tval Node.t
  -> [ `unit ] Node.t

val inv
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val invertPermutation
  :  ?name:string
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t

val isFinite
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `bool ] Node.t

val isInf
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `bool ] Node.t

val isNan
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `bool ] Node.t

val l2Loss
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val lRN
  :  ?name:string
  -> ?depth_radius:int
  -> ?bias:float
  -> ?alpha:float
  -> ?beta:float
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val lRNGrad
  :  ?name:string
  -> ?depth_radius:int
  -> ?bias:float
  -> ?alpha:float
  -> ?beta:float
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val less
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

val lessEqual
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

val lgamma
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val linSpace
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

val log
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val logicalAnd
  :  ?name:string
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t

val logicalNot
  :  ?name:string
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t

val logicalOr
  :  ?name:string
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t

val lookupTableFind
  :  ?name:string
  -> [ `string ] Node.t
  -> 'tin Node.t
  -> 'tout Node.t
  -> 'tout Node.t

val lookupTableSize
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int64 ] Node.t

val loopCond
  :  ?name:string
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t

val matMul
  :  ?name:string
  -> ?transpose_a:bool
  -> ?transpose_b:bool
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t

val matchingFiles
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t

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
  -> ?fast:bool
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> [ `double ] Node.t
  -> ([< `float | `double ] as 't) Node.t

val matrixTriangularSolve
  :  ?name:string
  -> ?lower:bool
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val max
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val maxPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val maxPoolGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val maxPoolGradWithArgmax
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> ([< `int32 | `int64 ] as 'targmax) Node.t
  -> [ `float ] Node.t

val maximum
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val mean
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val mergeSummary
  :  ?name:string
  -> n:int
  -> [ `string ] Node.t
  -> [ `string ] Node.t

val min
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val minimum
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val mirrorPad
  :  ?name:string
  -> mode:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

val mirrorPadGrad
  :  ?name:string
  -> mode:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

val mod_
  :  ?name:string
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t

val mul
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t

val neg
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val negTrain
  :  ?name:string
  -> vocab_count:int list
  -> num_negative_samples:int
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t
  -> [ `unit ] Node.t

val nextIteration
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val noOp
  :  ?name:string
  -> unit
  -> [ `unit ] Node.t

val notEqual
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> [ `bool ] Node.t

val oneHot
  :  ?name:string
  -> ?axis:int
  -> [ `int64 ] Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

val pack
  :  ?name:string
  -> n:int
  -> 't Node.t
  -> 't Node.t

val pad
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

val paddingFIFOQueue
  :  ?name:string
  -> component_types:Type.p list
  -> ?shapes:Dim.t list list
  -> ?capacity:int
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

val placeholder
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> ?shape:Dim.t list
  -> unit
  -> 'dtype Node.t

val pow
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val prod
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val queueClose
  :  ?name:string
  -> ?cancel_pending_enqueues:bool
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

val queueSize
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t

val rGBToHSV
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val randomCrop
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int64 ] Node.t
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t

val randomShuffle
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> 't Node.t
  -> 't Node.t

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
  -> [ `string ] Node.t

val randomStandardNormal
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'dtype) Node.Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double ] as 'dtype) Node.t

val randomUniform
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'dtype) Node.Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double ] as 'dtype) Node.t

val randomUniformInt
  :  ?name:string
  -> ?seed:int
  -> ?seed2:int
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tout) Node.t
  -> ([< `int32 | `int64 ] as 'tout) Node.t
  -> ([< `int32 | `int64 ] as 'tout) Node.t

val range
  :  ?name:string
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t

val rank
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t

val readFile
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t

val readerNumRecordsProduced
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int64 ] Node.t

val readerNumWorkUnitsCompleted
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int64 ] Node.t

val readerReset
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

val readerRestoreState
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

val readerSerializeState
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t

val real
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `float ] Node.t

val refEnter
  :  ?name:string
  -> frame_name:string
  -> ?is_constant:bool
  -> ?parallel_iterations:int
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

val refSelect
  :  ?name:string
  -> n:int
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

val relu
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val relu6
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val relu6Grad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val reluGrad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val reshape
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

val resizeArea
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t

val resizeBicubic
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t

val resizeBilinear
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t

val resizeBilinearGrad
  :  ?name:string
  -> ?align_corners:bool
  -> [ `float ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val resizeNearestNeighbor
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t

val resizeNearestNeighborGrad
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `int32 | `float | `double ] as 't) Node.t

val restore
  :  ?name:string
  -> type_ : 'dt Node.Type.t
  -> ?preferred_shard:int
  -> [ `string ] Node.t
  -> [ `string ] Node.t
  -> 'dt Node.t

val restoreSlice
  :  ?name:string
  -> type_ : 'dt Node.Type.t
  -> ?preferred_shard:int
  -> [ `string ] Node.t
  -> [ `string ] Node.t
  -> [ `string ] Node.t
  -> 'dt Node.t

val reverse
  :  ?name:string
  -> ([< `int32 | `bool | `float | `double ] as 't) Node.t
  -> [ `bool ] Node.t
  -> ([< `int32 | `bool | `float | `double ] as 't) Node.t

val reverseSequence
  :  ?name:string
  -> seq_dim:int
  -> ?batch_dim:int
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> 't Node.t

val rsqrt
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val scalarSummary
  :  ?name:string
  -> [ `string ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `string ] Node.t

val scatterAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val scatterSub
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val scatterUpdate
  :  ?name:string
  -> ?use_locking:bool
  -> 't Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> 't Node.t
  -> 't Node.t

val segmentMax
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val segmentMean
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val segmentMin
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val segmentProd
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val segmentSum
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val select
  :  ?name:string
  -> [ `bool ] Node.t
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

val selfAdjointEig
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

val serializeManySparse
  :  ?name:string
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> [ `string ] Node.t

val serializeSparse
  :  ?name:string
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> [ `string ] Node.t

val shape
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t

val shapeN
  :  ?name:string
  -> n:int
  -> 't Node.t
  -> [ `int32 ] Node.t

val shardedFilename
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `string ] Node.t

val shardedFilespec
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t
  -> [ `string ] Node.t

val sigmoid
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val sign
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val sin
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val size
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t

val slice
  :  ?name:string
  -> 't Node.t
  -> ([< `int32 | `int64 ] as 'index) Node.t
  -> ([< `int32 | `int64 ] as 'index) Node.t
  -> 't Node.t

val softmax
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

val softplus
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val softplusGrad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val softsign
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val softsignGrad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val spaceToDepth
  :  ?name:string
  -> block_size:int
  -> 't Node.t
  -> 't Node.t

val sparseApplyAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val sparseApplyFtrl
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val sparseApplyMomentum
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val sparseMatMul
  :  ?name:string
  -> ?transpose_a:bool
  -> ?transpose_b:bool
  -> ?a_is_sparse:bool
  -> ?b_is_sparse:bool
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val sparseSegmentMean
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

val sparseSegmentMeanGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

val sparseSegmentSqrtN
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

val sparseSegmentSqrtNGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

val sparseSegmentSum
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val sparseTensorDenseMatMul
  :  ?name:string
  -> ?adjoint_a:bool
  -> ?adjoint_b:bool
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> 't Node.t

val sparseToDense
  :  ?name:string
  -> ?validate_indices:bool
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

val split
  :  ?name:string
  -> num_split:int
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

val sqrt
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val square
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val squaredDifference
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val squeeze
  :  ?name:string
  -> ?squeeze_dims:int list
  -> 't Node.t
  -> 't Node.t

val stack
  :  ?name:string
  -> ?stack_name:string
  -> unit
  -> [ `string ] Node.t

val stackClose
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

val stackPop
  :  ?name:string
  -> type_ : 'elem_type Node.Type.t
  -> [ `string ] Node.t
  -> 'elem_type Node.t

val stackPush
  :  ?name:string
  -> ?swap_memory:bool
  -> [ `string ] Node.t
  -> 't Node.t
  -> 't Node.t

val stopGradient
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

val stringToHashBucket
  :  ?name:string
  -> num_buckets:int
  -> [ `string ] Node.t
  -> [ `int64 ] Node.t

val stringToNumber
  :  ?name:string
  -> type_ : ([< `float | `int32 ] as 'out_type) Node.Type.t
  -> [ `string ] Node.t
  -> ([< `float | `int32 ] as 'out_type) Node.t

val sub
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val sum
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

val tFRecordReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

val tanh
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

val temporaryVariable
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> shape:Dim.t list
  -> ?var_name:string
  -> unit
  -> 'dtype Node.t

val tensorArray
  :  ?name:string
  -> ?dynamic_size:bool
  -> ?tensor_array_name:string
  -> [ `int32 ] Node.t
  -> [ `string ] Node.t

val tensorArrayClose
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

val tensorArrayGrad
  :  ?name:string
  -> source:string
  -> [ `string ] Node.t
  -> [ `float ] Node.t
  -> [ `string ] Node.t

val tensorArrayPack
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> [ `string ] Node.t
  -> [ `float ] Node.t
  -> 'dtype Node.t

val tensorArrayRead
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t
  -> 'dtype Node.t

val tensorArraySize
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `float ] Node.t
  -> [ `int32 ] Node.t

val tensorArraySplit
  :  ?name:string
  -> [ `string ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val tensorArrayUnpack
  :  ?name:string
  -> [ `string ] Node.t
  -> 't Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val tensorArrayWrite
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

val textLineReader
  :  ?name:string
  -> ?skip_header_lines:int
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

val tile
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

val tileGrad
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

val transpose
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

val truncatedNormal
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'dtype) Node.Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double ] as 'dtype) Node.t

val unpack
  :  ?name:string
  -> num:int
  -> 't Node.t
  -> 't Node.t

val unsortedSegmentSum
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

val variable
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> 'dtype Node.t

val where
  :  ?name:string
  -> [ `bool ] Node.t
  -> [ `int64 ] Node.t

val wholeFileReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

val zerosLike
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

