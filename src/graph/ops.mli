(* THIS FILE HAS BEEN AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND! *)
open Node

(* Computes the absolute value of a tensor. *)
(* Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\). *)
val abs
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Returns x + y element-wise. *)
(* *NOTE*: Add supports broadcasting. AddN does not. *)
val add
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t

(* Add all input tensors element wise. *)
val addN
  :  ?name:string
  -> n:int
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Deprecated. Disallowed in GraphDef version >= 2. *)
val adjustContrast
  :  ?name:string
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

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
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

(* Computes the "logical and" of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val all
  :  ?name:string
  -> ?keep_dims:bool
  -> [ `bool ] Node.t
  -> [ `int32 ] Node.t
  -> [ `bool ] Node.t

(* Computes the "logical or" of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val any
  :  ?name:string
  -> ?keep_dims:bool
  -> [ `bool ] Node.t
  -> [ `int32 ] Node.t
  -> [ `bool ] Node.t

(* Update '*var' according to the adagrad scheme. *)
(* accum += grad * grad
var -= lr * grad * (1 / sqrt(accum)) *)
val applyAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Update '*var' according to the Adam algorithm. *)
(* lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon) *)
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

(* Update '*var' according to the Ftrl-proximal scheme. *)
(* accum_new = accum + grad * grad
linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
accum = accum_new *)
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

(* Update '*var' by subtracting 'alpha' * 'delta' from it. *)
val applyGradientDescent
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Update '*var' according to the momentum scheme. *)
(* accum = accum * momentum + grad
var -= lr * accum *)
val applyMomentum
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Update '*var' according to the RMSProp algorithm. *)
(* mean_square = decay * mean_square + (1-decay) * gradient ** 2
Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

ms <- rho * ms_{t-1} + (1-rho) * grad * grad
mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
var <- var - mom *)
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

(* Returns the index with the largest value across dimensions of a tensor. *)
val argMax
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int64 ] Node.t

(* Returns the index with the smallest value across dimensions of a tensor. *)
val argMin
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int64 ] Node.t

(* Update 'ref' by assigning 'value' to it. *)
(* This operation outputs "ref" after the assignment is done.
This makes it easier to chain operations that need to use the reset value. *)
val assign
  :  ?name:string
  -> ?validate_shape:bool
  -> ?use_locking:bool
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

(* Update 'ref' by adding 'value' to it. *)
(* This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value. *)
val assignAdd
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Update 'ref' by subtracting 'value' from it. *)
(* This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value. *)
val assignSub
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Performs average pooling on the input. *)
(* Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. *)
val avgPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes gradients of the average pooling function. *)
val avgPoolGrad
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Calculates the Cholesky decomposition of a batch of square matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix Cholesky
decomposition above. The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`. *)
val batchCholesky
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

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

    out[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :]) *)
val batchMatMul
  :  ?name:string
  -> ?adj_x:bool
  -> ?adj_y:bool
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t

(* Calculates the determinants for a batch of square matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a 1-D tensor containing the determinants
for all input submatrices `[..., :, :]`. *)
val batchMatrixDeterminant
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Calculates the inverse of square invertible matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.

The op uses the Cholesky decomposition if the matrices are symmetric positive
definite and LU decomposition with partial pivoting otherwise.

If a matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result. *)
val batchMatrixInverse
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Solves systems of linear equations. Checks for invertibility. *)
(* Matrix is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. Rhs is a tensor of shape
`[..., M, K]`. The output is a tensor shape `[..., M, K]` where each output
matrix satisfies matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]. *)
val batchMatrixSolve
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

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
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> [ `double ] Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Solves systems of linear equations with upper or lower triangular matrices by *)
(* backsubstitution.

`matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
square matrices. If `lower` is `True` then the strictly upper triangular part
of each inner-most matrix is ignored. If `lower` is False then the strictly
lower triangular part of each inner-most matrix is ignored. `rhs` is a tensor
of shape [..., M, K]`.

The output is a tensor of shape `[..., M, K]`. If `lower` is `True` then the
output satisfies
\\(\sum_{k=0}^{i}\\) matrix[..., i, k] * output[..., k, j] = rhs[..., i, j].
If `lower` is false then the strictly then the output satisfies
\\(sum_{k=i}^{K-1}\\) matrix[..., i, k] * output[..., k, j] = rhs[..., i, j]. *)
val batchMatrixTriangularSolve
  :  ?name:string
  -> ?lower:bool
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Batch normalization. *)
(* This op is deprecated. Prefer `tf.nn.batch_normalization`. *)
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

(* Calculates the Eigen Decomposition of a batch of square self-adjoint matrices. *)
(* The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix
SelfAdjointEig.

The result is a '[..., M+1, M] matrix with [..., 0,:] containing the
eigenvalues, and subsequent [...,1:, :] containing the eigenvectors. *)
val batchSelfAdjointEig
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

(* Adds `bias` to `value`. *)
(* This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions. *)
val biasAdd
  :  ?name:string
  -> ?data_format:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* The backward operation for "BiasAdd" on the "bias" tensor. *)
(* It accumulates all the values from out_backprop into the feature dimension.
For NHWC data format, the feature dimension is the last. For NCHW data format,
the feature dimension is the third-to-last. *)
val biasAddGrad
  :  ?name:string
  -> ?data_format:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Adds `bias` to `value`. *)
(* This is a deprecated version of BiasAdd and will be soon removed.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions. *)
val biasAddV1
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

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
  -> type_ : ([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) Node.Type.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 'type__) Node.t

(* Cast x of type SrcT to y of DstT. *)
val cast
  :  ?name:string
  -> type_ : 'dstT Node.Type.t
  -> 'srcT Node.t
  -> 'dstT Node.t

(* Returns element-wise smallest integer in not less than x. *)
val ceil
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Checks a tensor for NaN and Inf values. *)
(* When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is. *)
val checkNumerics
  :  ?name:string
  -> message:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Calculates the Cholesky decomposition of a square matrix. *)
(* The input has to be symmetric and positive definite. Only the lower-triangular
part of the input will be used for this operation. The upper-triangular part
will not be read.

The result is the lower-triangular matrix of the Cholesky decomposition of the
input. *)
val cholesky
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

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
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `complex64 ] Node.t

(* Computes the complex absolute value of a tensor. *)
(* Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` that is the absolute value of each element in `x`. All elements in `x`
must be complex numbers of the form \\(a + bj\\). The absolute value is
computed as \\( \sqrt{a^2 + b^2}\\).

For example:

```
# tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
tf.complex_abs(x) ==> [5.25594902, 6.60492229]
``` *)
val complexAbs
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `float ] Node.t

(* Concatenates tensors along one dimension. *)
val concat
  :  ?name:string
  -> n:int
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

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
  -> n:int
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t

(* Returns the complex conjugate of a complex number. *)
(* Given a tensor `in` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `in`. The
complex numbers in `in` must be of the form \\(a + bj\\), where *a* is the real
part and *b* is the imaginary part.

The complex conjugate returned by this operation is of the form \\(a - bj\\).

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.conj(in) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
``` *)
val conj
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `complex64 ] Node.t

(* Does nothing. Serves as a control trigger for scheduling. Only useful as a *)
(* placeholder for control edges. *)
val controlTrigger
  :  ?name:string
  -> unit
  -> [ `unit ] Node.t

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

In detail, with the default NCHW format,

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
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes the gradients of convolution with respect to the filter. *)
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

(* Computes the gradients of convolution with respect to the input. *)
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

(* Computes cos of x element-wise. *)
val cos
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Increments 'ref' until it reaches 'limit'. *)
(* This operation outputs "ref" after the update is done.  This makes it
easier to chain operations that need to use the updated value. *)
val countUpTo
  :  ?name:string
  -> limit:int
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 't) Node.t

(* Compute the pairwise cross product. *)
(* `a` and `b` must be the same shape; they can either be simple 3-element vectors,
or any shape where the innermost dimension is 3. In the latter case, each pair
of corresponding 3-element vectors is cross-multiplied independently. *)
val cross
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Convert JSON-encoded Example records to binary protocol buffer strings. *)
(* This op translates a tensor containing Example records, encoded using
the [standard JSON
mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
into a tensor containing the same records encoded as binary protocol
buffers. The resulting tensor can then be fed to any of the other
Example-parsing ops. *)
val decodeJSONExample
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t

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
  -> type_ : 'dtype Node.Type.t
  -> ?channels:int
  -> [ `string ] Node.t
  -> 'dtype Node.t

(* Reinterpret the bytes of a string as a vector of numbers. *)
val decodeRaw
  :  ?name:string
  -> type_ : ([< `float | `double | `int32 | `int64 ] as 'out_type) Node.Type.t
  -> ?little_endian:bool
  -> [ `string ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 'out_type) Node.t

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
  -> 't Node.t
  -> 't Node.t

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
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes the gradients of depthwise convolution with respect to the filter. *)
val depthwiseConv2dNativeBackpropFilter
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes the gradients of depthwise convolution with respect to the input. *)
val depthwiseConv2dNativeBackpropInput
  :  ?name:string
  -> strides:int list
  -> padding:string
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

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
  -> 't Node.t
  -> 't Node.t

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
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Computes Psi, the derivative of Lgamma (the log of the absolute value of *)
(* `Gamma(x)`), element-wise. *)
val digamma
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Returns x / y element-wise. *)
val div
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t

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
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

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
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

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
  -> n:int
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

(* Computes the (possibly normalized) Levenshtein Edit Distance. *)
(* The inputs are variable-length sequences provided by SparseTensors
  (hypothesis_indices, hypothesis_values, hypothesis_shape)
and
  (truth_indices, truth_values, truth_shape).

The inputs are: *)
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

(* Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise. *)
(* See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289) *)
val elu
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes gradients for the exponential linear (Elu) operation. *)
val eluGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

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
  -> 't Node.t
  -> [ `string ] Node.t

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
  -> 't Node.t
  -> 't Node.t

(* Returns the truth value of (x == y) element-wise. *)
val equal
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> [ `bool ] Node.t

(* Computes the Gauss error function of `x` element-wise. *)
val erf
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Computes the complementary error function of `x` element-wise. *)
val erfc
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Exits the current frame to its parent frame. *)
(* Exit makes its input `data` available to the parent frame. *)
val exit
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

(* Computes exponential of x element-wise.  \\(y = e^x\\). *)
val exp
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

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
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

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
  -> [ `float ] Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

(* Compute the 2-dimensional discrete Fourier Transform. *)
val fFT2D
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `complex64 ] Node.t

(* A queue that produces elements in first-in first-out order. *)
val fIFOQueue
  :  ?name:string
  -> component_types:Type.p list
  -> ?shapes:Dim.t list list
  -> ?capacity:int
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

(* Output a fact about factorials. *)
val fact
  :  ?name:string
  -> unit
  -> [ `string ] Node.t

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
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

(* A Reader that outputs fixed-length records from a file. *)
val fixedLengthRecordReader
  :  ?name:string
  -> ?header_bytes:int
  -> record_bytes:int
  -> ?footer_bytes:int
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

(* Returns element-wise largest integer not greater than x. *)
val floor
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

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
  -> 'tparams Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> 'tparams Node.t

(* Returns the truth value of (x > y) element-wise. *)
val greater
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

(* Returns the truth value of (x >= y) element-wise. *)
val greaterEqual
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

(* Convert one or more images from HSV to RGB. *)
(* Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

See `rgb_to_hsv` for a description of the HSV encoding. *)
val hSVToRGB
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

(* Creates a non-initialized hash table. *)
(* This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable. *)
val hashTable
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

(* Outputs a `Summary` protocol buffer with a histogram. *)
(* The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `OutOfRange` error if any value is not finite. *)
val histogramSummary
  :  ?name:string
  -> [ `string ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `string ] Node.t

(* Compute the inverse 2-dimensional discrete Fourier Transform. *)
val iFFT2D
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `complex64 ] Node.t

(* Return a tensor with the same shape and contents as the input tensor or value. *)
val identity
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

(* A Reader that outputs the queued work as both the key and value. *)
(* To use, enqueue strings in a Queue.  ReaderRead will take the front
work string and output (work, work). *)
val identityReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

(* Returns the imaginary part of a complex number. *)
(* Given a tensor `in` of complex numbers, this operation returns a tensor of type
`float` that is the imaginary part of each element in `in`. All elements in `in`
must be complex numbers of the form \\(a + bj\\), where *a* is the real part
and *b* is the imaginary part returned by this operation.

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(in) ==> [4.75, 5.75]
``` *)
val imag
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `float ] Node.t

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
  -> [ `string ] Node.t
  -> ([< `float ] as 't) Node.t
  -> [ `string ] Node.t

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
  -> [ `float ] Node.t
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

(* Table initializer that takes two tensors for keys and values respectively. *)
val initializeTable
  :  ?name:string
  -> [ `string ] Node.t
  -> 'tkey Node.t
  -> 'tval Node.t
  -> [ `unit ] Node.t

(* Computes the reciprocal of x element-wise. *)
(* I.e., \\(y = 1 / x\\). *)
val inv
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

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
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t

(* Returns which elements of x are finite. *)
val isFinite
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `bool ] Node.t

(* Returns which elements of x are Inf. *)
val isInf
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `bool ] Node.t

(* Returns which elements of x are NaN. *)
val isNan
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `bool ] Node.t

(* L2 Loss. *)
(* Computes half the L2 norm of a tensor without the `sqrt`:

    output = sum(t ** 2) / 2 *)
val l2Loss
  :  ?name:string
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

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
  -> [ `float ] Node.t
  -> [ `float ] Node.t

(* Gradients for Local Response Normalization. *)
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

(* Returns the truth value of (x < y) element-wise. *)
val less
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

(* Returns the truth value of (x <= y) element-wise. *)
val lessEqual
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `bool ] Node.t

(* Computes the log of the absolute value of `Gamma(x)` element-wise. *)
val lgamma
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

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
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes natural logarithm of x element-wise. *)
(* I.e., \\(y = \log_e x\\). *)
val log
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Returns the truth value of x AND y element-wise. *)
val logicalAnd
  :  ?name:string
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t

(* Returns the truth value of NOT x element-wise. *)
val logicalNot
  :  ?name:string
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t

(* Returns the truth value of x OR y element-wise. *)
val logicalOr
  :  ?name:string
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t

(* Looks up keys in a table, outputs the corresponding values. *)
(* The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values. *)
val lookupTableFind
  :  ?name:string
  -> [ `string ] Node.t
  -> 'tin Node.t
  -> 'tout Node.t
  -> 'tout Node.t

(* Computes the number of elements in the given table. *)
val lookupTableSize
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int64 ] Node.t

(* Forwards the input to the output. *)
(* This operator represents the loop termination condition used by the
"pivot" switches of a loop. *)
val loopCond
  :  ?name:string
  -> [ `bool ] Node.t
  -> [ `bool ] Node.t

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
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 ] as 't) Node.t

(* Returns the set of files matching a pattern. *)
(* Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion. *)
val matchingFiles
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t

(* Calculates the determinant of a square matrix. *)
val matrixDeterminant
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Calculates the inverse of a square invertible matrix. *)
(* The op uses the Cholesky decomposition if the matrix is symmetric positive
definite and LU decomposition with partial pivoting otherwise.

If the matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result. *)
val matrixInverse
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Solves a system of linear equations. Checks for invertibility. *)
val matrixSolve
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

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
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> [ `double ] Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Solves a system of linear equations with an upper or lower triangular matrix by *)
(* backsubstitution.

`matrix` is a matrix of shape `[M, M]`. If `lower` is `True` then the strictly
upper triangular part of `matrix` is ignored. If `lower` is False then the
strictly lower triangular part of `matrix` is ignored. `rhs` is a matrix of
shape [M, K]`.

The output is a matrix of shape `[M, K]`. If `lower` is `True` then the output
satisfies \\(\sum_{k=0}^{i}\\) matrix[i, k] * output[k, j] = rhs[i, j].
If `lower` is false then output satisfies
\\(\sum_{k=i}^{K-1}\\) matrix[i, k] * output[k, j] = rhs[i, j]. *)
val matrixTriangularSolve
  :  ?name:string
  -> ?lower:bool
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes the maximum of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val max
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Performs max pooling on the input. *)
val maxPool
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> ?data_format:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

(* Computes gradients of the maxpooling function. *)
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

(* Computes gradients of the maxpooling function. *)
val maxPoolGradWithArgmax
  :  ?name:string
  -> ksize:int list
  -> strides:int list
  -> padding:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> ([< `int32 | `int64 ] as 'targmax) Node.t
  -> [ `float ] Node.t

(* Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts. *)
val maximum
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Computes the mean of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val mean
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Merges summaries. *)
(* This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag. *)
val mergeSummary
  :  ?name:string
  -> n:int
  -> [ `string ] Node.t
  -> [ `string ] Node.t

(* Computes the minimum of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val min
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts. *)
val minimum
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

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
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

(* Returns element-wise remainder of division. *)
val mod_
  :  ?name:string
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t

(* Returns x * y element-wise. *)
val mul
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 ] as 't) Node.t

(* Computes numerical negative value element-wise. *)
(* I.e., \\(y = -x\\). *)
val neg
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Training via negative sampling. *)
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

(* Makes its input available to the next iteration. *)
val nextIteration
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

(* Does nothing. Only useful as a placeholder for control edges. *)
val noOp
  :  ?name:string
  -> unit
  -> [ `unit ] Node.t

(* Returns the truth value of (x != y) element-wise. *)
val notEqual
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 | `complex64 | `string ] as 't) Node.t
  -> [ `bool ] Node.t

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
  -> [ `int64 ] Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

(* Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor. *)
(* Packs the `N` tensors in `values` into a tensor with rank one higher than each
tensor in `values` and shape `[N] + values[0].shape`. The output satisfies
`output[i, ...] = values[i][...]`.

This is the opposite of `unpack`. *)
val pack
  :  ?name:string
  -> n:int
  -> 't Node.t
  -> 't Node.t

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
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

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
  -> [ `string ] Node.t

(* A placeholder op for a value that will be fed into the computation. *)
(* N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime. *)
val placeholder
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> ?shape:Dim.t list
  -> unit
  -> 'dtype Node.t

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
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Computes the product of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val prod
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* Closes the given queue. *)
(* This operation signals that no more elements will be enqueued in the
given queue. Subsequent Enqueue(Many) operations will fail.
Subsequent Dequeue(Many) operations will continue to succeed if
sufficient elements remain in the queue. Subsequent Dequeue(Many)
operations that would block will fail immediately. *)
val queueClose
  :  ?name:string
  -> ?cancel_pending_enqueues:bool
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

(* Computes the number of elements in the given queue. *)
val queueSize
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t

(* Converts one or more images from RGB to HSV. *)
(* Outputs a tensor of the same shape as the `images` tensor, containing the HSV
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
`output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue. *)
val rGBToHSV
  :  ?name:string
  -> [ `float ] Node.t
  -> [ `float ] Node.t

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
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int64 ] Node.t
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t

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
  -> 't Node.t
  -> 't Node.t

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
  -> [ `string ] Node.t

(* Outputs random values from a normal distribution. *)
(* The generated values will have mean 0 and standard deviation 1. *)
val randomStandardNormal
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'dtype) Node.Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double ] as 'dtype) Node.t

(* Outputs random values from a uniform distribution. *)
(* The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded. *)
val randomUniform
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'dtype) Node.Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double ] as 'dtype) Node.t

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
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tout) Node.t
  -> ([< `int32 | `int64 ] as 'tout) Node.t
  -> ([< `int32 | `int64 ] as 'tout) Node.t

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
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t

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
  -> 't Node.t
  -> [ `int32 ] Node.t

(* Reads and outputs the entire contents of the input filename. *)
val readFile
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t

(* Returns the number of records this Reader has produced. *)
(* This is the same as the number of ReaderRead executions that have
succeeded. *)
val readerNumRecordsProduced
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int64 ] Node.t

(* Returns the number of work units this Reader has finished processing. *)
val readerNumWorkUnitsCompleted
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int64 ] Node.t

(* Restore a Reader to its initial clean state. *)
val readerReset
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

(* Restore a reader to a previously saved state. *)
(* Not all Readers support being restored, so this can produce an
Unimplemented error. *)
val readerRestoreState
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

(* Produce a string tensor that encodes the state of a Reader. *)
(* Not all Readers support being serialized, so this can produce an
Unimplemented error. *)
val readerSerializeState
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `string ] Node.t

(* Returns the real part of a complex number. *)
(* Given a tensor `in` of complex numbers, this operation returns a tensor of type
`float` that is the real part of each element in `in`. All elements in `in`
must be complex numbers of the form \\(a + bj\\), where *a* is the real part
returned by this operation and *b* is the imaginary part.

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.real(in) ==> [-2.25, 3.25]
``` *)
val real
  :  ?name:string
  -> [ `complex64 ] Node.t
  -> [ `float ] Node.t

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
  -> 't Node.t
  -> 't Node.t

(* Exits the current frame to its parent frame. *)
(* Exit makes its input `data` available to the parent frame. *)
val refExit
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

(* Return the same ref tensor as the input ref tensor. *)
val refIdentity
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

(* Makes its input available to the next iteration. *)
val refNextIteration
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

(* Forwards the `index`th element of `inputs` to `output`. *)
val refSelect
  :  ?name:string
  -> n:int
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

(* Computes rectified linear: `max(features, 0)`. *)
val relu
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Computes rectified linear 6: `min(max(features, 0), 6)`. *)
val relu6
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Computes rectified linear 6 gradients for a Relu6 operation. *)
val relu6Grad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Computes rectified linear gradients for a Relu operation. *)
val reluGrad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
reshape(t, [3, 3]) ==> [[1, 2, 3]
                        [4, 5, 6]
                        [7, 8, 9]]

# tensor 't' is [[[1, 1], [2, 2]]
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2]
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
# -1 can also be used with higher dimensional shapes
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]

# tensor 't' is [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
``` *)
val reshape
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

(* Resize `images` to `size` using area interpolation. *)
(* Input images can be of different types but output images are always float. *)
val resizeArea
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t

(* Resize `images` to `size` using bicubic interpolation. *)
(* Input images can be of different types but output images are always float. *)
val resizeBicubic
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t

(* Resize `images` to `size` using bilinear interpolation. *)
(* Input images can be of different types but output images are always float. *)
val resizeBilinear
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t

(* Computes the gradient of bilinear interpolation. *)
val resizeBilinearGrad
  :  ?name:string
  -> ?align_corners:bool
  -> [ `float ] Node.t
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Resize `images` to `size` using nearest neighbor interpolation. *)
val resizeNearestNeighbor
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `int32 | `int64 | `float | `double ] as 't) Node.t

(* Computes the gradient of nearest neighbor interpolation. *)
val resizeNearestNeighborGrad
  :  ?name:string
  -> ?align_corners:bool
  -> ([< `int32 | `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `int32 | `float | `double ] as 't) Node.t

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
  -> type_ : 'dt Node.Type.t
  -> ?preferred_shard:int
  -> [ `string ] Node.t
  -> [ `string ] Node.t
  -> 'dt Node.t

(* Restores a tensor from checkpoint files. *)
(* This is like `Restore` except that restored tensor can be listed as filling
only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
larger tensor and the slice that the restored tensor covers.

The `shape_and_slice` input has the same format as the
elements of the `shapes_and_slices` input of the `SaveSlices` op. *)
val restoreSlice
  :  ?name:string
  -> type_ : 'dt Node.Type.t
  -> ?preferred_shard:int
  -> [ `string ] Node.t
  -> [ `string ] Node.t
  -> [ `string ] Node.t
  -> 'dt Node.t

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
  -> ([< `int32 | `bool | `float | `double ] as 't) Node.t
  -> [ `bool ] Node.t
  -> ([< `int32 | `bool | `float | `double ] as 't) Node.t

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
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> 't Node.t

(* Computes reciprocal of square root of x element-wise. *)
(* I.e., \\(y = 1 / \sqrt{x}\\). *)
val rsqrt
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Outputs a `Summary` protocol buffer with scalar values. *)
(* The input `tags` and `values` must have the same shape.  The generated summary
has a summary value for each tag-value pair in `tags` and `values`. *)
val scalarSummary
  :  ?name:string
  -> [ `string ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `string ] Node.t

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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

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
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

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
  -> 't Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> 't Node.t
  -> 't Node.t

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
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
  -> [ `bool ] Node.t
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

(* Calculates the Eigen Decomposition of a square Self-Adjoint matrix. *)
(* Only the lower-triangular part of the input will be used in this case. The
upper-triangular part will not be read.

The result is a M+1 x M matrix whose first row is the eigenvalues, and
subsequent rows are eigenvectors. *)
val selfAdjointEig
  :  ?name:string
  -> ([< `double | `float ] as 't) Node.t
  -> ([< `double | `float ] as 't) Node.t

(* Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`. *)
(* The `SparseTensor` must have rank `R` greater than 1, and the first dimension
is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The serialized
`SparseTensor` objects going into each row of `serialized_sparse` will have
rank `R-1`.

The minibatch size `N` is extracted from `sparse_shape[0]`. *)
val serializeManySparse
  :  ?name:string
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> [ `string ] Node.t

(* Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object. *)
val serializeSparse
  :  ?name:string
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> [ `string ] Node.t

(* Returns the shape of a tensor. *)
(* This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```prettyprint
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
``` *)
val shape
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t

(* Returns shape of tensors. *)
(* This operation returns N 1-D integer tensors representing shape of `input[i]s`. *)
val shapeN
  :  ?name:string
  -> n:int
  -> 't Node.t
  -> [ `int32 ] Node.t

(* Generate a sharded filename. The filename is printf formatted as *)
(*    %s-%05d-of-%05d, basename, shard, num_shards. *)
val shardedFilename
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `string ] Node.t

(* Generate a glob pattern matching all sharded file names. *)
val shardedFilespec
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t
  -> [ `string ] Node.t

(* Computes sigmoid of `x` element-wise. *)
(* Specifically, `y = 1 / (1 + exp(-x))`. *)
val sigmoid
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Returns an element-wise indication of the sign of a number. *)
(* y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0. *)
val sign
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Computes sin of x element-wise. *)
val sin
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

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
  -> 't Node.t
  -> [ `int32 ] Node.t

(* Return a slice from 'input'. *)
(* The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.

*Requirements*:
  0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n) *)
val slice
  :  ?name:string
  -> 't Node.t
  -> ([< `int32 | `int64 ] as 'index) Node.t
  -> ([< `int32 | `int64 ] as 'index) Node.t
  -> 't Node.t

(* Computes softmax activations. *)
(* For each batch `i` and class `j` we have

    softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i])) *)
val softmax
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes softplus: `log(exp(features) + 1)`. *)
val softplus
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Computes softplus gradients for a softplus operation. *)
val softplusGrad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Computes softsign: `features / (abs(features) + 1)`. *)
val softsign
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Computes softsign gradients for a softsign operation. *)
val softsignGrad
  :  ?name:string
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
x = [[ [1],   [2],  [5],  [6]],
     [ [3],   [4],  [7],  [8]],
     [ [9],  [10], [13],  [14]],
     [ [11], [12], [15],  [16]]]
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
  -> 't Node.t
  -> 't Node.t

(* Update relevant entries in '*var' and '*accum' according to the adagrad scheme. *)
(* That is for rows we have grad for, we update var and accum as follows:
accum += grad * grad
var -= lr * grad * (1 / sqrt(accum)) *)
val sparseApplyAdagrad
  :  ?name:string
  -> ?use_locking:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

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

(* Update relevant entries in '*var' and '*accum' according to the momentum scheme. *)
(* That is for rows we have grad for, we update var and accum as follows:

accum = accum * momentum + grad
var -= lr * accum *)
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
  -> [ `float ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

(* Computes the mean along sparse segments of a tensor. *)
(* Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`. *)
val sparseSegmentMean
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes gradients for SparseSegmentMean. *)
(* Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0. *)
val sparseSegmentMeanGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes the sum along sparse segments of a tensor divided by the sqrt of N. *)
(* N is the size of the segment being reduced.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments. *)
val sparseSegmentSqrtN
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

(* Computes gradients for SparseSegmentSqrtN. *)
(* Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0. *)
val sparseSegmentSqrtNGrad
  :  ?name:string
  -> ([< `float | `double ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double ] as 't) Node.t

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
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

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
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> 't Node.t
  -> 't Node.t

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
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> 't Node.t
  -> 't Node.t
  -> 't Node.t

(* Splits a tensor into `num_split` tensors along one dimension. *)
val split
  :  ?name:string
  -> num_split:int
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> 't Node.t

(* Computes square root of x element-wise. *)
(* I.e., \\(y = \sqrt{x} = x^{1/2}\\). *)
val sqrt
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Computes square of x element-wise. *)
(* I.e., \\(y = x * x = x^2\\). *)
val square
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Returns (x - y)(x - y) element-wise. *)
val squaredDifference
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

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
  -> 't Node.t
  -> 't Node.t

(* A stack that produces elements in first-in last-out order. *)
val stack
  :  ?name:string
  -> ?stack_name:string
  -> unit
  -> [ `string ] Node.t

(* Delete the stack from its resource container. *)
val stackClose
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

(* Pop the element at the top of the stack. *)
val stackPop
  :  ?name:string
  -> type_ : 'elem_type Node.Type.t
  -> [ `string ] Node.t
  -> 'elem_type Node.t

(* Push an element onto the stack. *)
val stackPush
  :  ?name:string
  -> ?swap_memory:bool
  -> [ `string ] Node.t
  -> 't Node.t
  -> 't Node.t

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
  -> 't Node.t
  -> 't Node.t

(* Converts each string in the input Tensor to its hash mod by a number of buckets. *)
(* The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time. *)
val stringToHashBucket
  :  ?name:string
  -> num_buckets:int
  -> [ `string ] Node.t
  -> [ `int64 ] Node.t

(* Converts each string in the input Tensor to the specified numeric type. *)
(* (Note that int32 overflow results in an error while float overflow
results in a rounded value.) *)
val stringToNumber
  :  ?name:string
  -> type_ : ([< `float | `int32 ] as 'out_type) Node.Type.t
  -> [ `string ] Node.t
  -> ([< `float | `int32 ] as 'out_type) Node.t

(* Returns x - y element-wise. *)
val sub
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

(* Computes the sum of elements across dimensions of a tensor. *)
(* Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1. *)
val sum
  :  ?name:string
  -> ?keep_dims:bool
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int64 | `int32 | `complex64 ] as 't) Node.t

(* A Reader that outputs the records from a TensorFlow Records file. *)
val tFRecordReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

(* Computes hyperbolic tangent of `x` element-wise. *)
val tanh
  :  ?name:string
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t
  -> ([< `float | `double | `int32 | `complex64 | `int64 ] as 't) Node.t

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
  -> type_ : 'dtype Node.Type.t
  -> shape:Dim.t list
  -> ?var_name:string
  -> unit
  -> 'dtype Node.t

(* An array of Tensors of given size, with data written via Write and read *)
(* via Read or Pack. *)
val tensorArray
  :  ?name:string
  -> ?dynamic_size:bool
  -> ?tensor_array_name:string
  -> [ `int32 ] Node.t
  -> [ `string ] Node.t

(* Delete the TensorArray from its resource container.  This enables *)
(* the user to close and release the resource in the middle of a step/run. *)
val tensorArrayClose
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `unit ] Node.t

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
  -> [ `string ] Node.t
  -> [ `float ] Node.t
  -> [ `string ] Node.t

(* Pack the elements from the TensorArray. *)
(* All elements must have the same shape. *)
val tensorArrayPack
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> [ `string ] Node.t
  -> [ `float ] Node.t
  -> 'dtype Node.t

(* Read an element from the TensorArray. *)
val tensorArrayRead
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t
  -> [ `float ] Node.t
  -> 'dtype Node.t

(* Get the current size of the TensorArray. *)
val tensorArraySize
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `float ] Node.t
  -> [ `int32 ] Node.t

(* Split the data from the input value into TensorArray elements. *)
(* Assuming that `lengths` takes on values
  (n0, n1, ..., n(T-1))
and that `value` has shape
  (n0 + n1 + ... + n(T-1) x d0 x d1 x ...),
this splits values into a TensorArray with T tensors.

TensorArray index t will be the subtensor of values with starting position
  (n0 + n1 + ... + n(t-1), 0, 0, ...)
and having size
  nt x d0 x d1 x ... *)
val tensorArraySplit
  :  ?name:string
  -> [ `string ] Node.t
  -> 't Node.t
  -> [ `int64 ] Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

(* Unpack the data from the input value into TensorArray elements. *)
val tensorArrayUnpack
  :  ?name:string
  -> [ `string ] Node.t
  -> 't Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

(* Push an element onto the tensor_array. *)
val tensorArrayWrite
  :  ?name:string
  -> [ `string ] Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t
  -> [ `float ] Node.t
  -> [ `float ] Node.t

(* A Reader that outputs the lines of a file delimited by '\n'. *)
val textLineReader
  :  ?name:string
  -> ?skip_header_lines:int
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

(* Constructs a tensor by tiling a given tensor. *)
(* This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`. *)
val tile
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

(* Returns the gradient of `Tile`. *)
(* Since `Tile` takes an input and repeats the input `multiples` times
along each dimension, `TileGrad` takes in `multiples` and aggregates
each repeated tile of `input` into `output`. *)
val tileGrad
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

(* Shuffle dimensions of x according to a permutation. *)
(* The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]` *)
val transpose
  :  ?name:string
  -> 't Node.t
  -> [ `int32 ] Node.t
  -> 't Node.t

(* Outputs random values from a truncated normal distribution. *)
(* The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked. *)
val truncatedNormal
  :  ?name:string
  -> type_ : ([< `float | `double ] as 'dtype) Node.Type.t
  -> ?seed:int
  -> ?seed2:int
  -> ([< `int32 | `int64 ] as 't) Node.t
  -> ([< `float | `double ] as 'dtype) Node.t

(* Unpacks the outer dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors. *)
(* Unpacks `num` tensors from `value` by chipping it along the first dimension.
The i'th tensor in `output` is the slice `value[i, ...]`. Each tensor in
`output` has shape `value.shape[1:]`.

This is the opposite of `pack`. *)
val unpack
  :  ?name:string
  -> num:int
  -> 't Node.t
  -> 't Node.t

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
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t
  -> ([< `int32 | `int64 ] as 'tindices) Node.t
  -> [ `int32 ] Node.t
  -> ([< `float | `double | `int32 | `int64 ] as 't) Node.t

(* Holds state in the form of a tensor that persists across steps. *)
(* Outputs a ref to the tensor state so it may be read or modified.
TODO(zhifengc/mrry): Adds a pointer to a more detail document
about sharing states in tensorflow. *)
val variable
  :  ?name:string
  -> type_ : 'dtype Node.Type.t
  -> shape:Dim.t list
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> 'dtype Node.t

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
  -> [ `bool ] Node.t
  -> [ `int64 ] Node.t

(* A Reader that outputs the entire contents of a file as a value. *)
(* To use, enqueue filenames in a Queue.  The output of ReaderRead will
be a filename (key) and the contents of that file (value). *)
val wholeFileReader
  :  ?name:string
  -> ?container:string
  -> ?shared_name:string
  -> unit
  -> [ `string ] Node.t

(* Returns a tensor of zeros with the same shape and type as x. *)
val zerosLike
  :  ?name:string
  -> 't Node.t
  -> 't Node.t

