open Base
open Tensorflow

val lstm
  :  size_c:int
  -> size_x:int
  -> (h:[ `float ] Node.t
      -> x:[ `float ] Node.t
      -> c:[ `float ] Node.t
      -> [ `h of [ `float ] Node.t ] * [ `c of [ `float ] Node.t ])
     Staged.t

val lstm_d
  :  size_c:int
  -> size_x:int
  -> (h:[ `double ] Node.t
      -> x:[ `double ] Node.t
      -> c:[ `double ] Node.t
      -> [ `h of [ `double ] Node.t ] * [ `c of [ `double ] Node.t ])
     Staged.t

val gru
  :  size_h:int
  -> size_x:int
  -> (h:[ `float ] Node.t -> x:[ `float ] Node.t -> [ `float ] Node.t) Staged.t

val gru_d
  :  size_h:int
  -> size_x:int
  -> (h:[ `double ] Node.t -> x:[ `double ] Node.t -> [ `double ] Node.t) Staged.t

module Unfold : sig
  (* [unfold ~xs ~seq_len ~dim ~init ~f] returns the full sequence obtained by applying
     recursively [f] on [seq_len] slices of [xs]. The initial memory value is [init].
     [xs] should have shape [(batch_dim, seq_len, dim)].
  *)
  val unfold
    :  xs:'b Node.t
    -> seq_len:int
    -> dim:int
    -> init:'a
    -> f:(x:'b Node.t -> mem:'a -> 'b Node.t * [ `mem of 'a ])
    -> 'b Node.t

  (* This is similar to [unfold] except that only the last output of [f] is returned rather
     than the full sequence.
  *)
  val unfold_last
    :  xs:'b Node.t
    -> seq_len:int
    -> input_dim:int
    -> output_dim:int
    -> init:'a
    -> f:(x:'b Node.t -> mem:'a -> 'b Node.t * [ `mem of 'a ])
    -> 'b Node.t option
end
