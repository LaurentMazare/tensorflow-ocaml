type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt

type ('a, 'b) t

type p = P : (_, _) t -> p

val of_bigarray
  :  ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
  -> scalar:bool
  -> ('a, 'b) t

val to_bigarray : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
val print : p -> unit

val to_elt_list : ('a, 'b) t -> 'a list
val to_float_list : p -> float list
val copy_elt_list : ('a, 'b) t -> 'a list -> unit

val create : ('a, 'b) Bigarray.kind -> int array -> ('a, 'b) t
val create0 : ('a, 'b) Bigarray.kind -> ('a, 'b) t
val create1 : ('a, 'b) Bigarray.kind -> int -> ('a, 'b) t
val create2 : ('a, 'b) Bigarray.kind -> int -> int -> ('a, 'b) t
val create3 : ('a, 'b) Bigarray.kind -> int -> int -> int -> ('a, 'b) t

val copy : ('a, 'b) t -> ('a, 'b) t

val set : ('a, 'b) t -> int array -> 'a -> unit
val get : ('a, 'b) t -> int array -> 'a
val dims : ('a, 'b) t -> int array
val num_dims : ('a, 'b) t -> int
val kind : ('a, 'b) t -> ('a, 'b) Bigarray.kind
val sub_left : ('a, 'b) t -> int -> int -> ('a, 'b) t
val fill : ('a, 'b) t -> 'a -> unit
val blit : ('a, 'b) t -> ('a, 'b) t -> unit

type 'a eq =
  | Float : (float32_elt * [ `float ]) eq
  | Double : (float64_elt * [ `double ]) eq

val float32 : p -> (float, float32_elt) t option
val float64 : p -> (float, float64_elt) t option

val set_float_array1
  :  (float, 'a) t
  -> float array
  -> unit

val set_float_array2
  :  (float, 'a) t
  -> float array array
  -> unit

val of_float_array1
  :  float array
  -> (float, 'a) Bigarray.kind
  -> (float, 'a) t

val of_float_array2
  :  float array array
  -> (float, 'a) Bigarray.kind
  -> (float, 'a) t

val of_float_array3
  :  float array array array
  -> (float, 'a) Bigarray.kind
  -> (float, 'a) t

val to_float_array1
  :  (float, _) t
  -> float array

val to_float_array2
  :  (float, _) t
  -> float array array

val to_float_array3
  :  (float, _) t
  -> float array array array
