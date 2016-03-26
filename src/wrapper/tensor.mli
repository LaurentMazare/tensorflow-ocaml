type ('a, 'b) t = ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

type p = P : (_, _) t -> p

val print : p -> unit

val to_elt_list : ('a, 'b) t -> 'a list
val to_float_list : p -> float list
