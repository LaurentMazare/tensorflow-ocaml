type ('a, 'b) t = ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

type p = P : (_, _) t -> p

val print : p -> unit

val to_elt_list : ('a, 'b) t -> 'a list
val to_float_list : p -> float list

val create : ('a, 'b) Bigarray.kind -> int array -> ('a, 'b) t
val create1 : ('a, 'b) Bigarray.kind -> int -> ('a, 'b) t
val create2 : ('a, 'b) Bigarray.kind -> int -> int -> ('a, 'b) t
val create3 : ('a, 'b) Bigarray.kind -> int -> int -> int -> ('a, 'b) t
