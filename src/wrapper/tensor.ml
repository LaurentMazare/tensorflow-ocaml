type ('a, 'b) t =
  { data : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
  ; kind : ('a, 'b) Bigarray.kind
  }

type p = P : (_, _) t -> p
