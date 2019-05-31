val create : int list -> type_:'a Node.Type.t -> init:'a Node.t -> 'a Node.t
val float : int list -> init:[ `float ] Node.t -> [ `float ] Node.t
val double : int list -> init:[ `double ] Node.t -> [ `double ] Node.t

val f_or_d
  :  int list
  -> float
  -> type_:([< `float | `double ] as 'a) Node.Type.t
  -> 'a Node.t

val f : int list -> float -> [ `float ] Node.t
val d : int list -> float -> [ `double ] Node.t

val normal
  :  int list
  -> stddev:float
  -> type_:([< `float | `double ] as 'a) Node.Type.t
  -> 'a Node.t

val normalf : int list -> stddev:float -> [ `float ] Node.t
val normald : int list -> stddev:float -> [ `double ] Node.t

val truncated_normal
  :  int list
  -> stddev:float
  -> type_:([< `float | `double ] as 'a) Node.Type.t
  -> 'a Node.t

val truncated_normalf : int list -> stddev:float -> [ `float ] Node.t
val truncated_normald : int list -> stddev:float -> [ `double ] Node.t

val uniform
  :  int list
  -> lo:float
  -> hi:float
  -> type_:([< `float | `double ] as 'a) Node.Type.t
  -> 'a Node.t

val uniformf : int list -> lo:float -> hi:float -> [ `float ] Node.t
val uniformd : int list -> lo:float -> hi:float -> [ `double ] Node.t
val load_f : int list -> filename:string -> tensor:string -> [ `float ] Node.t
val load_d : int list -> filename:string -> tensor:string -> [ `double ] Node.t

(** [get_all_vars nodes] returns all the variables that can be used when
    evaluating a node in [nodes].
    Each variable is only returned once.
*)
val get_all_vars : Node.p list -> Node.p list
