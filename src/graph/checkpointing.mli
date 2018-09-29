(** [loop ~start_index ~end_index ~save_vars ~checkpoint_base f] starts a loop
    ranging from [start_index] to [end_index]. On each iteration [f] is called
    on the current index.

    This loop is checkpointed: regularly the state of all the variables
    reachable from [save_vars_from] are saved on disk to file
    checkpoint_baseXXX.
    If such files already exist when starting the loop, the last one of this
    file is used to restore the variables content.

    Note that this assumes that the graph is generated in the same way on both
    side as variables need to have the same ids in current graph and in the
    loaded checkpoint.
*)
val loop
  :  start_index:int
  -> end_index:int
  -> save_vars_from:Node.p list
  -> checkpoint_base:string
  -> ?checkpoint_every:[ `iters of int | `seconds of float ] (* default : `second 600 *)
  -> (index:int -> unit)
  -> unit
