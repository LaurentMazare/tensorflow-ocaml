open Core_kernel.Std

let table = Node.Op_name.Table.create ()

type t =
  { f : 'a .
          (  self:([< `float | `double] as 'a) Node.t
          -> gradient:'a Node.t
          -> Node.p option list)
  }

let add op t =
  let f ~self:(Node.P self) ~gradient:(Node.P gradient) =
    match self.output_type, gradient.output_type with
    | Node.Type.Double, Node.Type.Double -> t.f ~self ~gradient
    | Node.Type.Float, Node.Type.Float -> t.f ~self ~gradient
    | _, _ ->
      failwithf "Inconsistent types %s" (Node.Op_name.to_string op) ()
  in
  Hashtbl.set table ~key:op ~data:f

let find = Hashtbl.find table
