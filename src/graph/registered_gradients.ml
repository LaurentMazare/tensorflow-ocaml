open Base

let table = Hashtbl.create (module Node.Op_name)

type t =
  { f :
      'a. self:([< `float | `double ] as 'a) Node.t -> gradient:'a Node.t
      -> Node.p option list
  }

let add op t =
  let f ~self:(Node.P self) ~gradient:(Node.P gradient) =
    match Node.output_type self, Node.output_type gradient with
    | Node.Type.Double, Node.Type.Double -> t.f ~self ~gradient
    | Node.Type.Float, Node.Type.Float -> t.f ~self ~gradient
    | _, _ -> Printf.failwithf "Inconsistent types %s" (Node.Op_name.to_string op) ()
  in
  Hashtbl.set table ~key:op ~data:f

let find = Hashtbl.find table
let table_multi = Hashtbl.create (module Node.Op_name)

type multi =
  { g :
      'a. self:([< `float | `double ] as 'a) Node.t -> gradients:'a Node.t Map.M(Int).t
      -> Node.p option list
  }

let add_multi op t =
  let f ~self:(Node.P self) ~gradients =
    match Node.output_type self with
    | Node.Type.Double ->
      let gradients =
        Map.map gradients ~f:(fun gradient ->
            Option.value_exn (Node.extract gradient Double))
      in
      t.g ~self ~gradients
    | Node.Type.Float ->
      let gradients =
        Map.map gradients ~f:(fun gradient ->
            Option.value_exn (Node.extract gradient Float))
      in
      t.g ~self ~gradients
    | _ -> Printf.failwithf "Inconsistent types %s" (Node.Op_name.to_string op) ()
  in
  Hashtbl.set table_multi ~key:op ~data:f

let find_multi = Hashtbl.find table_multi
