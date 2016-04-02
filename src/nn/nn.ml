open Core_kernel.Std

type t =
  { shape : int list (* output shape *)
  ; node : [ `float ] Node.t
  ; variables : [ `float ] Node.t list
  }

let input ~shape =
  let placeholder = Ops.placeholder ~type_:Float shape in
  let t =
    { shape
    ; node = placeholder
    ; variables = []
    }
  in
  placeholder, t

let dense t ~shape =
  if List.length shape <> List.length t.shape
  then
    failwithf "Dense has different input and output shape sizes %d<>%d"
      (List.length shape)
      (List.length t.shape) ();
  match shape, t.shape with
  | [ output_size ], [ input_size ] ->
    let w = Var.f [ input_size; output_size ] 0. in
    let b = Var.f [ output_size ] 0. in
    let node = Ops.(t.node *^ w + b) in
    { shape
    ; node
    ; variables = [ w; b ]
    }
  | _ -> failwith "TODO"

let sigmoid t =
  { t with node = Ops.sigmoid t.node }

let relu t =
  { t with node = Ops.relu t.node }

let tanh t =
  { t with node = Ops.tanh t.node }
