open Core_kernel.Std
exception Shape_mismatch of int list * int list * string

(* TODO: handle double ? *)
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

module Shared_var = struct

  let with_shape ~f g =
    let shape_a = ref (`F f) in
    let f t =
      let s = t.shape in
      match !shape_a with
      | `F f ->
        let a = f ~shape:s in
        shape_a := `Computed (s, a);
        a
      | `Computed (shape, a) ->
        if s <> shape
        then failwith "Dimensions do not match"
        else a
    in
    Staged.stage (g f)

  let dense ~shape =
    with_shape
      ~f:(fun ~shape:input_shape ->
          if List.length shape <> List.length input_shape
          then
            failwithf "Dense has different input and output shape sizes %d<>%d"
              (List.length shape)
              (List.length input_shape) ();
          match shape, input_shape with
          | [ output_size ], [ input_size ] ->
            let w = Var.f [ input_size; output_size ] 0. in
            let b = Var.f [ output_size ] 0. in
            (w, b)
          | _ -> failwith "TODO")
      (fun f t ->
         let w, b = f t in
         let node = Ops.(t.node *^ w + b) in
         { shape
         ; node
         ; variables = [ w; b ]
         })
end

let unary op t = { t with node = op t.node }

let sigmoid = unary Ops.sigmoid
let relu = unary Ops.relu
let tanh = unary Ops.tanh
let softmax = unary Ops.softmax

let dense t ~shape =
  Staged.unstage (Shared_var.dense ~shape) t

let concat t1 t2 =
  { variables = t1.variables @ t2.variables;
    shape = List.zip_exn t1.shape t2.shape |> List.map ~f:(fun (i,j) -> i + j);
    node = Ops.(concat one32 [ t1.node; t2.node ])}

let binary ~op_name op t1 t2 =
  if t1.shape <> t2.shape
  then raise (Shape_mismatch (t1.shape, t2.shape, op_name));
  { node = op t1.node t2.node
  ; shape = t1.shape
  ; variables = t1.variables @ t2.variables
  }

let ( * ) = binary ~op_name:"Mul" Ops.( * )

let (+) = binary ~op_name:"Add" Ops.(+)
let (-) = binary ~op_name:"Add" Ops.(-)

let f c =
  { shape = []
  ; node = Ops.f c
  ; variables = []
  }

module Model = struct
  type net = t
  type t
  type optimizer =
    | Gradient_descent of float

  type loss =
    | Cross_entropy

  let create _net = failwith "TODO"

  let fit t ~loss ~optimizer ~epochs ~xs ~ys =
    ignore (t, loss, optimizer, epochs, xs, ys);
    failwith "TODO"

  let evaluate t xs =
    ignore (t, xs);
    failwith "TODO"
end
