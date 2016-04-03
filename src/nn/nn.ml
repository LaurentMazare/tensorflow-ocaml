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

module Shared_var = struct

  let with_shape ~f g =
    let shape = ref None in
    let set_shape s =
      match !shape with
      | None -> shape := Some s
      | Some shape ->
        if s <> shape
        then failwith "Dimensions do not match"
    in
    let v = lazy (f ~shape:(Option.value_exn !shape)) in
    let f t =
      set_shape (t.shape);
      Lazy.force v
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

let sigmoid t =
  { t with node = Ops.sigmoid t.node }

let relu t =
  { t with node = Ops.relu t.node }

let tanh t =
  { t with node = Ops.tanh t.node }


let dense t ~shape =
  Staged.unstage (Shared_var.dense ~shape) t

let concat t1 t2 =
  { variables = t1.variables @ t2.variables;
    shape = List.zip_exn t1.shape t2.shape |> List.map ~f:(fun (i,j) -> i + j);
    node = Ops.(concat one32 [ t1.node; t2.node ])}

let ( * ) t1 t2 =
  {t1 with
   node = Ops.( * ) t1.node t2.node;
   variables = t1.variables @ t2.variables
  }

let ( + ) t1 t2 =
  {t1 with
   node = Ops.( + ) t1.node t2.node;
   variables = t1.variables @ t2.variables
  }

let ( - ) t1 t2 =
  {t1 with
   node = Ops.( - ) t1.node t2.node;
   variables = t1.variables @ t2.variables
  }

let f c =
  {shape = [];
   node = Ops.f c;
   variables = []}
