type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt

type ('a, 'b) t =
  { data : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
  ; scalar : bool
  }

type p = P : (_, _) t -> p

let create kind dims =
  let dims, scalar =
    if Array.length dims = 0
    then [| 1 |], true
    else dims, false
  in
  { data = Bigarray.Genarray.create kind Bigarray.c_layout dims
  ; scalar
  }

let to_bigarray t = t.data
let of_bigarray data ~scalar = { data; scalar }

let copy t =
  let copy =
    Bigarray.Genarray.create
      (Bigarray.Genarray.kind t.data)
      Bigarray.c_layout
      (Bigarray.Genarray.dims t.data)
  in
  Bigarray.Genarray.blit t.data copy;
  { data = copy
  ; scalar = t.scalar
  }

let create0 kind = create kind [||]
let create1 kind d = create kind [| d |]
let create2 kind d d' = create kind [| d; d' |]
let create3 kind d d' d'' = create kind [| d; d'; d'' |]

(* Abstract a couple Bigarray functions to have a coherent interface. *)
let get t indexes =
  let indexes =
    if t.scalar && Array.length indexes = 0
    then [| 0 |]
    else indexes
  in
  Bigarray.Genarray.get t.data indexes

let set t indexes v =
  let indexes =
    if t.scalar && Array.length indexes = 0
    then [| 0 |]
    else indexes
  in
  Bigarray.Genarray.set t.data indexes v

let dims t =
  if t.scalar
  then [||]
  else Bigarray.Genarray.dims t.data

let num_dims t =
  if t.scalar
  then 0
  else Bigarray.Genarray.num_dims t.data

let kind t = Bigarray.Genarray.kind t.data

let sub_left t start stop =
  { data = Bigarray.Genarray.sub_left t.data start stop
  ; scalar = t.scalar
  }

let fill t v = Bigarray.Genarray.fill t.data v

let blit t t' = Bigarray.Genarray.blit t.data t'.data

let print (P tensor) =
  let print (type a) (type b) (tensor : (a, b) t) (elt_to_string : a -> string) =
    match dims tensor with
    | [||] -> Printf.printf "%s\n%!" (get tensor [||] |> elt_to_string)
    | [| dim |] ->
      for d = 0 to dim - 1 do
        Printf.printf "%d %s\n%!"
          d (get tensor [| d |] |> elt_to_string)
      done
    | [| d0; d1 |] ->
      for x = 0 to d0 - 1 do
        Printf.printf "%d " x;
        for y = 0 to d1 - 1 do
          Printf.printf "%s "
            (get tensor [| x; y |] |> elt_to_string)
        done;
        Printf.printf "\n%!";
      done
    | otherwise -> Printf.printf "%d dims\n%!" (Array.length otherwise)
  in
  match kind tensor with
  | Bigarray.Float32 -> print tensor (Printf.sprintf "%f")
  | Bigarray.Float64 -> print tensor (Printf.sprintf "%f")
  | Bigarray.Int32 -> print tensor (fun i -> Printf.sprintf "%d" (Int32.to_int i))
  | Bigarray.Int64 -> print tensor (fun i -> Printf.sprintf "%d" (Int64.to_int i))
  | _ -> Printf.printf "Unsupported kind"

let to_elt_list : type a b. (a, b) t -> a list = fun tensor ->
  let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims tensor.data) in
  let tensor = Bigarray.reshape_1 tensor.data size in
  let result = ref [] in
  for i = size - 1 downto 0 do
    result := Bigarray.Array1.get tensor i :: !result
  done;
  !result

let to_float_list (P tensor) =
  let to_elt_list : type a. (float, a) t -> float list = to_elt_list in
  match kind tensor with
  | Bigarray.Float32 -> to_elt_list tensor
  | Bigarray.Float64 -> to_elt_list tensor
  | _ -> failwith "Not a float tensor"

let copy_elt_list : type a b. (a, b) t -> a list -> unit = fun t data ->
  let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims t.data) in
  let t_data = Bigarray.reshape_1 t.data size in
  List.iteri
    (fun i v -> Bigarray.Array1.set t_data i v)
    data

type 'a eq =
  | Float : (float32_elt * [ `float ]) eq
  | Double : (float64_elt * [ `double ]) eq

let float32 (P tensor) =
  match kind tensor with
  | Bigarray.Float32 -> Some (tensor : (float, float32_elt) t)
  | _ -> None

let float64 (P tensor) =
  match kind tensor with
  | Bigarray.Float64 -> Some (tensor : (float, float64_elt) t)
  | _ -> None

let set_float_array1 t array =
  let length = Array.length array in
  let num_dims = Bigarray.Genarray.num_dims t.data in
  if num_dims <> 1
  then failwith (Printf.sprintf "Improper number of dimension %d <> 1" num_dims);
  let data = Bigarray.array1_of_genarray t.data in
  let length' = Bigarray.Array1.dim data in
  if length <> length'
  then failwith (Printf.sprintf "Dimension mismatch %d <> %d" length length');
  for i = 0 to length - 1 do
    data.{i} <- array.(i)
  done

let set_float_array2 t array =
  let dim1 = Array.length array in
  let num_dims = Bigarray.Genarray.num_dims t.data in
  if num_dims <> 2
  then failwith (Printf.sprintf "Improper number of dimension %d <> 2" num_dims);
  let data = Bigarray.array2_of_genarray t.data in
  let dim1' = Bigarray.Array2.dim1 data in
  if dim1 <> dim1'
  then failwith (Printf.sprintf "Dimension 1 mismatch %d <> %d" dim1 dim1');
  for i = 0 to dim1 - 1 do
    let dim2 = Array.length array.(i) in
    let dim2' = Bigarray.Array2.dim2 data in
    if dim2 <> dim2'
    then failwith (Printf.sprintf "Dimension 2 mismatch %d <> %d" dim2 dim2');
    for j = 0 to dim2 - 1 do
      data.{i, j} <- array.(i).(j)
    done
  done

let set_float_array3 t array =
  let dim1 = Array.length array in
  let num_dims = Bigarray.Genarray.num_dims t.data in
  if num_dims <> 3
  then failwith (Printf.sprintf "Improper number of dimension %d <> 3" num_dims);
  let data = Bigarray.array3_of_genarray t.data in
  let dim1' = Bigarray.Array3.dim1 data in
  if dim1 <> dim1'
  then failwith (Printf.sprintf "Dimension 1 mismatch %d <> %d" dim1 dim1');
  for i = 0 to dim1 - 1 do
    let dim2 = Array.length array.(i) in
    let dim2' = Bigarray.Array3.dim2 data in
    if dim2 <> dim2'
    then failwith (Printf.sprintf "Dimension 2 mismatch %d <> %d" dim2 dim2');
    for j = 0 to dim2 - 1 do
      let dim3 = Array.length array.(i).(j) in
      let dim3' = Bigarray.Array3.dim3 data in
      if dim3 <> dim3'
      then failwith (Printf.sprintf "Dimension 3 mismatch %d <> %d" dim3 dim3');
      for k = 0 to dim3 - 1 do
        data.{i, j, k} <- array.(i).(j).(k)
      done
    done
  done

let of_float_array1 array kind =
  let t = create1 kind (Array.length array) in
  set_float_array1 t array;
  t

let of_float_array2 array kind =
  let dim1 = Array.length array in
  let dim2 =
    if dim1 = 0
    then 0
    else Array.length array.(0)
  in
  let t = create2 kind dim1 dim2 in
  set_float_array2 t array;
  t

let of_float_array3 array kind =
  let dim1 = Array.length array in
  let dim2 =
    if dim1 = 0
    then 0
    else Array.length array.(0)
  in
  let dim3 =
    if dim1 = 0 || dim2 = 0
    then 0
    else Array.length array.(0).(0)
  in
  let t = create3 kind dim1 dim2 dim3 in
  set_float_array3 t array;
  t

let to_float_array1 t =
  let ba = Bigarray.array1_of_genarray t.data in
  Array.init (Bigarray.Array1.dim ba) (fun i -> ba.{i})

let to_float_array2 t =
  let ba = Bigarray.array2_of_genarray t.data in
  Array.init (Bigarray.Array2.dim1 ba) (fun i ->
    Array.init (Bigarray.Array2.dim2 ba) (fun j -> ba.{i, j}))

let to_float_array3 t =
  let ba = Bigarray.array3_of_genarray t.data in
  Array.init (Bigarray.Array3.dim1 ba) (fun i ->
    Array.init (Bigarray.Array3.dim2 ba) (fun j ->
      Array.init (Bigarray.Array3.dim3 ba) (fun k -> ba.{i, j, k})))
