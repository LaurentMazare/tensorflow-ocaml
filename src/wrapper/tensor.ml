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
  | Float : (Bigarray.float32_elt * [ `float ]) eq
  | Double : (Bigarray.float64_elt * [ `double ]) eq
