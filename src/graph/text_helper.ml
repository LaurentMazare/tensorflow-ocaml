open Core_kernel.Std

type 'a t =
  { content : (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  ; map : int Int.Map.t
  ; kind : (float, 'a) Bigarray.kind
  ; dim : int
  }

let create filename kind =
  let file_descr = Unix.openfile filename [ O_RDONLY ] 0 in
  let content = Bigarray.Array1.map_file file_descr Int8_unsigned C_layout false 0 in
  let table = Int.Table.create () in
  for i = 0 to Bigarray.Array1.dim content - 1 do
    let v = content.{i} in
    if not (Hashtbl.mem table v) then
      Hashtbl.add_exn table ~key:v ~data:(Hashtbl.length table)
  done;
  Unix.close file_descr;
  let map = Hashtbl.to_alist table |> Int.Map.of_alist_exn in
  let dim = Map.length map in
  { content
  ; map
  ; kind
  ; dim
  }

let batch t ~batch_size ~seq_len ~batchX ~batchY ~pos =
  Bigarray.Array3.fill batchX 0.;
  Bigarray.Array3.fill batchY 0.;
  for batch_idx = 0 to batch_size - 1 do
    for i = 0 to seq_len do
      let v = Map.find_exn t.map t.content.{pos + batch_idx * seq_len + i} in
      if i < seq_len
      then batchX.{batch_idx, i, v} <- 1.;
      if 0 < i
      then batchY.{batch_idx, i-1, v} <- 1.;
    done;
  done;
  batchX, batchY

let batch_sequence t ~pos ~len ~seq_len ~batch_size =
  let open Sequence.Generator in
  let pos_plus_len = pos + len in
  let total_batch_size = seq_len * batch_size in
  let batchX = Bigarray.Array3.create t.kind C_layout batch_size seq_len t.dim in
  let batchY = Bigarray.Array3.create t.kind C_layout batch_size seq_len t.dim in
  let rec loop pos =
    if pos + total_batch_size + 1 >= pos_plus_len
    then return ()
    else
      yield (batch t ~batch_size ~seq_len ~batchX ~batchY ~pos)
      >>= fun () ->
      loop (pos + total_batch_size)
  in
  loop pos |> run

let map t = t.map

let length t = Bigarray.Array1.dim t.content

let dim t = t.dim
