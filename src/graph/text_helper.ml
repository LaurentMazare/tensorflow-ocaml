open Base
open Tensorflow_core

type 'a t =
  { content : (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  ; map : int Map.M(Int).t
  ; kind : (float, 'a) Bigarray.kind
  ; dim : int
  }

let create filename kind =
  let file_descr = Unix.openfile filename [ O_RDONLY ] 0 in
  let content =
    Unix.map_file file_descr Int8_unsigned C_layout false [| -1 |]
    |> Bigarray.array1_of_genarray
  in
  let table = Hashtbl.Poly.create () in
  for i = 0 to Bigarray.Array1.dim content - 1 do
    let v = content.{i} in
    if not (Hashtbl.mem table v)
    then Hashtbl.add_exn table ~key:v ~data:(Hashtbl.length table)
  done;
  Unix.close file_descr;
  let map = Hashtbl.to_alist table |> Map.of_alist_exn (module Int) in
  let dim = Map.length map in
  { content; map; kind; dim }

let batch t ~batch_size ~seq_len ~batchX ~batchY ~pos =
  Tensor.fill batchX 0.;
  Tensor.fill batchY 0.;
  for batch_idx = 0 to batch_size - 1 do
    for i = 0 to seq_len do
      let v = Map.find_exn t.map t.content.{pos + (batch_idx * seq_len) + i} in
      if i < seq_len then Tensor.set batchX [| batch_idx; i; v |] 1.;
      if 0 < i then Tensor.set batchY [| batch_idx; i - 1; v |] 1.
    done
  done;
  batchX, batchY

let length t = Bigarray.Array1.dim t.content

let batch_sequence ?pos ?len t ~seq_len ~batch_size =
  let open Sequence.Generator in
  let pos = Option.value pos ~default:0 in
  let pos_plus_len =
    Option.value_map len ~default:(length t) ~f:(fun len -> pos + len)
  in
  let total_batch_size = seq_len * batch_size in
  let batchX = Tensor.create3 t.kind batch_size seq_len t.dim in
  let batchY = Tensor.create3 t.kind batch_size seq_len t.dim in
  let rec loop pos =
    if pos + total_batch_size + 1 >= pos_plus_len
    then return ()
    else
      yield (batch t ~batch_size ~seq_len ~batchX ~batchY ~pos)
      >>= fun () -> loop (pos + total_batch_size)
  in
  loop pos |> run

let map t = t.map
let dim t = t.dim
