open Core_kernel.Std

type t =
  { content : (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  ; map : int Int.Map.t
  }

let create filename =
  let file_descr = Unix.openfile filename [ O_RDONLY ] 0 in
  let content = Bigarray.Array1.map_file file_descr Int8_unsigned C_layout false 0 in
  let table = Int.Table.create () in
  for i = 0 to Bigarray.Array1.dim content - 1 do
    let v = content.{i} in
    if not (Hashtbl.mem table v) then
      Hashtbl.add_exn table ~key:v ~data:(Hashtbl.length table)
  done;
  Unix.close file_descr;
  { content
  ; map = Hashtbl.to_alist table |> Int.Map.of_alist_exn
  }

let onehot t kind ~pos ~len =
  let onehot = Bigarray.Array2.create kind C_layout len (Map.length t.map) in
  Bigarray.Array2.fill onehot 0.;
  for i = 0 to len - 1 do
    let index = Map.find_exn t.map t.content.{pos + i} in
    onehot.{i, index} <- 1.;
  done;
  onehot

let map t = t.map
