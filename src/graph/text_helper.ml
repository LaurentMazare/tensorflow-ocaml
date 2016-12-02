open Core_kernel.Std

let read_file filename =
  let file_descr = Unix.openfile filename [ O_RDONLY ] 0 in
  let array = Bigarray.Array1.map_file file_descr Int8_unsigned C_layout false 0 in
  let table = Int.Table.create () in
  for i = 0 to Bigarray.Array1.dim array - 1 do
    let v = array.{i} in
    let v =
      Hashtbl.find_or_add table v
        ~default:(fun () -> Hashtbl.length table)
    in
    array.{i} <- v
  done;
  Unix.close file_descr;
  array, Hashtbl.to_alist table |> Int.Map.of_alist_exn

let read_file_onehot filename kind =
  let array1, map = read_file filename in
  let dim = Bigarray.Array1.dim array1 in
  let onehot = Bigarray.Array2.create kind C_layout dim (Map.length map) in
  Bigarray.Array2.fill onehot 0.;
  for i = 0 to dim - 1 do
    onehot.{i, array1.{i}} <- 1.;
  done;
  onehot, map
