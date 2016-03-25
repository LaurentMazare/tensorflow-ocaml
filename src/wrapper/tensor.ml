type ('a, 'b) t = ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t

type p = P : (_, _) t -> p

let print (P tensor) =
  let print_float tensor elt_to_string =
    match Bigarray.Genarray.dims tensor with
    | [| dim |] ->
      for d = 0 to dim - 1 do
        Printf.printf "%d %s\n%!"
          d (Bigarray.Genarray.get tensor [| d |] |> elt_to_string)
      done
    | [| d0; d1 |] ->
      for x = 0 to d0 - 1 do
        Printf.printf "%d " x;
        for y = 0 to d1 - 1 do
          Printf.printf "%s "
            (Bigarray.Genarray.get tensor [| x; y |] |> elt_to_string)
        done;
        Printf.printf "\n%!";
      done
    | otherwise -> Printf.printf "%d dims\n%!" (Array.length otherwise)
  in
  match Bigarray.Genarray.kind tensor with
  | Bigarray.Float32 -> print_float tensor (Printf.sprintf "%f")
  | Bigarray.Float64 -> print_float tensor (Printf.sprintf "%f")
  | _ -> Printf.printf "Unsupported kind"
