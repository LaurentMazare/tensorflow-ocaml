(* This example uses the tinyshakespeare dataset which can be downloaded at:
   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
*)
open Core_kernel.Std
open Tensorflow

let train_size = 10000
let batch_size = 256
let alphabet_size = 27
let epochs = 100000
let size_c = 100
let gen_len = 220
let sample_size = 20

let lstm ~size_c ~size_x ~size_y x_and_ys =
  let wy, by = Var.normalf [ size_c; size_y ] ~stddev:0.1, Var.f [ size_y ] 0. in
  let lstm1 = Staged.unstage (Cell.lstm ~size_c ~size_x) in
  let lstm2 = Staged.unstage (Cell.lstm ~size_c ~size_x:size_c) in
  let two_lstm ~h1 ~c1 ~h2 ~c2 ~x =
    let `h h1, `c c1 = lstm1 ~h:h1 ~c:c1 ~x in
    let `h h2, `c c2 = lstm2 ~h:h2 ~c:c2 ~x:h1 in
    let y_bar = Ops.(h2 *^ wy + by) |> Ops.softmax in
    y_bar, h1, c1, h2, c2
  in
  let err =
    let zero = Ops.f ~shape:[ batch_size; size_c ] 0. in
    List.fold x_and_ys ~init:([], zero, zero, zero, zero)
      ~f:(fun (errs, h1, c1, h2, c2) (x, y) ->
        let y_bar, h1, c1, h2, c2 = two_lstm ~h1 ~c1 ~h2 ~c2 ~x in
        let err = Ops.(neg (y * log y_bar)) in
        err :: errs, h1, c1, h2, c2)
    |> fun (errs, _, _, _, _) ->
    match errs with
    | [] -> failwith "Empty input list"
    | [ err ] -> err
    | errs -> Ops.concat Ops.one32 errs |> Ops.reduce_mean
  in
  err, two_lstm

let fit_and_evaluate x_data y_data =
  let placeholder_x = Ops.placeholder ~type_:Float [] in
  let placeholder_y = Ops.placeholder ~type_:Float [] in
  let split node =
    Ops.split Ops.one32 node ~num_split:sample_size
    |> List.map ~f:(fun n ->
      Ops.reshape n (Ops.const_int ~type_:Int32 [ batch_size; alphabet_size ]))
  in
  let xs = split placeholder_x in
  let ys = split placeholder_y in
  let x_and_ys = List.zip_exn xs ys in
  let err, two_lstm = lstm ~size_c ~size_x:alphabet_size ~size_y:alphabet_size x_and_ys in
  let gd = Optimizers.adam_minimizer err ~learning_rate:(Ops.f 0.004) in
  let print_sample =
    let h1 = Ops.placeholder [] ~type_:Float in
    let c1 = Ops.placeholder [] ~type_:Float in
    let h2 = Ops.placeholder [] ~type_:Float in
    let c2 = Ops.placeholder [] ~type_:Float in
    let x = Ops.placeholder [] ~type_:Float in
    let y_bar, h1_out, c1_out, h2_out, c2_out = two_lstm ~x ~h1 ~c1 ~h2 ~c2 in
    let y_char = Ops.argMax y_bar Ops.one32 in
    fun () ->
      let tensor size =
        let tensor = Tensor.create2 Float32 1 size in
        Tensor.fill tensor 0.;
        tensor
      in
      let init = [], tensor alphabet_size, tensor size_c, tensor size_c, tensor size_c, tensor size_c in
      let ys, _, _, _, _, _ =
        List.fold (List.range 0 gen_len) ~init ~f:(fun (acc_y, prev_y, prev_h1, prev_c1, prev_h2, prev_c2) _ ->
          let y_char, y_res, h1_res, c1_res, h2_res, c2_res =
            Session.run
              ~inputs:Session.Input.[ float x prev_y; float h1 prev_h1; float c1 prev_c1; float h2 prev_h2; float c2 prev_c2 ]
              Session.Output.(six (int64 y_char) (float y_bar) (float h1_out) (float c1_out) (float h2_out) (float c2_out))
          in
          let y = Tensor.get y_char [| 0 |] |> Int64.to_int_exn in
          y :: acc_y, y_res, h1_res, c1_res, h2_res, c2_res)
      in
      List.rev ys
      |> List.map ~f:(fun c ->
        if 0 <= c && c < 26 then Char.of_int_exn (c + Char.to_int 'a')
        else ' ')
      |> String.of_char_list
      |> printf "%s\n%!"
  in
  for i = 1 to epochs do
    let start_idx = (i * batch_size) % (train_size - batch_size) in
    let x_data = Tensor.sub_left x_data start_idx batch_size in
    let y_data = Tensor.sub_left y_data start_idx batch_size in
    let err =
      Session.run
        ~inputs:Session.Input.[ float placeholder_x x_data; float placeholder_y y_data ]
        ~targets:gd
        (Session.Output.scalar_float err);
    in
    if i % 20 = 0 then begin
      printf "%d %f\n%!" i err;
      print_sample ()
    end
  done;
  print_sample ()

let get_samples ?(filename = "data/input.txt") ~sample_size n =
  let input =
    In_channel.read_lines filename
    |> List.concat_map ~f:(fun str ->
      String.to_list str)
    |> Array.of_list
  in
  let create_vec () = Tensor.create3 Float32 n sample_size alphabet_size in
  let one_hot vec x y c =
    let c =
      let c = Char.lowercase c in
      if 'a' <= c && c <= 'z' then Char.to_int c - Char.to_int 'a'
      else 26
    in
    for z = 0 to alphabet_size - 1 do
      Tensor.set vec [| x; y; z |] (if c = z then 1. else 0.);
    done
  in
  let xs = create_vec () in
  let ys = create_vec () in
  for sample_idx = 0 to n - 1 do
    let offset = Random.int (Array.length input - sample_size - 1) in
    for pos = 0 to sample_size - 1 do
      one_hot xs sample_idx pos input.(offset + pos);
      one_hot ys sample_idx pos input.(offset + pos + 1);
    done;
  done;
  xs, ys

let () =
  Random.init 42;
  let xs, ys = get_samples ~sample_size train_size in
  fit_and_evaluate xs ys
