open Core_kernel.Std
open Tensorflow

let train_size = 1000

let lstm ~size_c ~size_x ~size_y x_and_ys =
  let create_vars () = Var.normalf [ size_c+size_x; size_c ] ~stddev:0.1, Var.f [ size_c ] 0. in
  let wf, bf = create_vars () in
  let wi, bi = create_vars () in
  let wC, bC = create_vars () in
  let wo, bo = create_vars () in
  let wy, by = Var.normalf [ size_c; size_y ] ~stddev:0.1, Var.f [ size_y ] 0. in
  let one_lstm ~h ~x ~c =
    let open Ops in
    let h_and_x = concat one32 [ h; x ] in
    let c =
      sigmoid (h_and_x *^ wf + bf) * c
      + sigmoid (h_and_x *^ wi + bi) * tanh (sigmoid (h_and_x *^ wC + bC))
    in
    let h = sigmoid (h_and_x *^ wo + bo) * tanh c in
    let y_bar = Ops.(h *^ wy + by) in
    y_bar, h, c
  in
  let err =
    let zero = Ops.f ~shape:[ train_size; size_c ] 0. in
    List.fold x_and_ys ~init:([], zero, zero) ~f:(fun (errs, h, c) (x, y) ->
      let y_bar, h, c = one_lstm ~h ~x ~c in
      let err = Ops.(y_bar - y) in
      err :: errs, h, c)
    |> fun (errs, _, _) ->
    let errs =
      match errs with
      | [] -> failwith "Empty input list"
      | [ err ] -> err
      | errs -> Ops.concat Ops.one32 errs
    in
    Ops.square errs |> Ops.reduce_mean
  in
  err, one_lstm

let epochs = 400
let size_c = 20
let steps = 50
let step_size = 0.1

let fit_1d fn =
  let x_and_ys =
    List.init steps ~f:(fun x ->
      let x = float x *. step_size in
      let xs = List.init train_size ~f:(fun i -> fn (x +. float i)) in
      let ys = List.init train_size ~f:(fun i -> fn (x +. step_size +. float i)) in
      let xs = Ops.const_float ~type_:Float ~shape:[ train_size; 1 ] xs in
      let ys = Ops.const_float ~type_:Float ~shape:[ train_size; 1 ] ys in
      xs, ys
    )
  in
  let err, one_lstm = lstm ~size_c ~size_x:1 ~size_y:1 x_and_ys in
  let gd = Optimizers.adam_minimizer err ~alpha:(Ops.f 0.004) in
  for i = 1 to epochs do
    let err =
      Session.run
        ~inputs:[]
        ~targets:gd
        (Session.Output.scalar_float err);
    in
    printf "%d %f\n%!" i err;
  done;
  let h = Ops.placeholder [] ~type_:Float in
  let x = Ops.placeholder [] ~type_:Float in
  let c = Ops.placeholder [] ~type_:Float in
  let y_bar, h_out, c_out = one_lstm ~h ~x ~c in
  let tensor size =
    Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 1; size |]
  in
  let init = [], tensor 1, tensor size_c, tensor size_c in
  let ys, _, _, _ =
    List.fold (List.range 0 500) ~init ~f:(fun (acc_y, prev_y, prev_h, prev_c) _ ->
      let y_res, h_res, c_res =
        Session.run
          ~inputs:Session.Input.[ float x prev_y; float h prev_h; float c prev_c ]
          Session.Output.(three (float y_bar) (float h_out) (float c_out))
      in
      let y = Bigarray.Genarray.get y_res [| 0; 0 |] in
      y :: acc_y, y_res, h_res, c_res)
  in
  List.rev ys

let get_samples ?(filename = "data/input.txt") ~sample_size n =
  let input =
    In_channel.read_lines filename
    |> List.concat_map ~f:(fun str ->
      String.to_list str)
    |> Array.of_list
  in
  let create_vec () =
    Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| n; sample_size; 256 |]
  in
  let one_hot vec x y c =
    let c = Char.to_int c in
    for z = 0 to 255 do
      Bigarray.Genarray.set vec [| x; y; z |] (if c = z then 1. else 0.)
    done
  in
  let xs = create_vec () in
  let ys = create_vec () in
  for sample_idx = 1 to n do
    let offset = Random.int (Array.length input - sample_size - 1) in
    for pos = 0 to sample_size - 1 do
      one_hot xs sample_idx pos input.(offset + pos);
      one_hot ys sample_idx pos input.(offset + pos + 1);
    done;
  done;
  xs, ys

let () =
  Random.init 42;
  ()
