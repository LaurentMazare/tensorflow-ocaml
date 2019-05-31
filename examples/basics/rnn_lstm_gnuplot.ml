(* See http://colah.github.io/posts/2015-08-Understanding-LSTMs
   for a simple description of LSTM networks.
*)
open Base
open Tensorflow

let train_size = 1000

let lstm ~size_c ~size_x ~size_y x_and_ys =
  let wy, by = Var.normalf [ size_c; size_y ] ~stddev:0.1, Var.f [ size_y ] 0. in
  let lstm = Staged.unstage (Cell.lstm ~size_c ~size_x) in
  let one_lstm ~h ~x ~c =
    let `h h, `c c = lstm ~h ~x ~c in
    let y_bar = Ops.((h *^ wy) + by) in
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
        xs, ys)
  in
  let err, one_lstm = lstm ~size_c ~size_x:1 ~size_y:1 x_and_ys in
  let gd = Optimizers.adam_minimizer err ~learning_rate:(Ops.f 0.004) in
  for i = 1 to epochs do
    let err = Session.run ~inputs:[] ~targets:gd (Session.Output.scalar_float err) in
    printf "%d %f\n%!" i err
  done;
  let h = Ops.placeholder [] ~type_:Float in
  let x = Ops.placeholder [] ~type_:Float in
  let c = Ops.placeholder [] ~type_:Float in
  let y_bar, h_out, c_out =
    one_lstm
      ~h:(Ops.Placeholder.to_node h)
      ~x:(Ops.Placeholder.to_node x)
      ~c:(Ops.Placeholder.to_node c)
  in
  let tensor size = Tensor.create2 Float32 1 size in
  let init = [], tensor 1, tensor size_c, tensor size_c in
  let ys, _, _, _ =
    List.fold (List.range 0 500) ~init ~f:(fun (acc_y, prev_y, prev_h, prev_c) _ ->
        let y_res, h_res, c_res =
          Session.run
            ~inputs:Session.Input.[ float x prev_y; float h prev_h; float c prev_c ]
            Session.Output.(three (float y_bar) (float h_out) (float c_out))
        in
        let y = Tensor.get y_res [| 0; 0 |] in
        y :: acc_y, y_res, h_res, c_res)
  in
  List.rev ys

let () =
  let open Gnuplot in
  let ys = fit_1d sin in
  let xys = List.mapi ys ~f:(fun i y -> float i *. step_size, y) in
  let gp = Gp.create () in
  Gp.plot_many gp ~output:(Output.create `Qt) [ Series.points_xy xys ];
  Gp.close gp
