(* See http://colah.github.io/posts/2015-08-Understanding-LSTMs
   for a simple description of LSTM networks.

   See http://arxiv.org/pdf/1412.3555v1.pdf for an explanation
   of the difference between LSTM and GRUs

   GRUs only have one hidden state that mixes long term and short term memory.
*)
open Core_kernel.Std
open Tensorflow

let train_size = 1000

let gru ~size_h ~size_x ~size_y x_and_ys =
  let create_vars () = Var.normalf [ size_h+size_x; size_h ] ~stddev:0.1, Var.f [ size_h ] 0. in
  (* The reset parameters *)
  let wr, br = create_vars () in
  (* The mixing variables *)
  let wz, bz = create_vars () in
  (* The contribution of x and the resetted old state *)
  let wH, bH = create_vars () in
  (* Output y from the hidden state *)
  let wy, by = Var.normalf [ size_h; size_y ] ~stddev:0.1, Var.f [ size_y ] 0. in
  let one_gru ~h ~x =
    let open Ops in
    let h_and_x = concat one32 [ h; x ] in
    (* h partly reseted reset h *)
    let rh = sigmoid (h_and_x *^ wr + br) * h in
    let rh_and_x = concat one32 [ rh; x ] in
    (* the new value of h *)
    let nh = tanh (rh_and_x *^ wH + bH) in
    (* How do we mix th new h and the old h *)
    let z = sigmoid (h_and_x *^ wz + bz) in
    (* we mix the old h and the new h *)
    let h = z * nh + (f 1.0 - z) * h in
    let y_bar = Ops.(h *^ wy + by) in
    y_bar, h
  in
  let err =
    let zero = Ops.f ~shape:[ train_size; size_h ] 0. in
    List.fold x_and_ys ~init:([], zero) ~f:(fun (errs, h) (x, y) ->
      let y_bar, h = one_gru ~h ~x in
      let err = Ops.(y_bar - y) in
      err :: errs, h)
    |> fun (errs, __) ->
    let errs =
      match errs with
      | [] -> failwith "Empty input list"
      | [ err ] -> err
      | errs -> Ops.concat Ops.one32 errs
    in
    Ops.square errs |> Ops.reduce_mean
  in
  err, one_gru

let epochs = 400
let size_h = 20
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
  let err, one_gru = gru ~size_h ~size_x:1 ~size_y:1 x_and_ys in
  let gd = Optimizers.adam_minimizer err ~learning_rate:(Ops.f 0.004) in
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
  let y_bar, h_out = one_gru ~h ~x in
  let tensor size = Tensor.create2 Float32 1 size in
  let init = [], tensor 1, tensor size_h in
  let ys, _, h_res =
    List.foldi (List.range 0 500) ~init ~f:(fun idx (acc_y, prev_y, prev_h) _ ->
      if idx < 5 then Tensor.set prev_y [| 0; 0 |] (sin (float idx *. step_size));
      let y_res, h_res =
        Session.run
          ~inputs:Session.Input.[ float x prev_y; float h prev_h ]
          Session.Output.(both (float y_bar) (float h_out))
      in
      let y = Tensor.get y_res [| 0; 0 |] in
      y :: acc_y, y_res, h_res)
  in
  let init = [], tensor 1, h_res in
  let ys', _, _ =
    List.fold (List.range 0 500) ~init ~f:(fun (acc_y, prev_y, prev_h) _ ->
      let y_res, h_res =
        Session.run
          ~inputs:Session.Input.[ float x prev_y; float h prev_h ]
          Session.Output.(both (float y_bar) (float h_out))
      in
      let y = Tensor.get y_res [| 0; 0 |] in
      y :: acc_y, y_res, h_res)
  in
  List.rev ys, List.rev ys'

let () =
  let open Gnuplot in
  let ys, ys' = fit_1d sin in
  let xys = List.mapi ys ~f:(fun i y -> float i *. step_size, y) in
  let xys' = List.mapi ys' ~f:(fun i y -> float i *. step_size, y) in
  let gp = Gp.create () in
  Gp.plot_many gp ~output:(Output.create `Qt)
    [ Series.points_xy xys ~title:"original"
    ; Series.points_xy xys'~title:"replay" ];
  Gp.close gp
