open Base
open Float.O_dot
open Tensorflow
open Tensorflow_fnn

let float = Float.of_int
let epochs = 100000
let size_c = 256
let seq_len = 180
let batch_size = 128

let all_vars_with_names node =
  Var.get_all_vars [ Node.P node ]
  |> List.mapi ~f:(fun i var -> Printf.sprintf "V%d" i, var)

let train filename learning_rate =
  let dataset = Text_helper.create filename Float32 in
  let dim = Text_helper.dim dataset in
  let placeholder_x = Ops.placeholder ~type_:Float [ -1; seq_len ] in
  let placeholder_y = Ops.placeholder ~type_:Float [ -1 ] in
  let cross_entropy =
    let wy, by = Var.normalf [ size_c; dim ] ~stddev:0.1, Var.f [ dim ] 0. in
    let lstm = Staged.unstage (Cell.lstm ~size_c ~size_x:dim) in
    let zero = Ops.f 0. ~shape:[ batch_size; size_c ] in
    let y_hats =
      Cell.Unfold.unfold
        ~xs:(Ops.Placeholder.to_node placeholder_x)
        ~seq_len
        ~dim
        ~init:(`h zero, `c zero)
        ~f:(fun ~x ~mem:(`h h, `c c) ->
          let mem = lstm ~h ~x ~c in
          let `h h, `c _ = mem in
          let y_bar = Ops.((h *^ wy) + by) |> Ops.softmax in
          y_bar, `mem mem)
    in
    Ops.cross_entropy ~ys:(Ops.Placeholder.to_node placeholder_y) ~y_hats `sum
  in
  let gd =
    Optimizers.adam_minimizer cross_entropy ~learning_rate:(Ops.f learning_rate)
  in
  let save_node = Ops.save ~filename:"out.ckpt" (all_vars_with_names cross_entropy) in
  let run sequence ~train =
    let targets = if train then gd else [] in
    let sum_err, batch_count =
      Sequence.fold
        sequence
        ~init:(0., 0)
        ~f:(fun (acc_err, acc_cnt) (batch_x, batch_y) ->
          let inputs =
            Session.Input.[ float placeholder_x batch_x; float placeholder_y batch_y ]
          in
          let sum_err =
            Session.run ~inputs ~targets Session.Output.(scalar_float cross_entropy)
          in
          acc_err +. sum_err, acc_cnt + 1)
    in
    sum_err /. (float batch_count *. float seq_len *. float batch_size *. Float.log 2.)
  in
  for epoch = 1 to epochs do
    let train_sequence =
      Text_helper.batch_sequence dataset ~seq_len ~batch_size ~len:90_000_000
    in
    let train_bpc = run train_sequence ~train:true in
    let valid_sequence =
      Text_helper.batch_sequence
        dataset
        ~seq_len
        ~batch_size
        ~pos:90_000_000
        ~len:5_000_000
    in
    let valid_bpc = run valid_sequence ~train:false in
    Stdio.printf "\nEpoch: %d IS: %.4fbpc   OoS: %.4fbpc\n%!" epoch train_bpc valid_bpc;
    Session.run ~targets:[ Node.P save_node ] Session.Output.empty
  done

let () =
  Random.init 42;
  let open Cmdliner in
  let train_cmd =
    let filename =
      let doc = "Data file to use for training/validation." in
      Arg.(value & opt file "data/text8" & info [ "data-file" ] ~docv:"FILE" ~doc)
    in
    let learning_rate =
      let doc = "Learning rate for the Adam optimizer" in
      Arg.(value & opt float 0.004 & info [ "learning_rate" ] ~docv:"FLOAT" ~doc)
    in
    let doc = "Train a RNN on a text dataset" in
    let man = [ `S "DESCRIPTION"; `P doc ] in
    Term.(const train $ filename $ learning_rate), Term.info "train" ~sdocs:"" ~doc ~man
  in
  let default_cmd =
    let doc = "text based RNN benchmarks" in
    ( Term.(ret (const (`Help (`Pager, None))))
    , Term.info "rnn" ~version:"0" ~sdocs:"" ~doc )
  in
  let cmds = [ train_cmd ] in
  match Term.eval_choice default_cmd cmds with
  | `Error _ -> Caml.exit 1
  | _ -> Caml.exit 0
