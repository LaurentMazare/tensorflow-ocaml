open Core_kernel.Std
open Tensorflow

let epochs = 100000
let size_c = 256
let seq_len = 180
let batch_size = 128

type t =
  { err              : [ `float ] Node.t
  ; placeholder_x    : [ `float ] Ops.Placeholder.t
  ; placeholder_y    : [ `float ] Ops.Placeholder.t
  }

let unfolded_lstm ~dim =
  let placeholder_x = Ops.placeholder ~type_:Float [] in
  let placeholder_y = Ops.placeholder ~type_:Float [] in
  let wy, by =
    Var.normalf [ size_c; dim ] ~stddev:0.1, Var.f [ dim ] 0.
  in
  let lstm = Staged.unstage (Cell.lstm ~size_c ~size_x:dim) in
  (* placeholder_x and placeholder_y should be tensor of dimension:
       (batch_size, seq_len, dim)
     Split them on the seq_len dimension to unroll the rnn.
  *)
  let x_and_ys =
    let split node =
      Ops.split Ops.one32 (Ops.Placeholder.to_node node) ~num_split:seq_len
      |> List.map ~f:(fun n ->
        Ops.reshape n (Ops.const_int ~type_:Int32 [ batch_size; dim ]))
    in
    List.zip_exn (split placeholder_x) (split placeholder_y)
  in
  let err, _output_mem =
    let zero = Ops.f 0. ~shape:[ batch_size; size_c ] in
    List.fold x_and_ys
      ~init:([], (`h zero, `c zero))
      ~f:(fun (errs, (`h h, `c c)) (x, y) ->
        let mem = lstm ~x ~h ~c in
        let `h h, `c _ = mem in
        let y_bar = Ops.(h *^ wy + by) |> Ops.softmax in
        let err = Ops.(neg (y * log y_bar)) in
        err :: errs, mem)
  in
  let err =
    match err with
    | [] -> failwith "seq_len is 0"
    | [ err ] -> err
    | errs -> Ops.concat Ops.one32 errs |> Ops.reduce_sum
  in
  { err
  ; placeholder_x
  ; placeholder_y
  }

let all_vars_with_names t =
  Var.get_all_vars t.err
  |> List.mapi ~f:(fun i var -> sprintf "V%d" i, var)

let train filename checkpoint learning_rate =
  let dataset = Text_helper.create filename Float32 in
  let dim = Text_helper.dim dataset in
  let t = unfolded_lstm ~dim in
  let gd = Optimizers.adam_minimizer t.err ~learning_rate:(Ops.f learning_rate) in
  let save_node = Ops.save ~filename:checkpoint (all_vars_with_names t) in
  let run sequence ~train =
    let targets = if train then gd else [] in
    let sum_err, batch_count =
      Sequence.fold sequence ~init:(0., 0) ~f:(fun (acc_err, acc_cnt) (batch_x, batch_y) ->
        let inputs =
          Session.Input.
            [ float t.placeholder_x batch_x
            ; float t.placeholder_y batch_y
            ]
        in
        let sum_err =
          Session.run ~inputs ~targets Session.Output.(scalar_float t.err)
        in
        printf "%c%!" (if train then '.' else 'v');
        acc_err +. sum_err, acc_cnt + 1)
    in
    sum_err /. (float batch_count *. float seq_len *. float batch_size *. log 2.)
  in
  for epoch = 1 to epochs do
    let train_sequence =
      Text_helper.batch_sequence dataset ~seq_len ~batch_size ~len:90_000_000
    in
    let train_bpc = run train_sequence ~train:true in
    let valid_sequence =
      Text_helper.batch_sequence dataset ~seq_len ~batch_size ~pos:90_000_000 ~len:5_000_000
    in
    let valid_bpc = run valid_sequence ~train:false in
    printf "\nEpoch: %d IS: %.4fbpc   OoS: %.4fbpc\n%!" epoch train_bpc valid_bpc;
    Session.run ~targets:[ Node.P save_node ] Session.Output.empty;
  done

let () =
  Random.init 42;
  let open Cmdliner in
  let train_cmd =
    let filename =
      let doc = "Data file to use for training/validation." in
      Arg.(value & opt file "data/text8"
        & info [ "data-file" ] ~docv:"FILE" ~doc)
    in
    let checkpoint =
      let doc = "Checkpoint file to store the current state." in
      Arg.(value & opt string "out.cpkt"
        & info [ "checkpoint" ] ~docv:"FILE" ~doc)
    in
    let learning_rate =
      let doc = "Learning rate for the Adam optimizer" in
      Arg.(value & opt float 0.004
        & info [ "learning_rate" ] ~docv:"FLOAT" ~doc)
    in
    let doc = "Train a RNN on a text dataset" in
    let man =
      [ `S "DESCRIPTION"
      ; `P doc
      ]
    in
    Term.(const train $ filename $ checkpoint $ learning_rate),
    Term.info "train" ~sdocs:"" ~doc ~man
  in
  let default_cmd =
    let doc = "text based RNN benchmarks" in
    Term.(ret (const (`Help (`Pager, None)))),
    Term.info "rnn" ~version:"0" ~sdocs:"" ~doc
  in
  let cmds = [ train_cmd ] in
  match Term.eval_choice default_cmd cmds with
  | `Error _ -> exit 1
  | _ -> exit 0

