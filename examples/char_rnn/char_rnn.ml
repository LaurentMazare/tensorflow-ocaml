(* This example uses the tinyshakespeare dataset which can be downloaded at:
   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

   It has been heavily inspired by https://github.com/karpathy/char-rnn
*)
open Base
open Float.O_dot
open Tensorflow_core
open Tensorflow
open Tensorflow_fnn

let float = Float.of_int
let epochs = 100000
let size_c = 256
let seq_len = 180
let batch_size = 128

type t =
  { train_err : [ `float ] Node.t
  ; train_placeholder_x : [ `float ] Ops.Placeholder.t
  ; train_placeholder_y : [ `float ] Ops.Placeholder.t
  ; sample_output : [ `float ] Node.t
  ; sample_output_mem : [ `float ] Node.t
  ; sample_placeholder_mem : [ `float ] Ops.Placeholder.t
  ; sample_placeholder_x : [ `float ] Ops.Placeholder.t
  ; initial_memory : (float, Bigarray.float32_elt) Tensor.t
  }

let tensor_zero size =
  let tensor = Tensor.create2 Float32 1 size in
  Tensor.fill tensor 0.;
  tensor

let rnn ~size_c ~dim =
  let train_placeholder_x = Ops.placeholder ~type_:Float [ -1; seq_len ] in
  let train_placeholder_y = Ops.placeholder ~type_:Float [ -1 ] in
  let sample_placeholder_mem = Ops.placeholder ~type_:Float [ 1; 4 * size_c ] in
  let sample_placeholder_x = Ops.placeholder ~type_:Float [ 1; dim ] in
  (* Two LSTM specific code. *)
  let wy, by = Var.normalf [ size_c; dim ] ~stddev:0.1, Var.f [ dim ] 0. in
  let lstm1 = Staged.unstage (Cell.lstm ~size_c ~size_x:dim) in
  let lstm2 = Staged.unstage (Cell.lstm ~size_c ~size_x:size_c) in
  let two_lstm ~x ~mem:(h1, c1, h2, c2) =
    let `h h1, `c c1 = lstm1 ~h:h1 ~c:c1 ~x in
    let `h h2, `c c2 = lstm2 ~h:h2 ~c:c2 ~x:h1 in
    let y_bar = Ops.((h2 *^ wy) + by) |> Ops.softmax in
    y_bar, `mem (h1, c1, h2, c2)
  in
  let y_hats =
    let zero = Ops.f 0. ~shape:[ batch_size; size_c ] in
    Cell.Unfold.unfold
      ~xs:(Ops.Placeholder.to_node train_placeholder_x)
      ~seq_len
      ~dim
      ~init:(zero, zero, zero, zero)
      ~f:two_lstm
  in
  let mem_split mem = Ops.split4 Ops.one32 (Ops.Placeholder.to_node mem) in
  let sample_output, `mem sample_output_mem =
    two_lstm
      ~mem:(mem_split sample_placeholder_mem)
      ~x:(Ops.Placeholder.to_node sample_placeholder_x)
  in
  let mem_concat (h1, c1, h2, c2) = Ops.concat Ops.one32 [ h1; c1; h2; c2 ] in
  { train_err =
      Ops.cross_entropy ~ys:(Ops.Placeholder.to_node train_placeholder_y) ~y_hats `sum
  ; train_placeholder_x
  ; train_placeholder_y
  ; sample_output
  ; sample_output_mem = mem_concat sample_output_mem
  ; sample_placeholder_mem
  ; sample_placeholder_x
  ; initial_memory = tensor_zero (4 * size_c)
  }

let all_vars_with_names t =
  Var.get_all_vars [ Node.P t.sample_output ]
  |> List.mapi ~f:(fun i var -> Printf.sprintf "V%d" i, var)

let fit_and_evaluate ~dataset ~learning_rate ~checkpoint =
  let t = rnn ~size_c ~dim:(Text_helper.dim dataset) in
  let gd = Optimizers.adam_minimizer t.train_err ~learning_rate:(Ops.f learning_rate) in
  let save_node = Ops.save ~filename:checkpoint (all_vars_with_names t) in
  for epoch = 1 to epochs do
    let train_sequence = Text_helper.batch_sequence dataset ~seq_len ~batch_size in
    let sum_err, batch_count =
      Sequence.fold
        train_sequence
        ~init:(0., 0)
        ~f:(fun (acc_err, acc_cnt) (batch_x, batch_y) ->
          let sum_err =
            Session.run
              Session.Output.(scalar_float t.train_err)
              ~inputs:
                Session.Input.
                  [ float t.train_placeholder_x batch_x
                  ; float t.train_placeholder_y batch_y
                  ]
              ~targets:gd
          in
          acc_err +. sum_err, acc_cnt + 1)
    in
    let bpc =
      sum_err /. (float batch_count *. float seq_len *. float batch_size *. Float.log 2.)
    in
    Stdio.printf "Epoch: %d   %.4fbpc\n%!" epoch bpc;
    Session.run ~targets:[ Node.P save_node ] Session.Output.empty
  done

let train filename checkpoint learning_rate =
  let dataset = Text_helper.create filename Float32 in
  fit_and_evaluate ~dataset ~checkpoint ~learning_rate

let sample filename checkpoint gen_size temperature seed =
  let dataset = Text_helper.create filename Float32 in
  let dim = Text_helper.dim dataset in
  let seed_length = String.length seed in
  let index_by_char = Text_helper.map dataset in
  let t = rnn ~size_c ~dim in
  let load_and_assign_nodes =
    let checkpoint = Ops.const_string0 checkpoint in
    List.map (all_vars_with_names t) ~f:(fun (var_name, Node.P var) ->
        Ops.restore ~type_:(Node.output_type var) checkpoint (Ops.const_string0 var_name)
        |> Ops.assign var
        |> fun node -> Node.P node)
  in
  Session.run ~targets:load_and_assign_nodes Session.Output.empty;
  let prev_y = tensor_zero dim in
  Tensor.set prev_y [| 0; 0 |] 1.;
  let init = [], prev_y, t.initial_memory in
  let ys, _, _ =
    List.fold (List.range 0 gen_size) ~init ~f:(fun (acc_y, prev_y, prev_mem_data) i ->
        let y_res, mem_data =
          Session.run
            Session.Output.(both (float t.sample_output) (float t.sample_output_mem))
            ~inputs:
              Session.Input.
                [ float t.sample_placeholder_x prev_y
                ; float t.sample_placeholder_mem prev_mem_data
                ]
        in
        let y =
          if i < seed_length
          then (
            match Map.find index_by_char (seed.[i] |> Char.to_int) with
            | None ->
              Printf.failwithf
                "Cannot find seed character '%c' in the train file"
                seed.[i]
                ()
            | Some y -> y)
          else (
            let dist =
              Array.init dim ~f:(fun i ->
                  Tensor.get y_res [| 0; i |] **. (1. /. temperature))
            in
            let p = Random.float (Array.reduce_exn dist ~f:( +. )) in
            let acc = ref 0. in
            let y = ref 0 in
            for i = 0 to dim - 1 do
              if Float.( <= ) !acc p then y := i;
              acc := !acc +. dist.(i)
            done;
            !y)
        in
        let y_res = tensor_zero dim in
        Tensor.set y_res [| 0; y |] 1.;
        y :: acc_y, y_res, mem_data)
  in
  let inv_map = Array.create ~len:dim '?' in
  Map.iteri index_by_char ~f:(fun ~key ~data -> inv_map.(data) <- Char.of_int_exn key);
  List.rev_map ys ~f:(fun i -> inv_map.(i))
  |> String.of_char_list
  |> Stdio.printf "%s\n\n%!"

let () =
  Random.init 42;
  let open Cmdliner in
  let sample_cmd =
    let train_filename =
      let doc = "Data file to use for training." in
      Arg.(value & opt file "data/input.txt" & info [ "train-file" ] ~docv:"FILE" ~doc)
    in
    let checkpoint =
      let doc = "Checkpoint file to store the current state." in
      Arg.(value & opt string "out.ckpt" & info [ "checkpoint" ] ~docv:"FILE" ~doc)
    in
    let length =
      let doc = "Length of the text to generate." in
      Arg.(value & opt int 1024 & info [ "length" ] ~docv:"INT" ~doc)
    in
    let temperature =
      let doc = "Temperature at which the text gets generated." in
      Arg.(value & opt float 0.5 & info [ "temperature" ] ~docv:"FLOAT" ~doc)
    in
    let seed =
      let doc = "Seed used to start generation." in
      Arg.(value & opt string "" & info [ "seed" ] ~docv:"STR" ~doc)
    in
    let doc = "Sample a text using a trained state" in
    let man = [ `S "DESCRIPTION"; `P "Sample a text using a trained state" ] in
    ( Term.(const sample $ train_filename $ checkpoint $ length $ temperature $ seed)
    , Term.info "sample" ~sdocs:"" ~doc ~man )
  in
  let train_cmd =
    let train_filename =
      let doc = "Data file to use for training." in
      Arg.(value & opt file "data/input.txt" & info [ "train-file" ] ~docv:"FILE" ~doc)
    in
    let checkpoint =
      let doc = "Checkpoint file to store the current state." in
      Arg.(value & opt string "out.ckpt" & info [ "checkpoint" ] ~docv:"FILE" ~doc)
    in
    let learning_rate =
      let doc = "Learning rate for the Adam optimizer" in
      Arg.(value & opt float 0.004 & info [ "learning_rate" ] ~docv:"FLOAT" ~doc)
    in
    let doc = "Train a char based RNN on a given file" in
    let man = [ `S "DESCRIPTION"; `P "Train a char based RNN on a given file" ] in
    ( Term.(const train $ train_filename $ checkpoint $ learning_rate)
    , Term.info "train" ~sdocs:"" ~doc ~man )
  in
  let default_cmd =
    let doc = "char based RNN" in
    ( Term.(ret (const (`Help (`Pager, None))))
    , Term.info "char_rnn" ~version:"0" ~sdocs:"" ~doc )
  in
  let cmds = [ train_cmd; sample_cmd ] in
  match Term.eval_choice default_cmd cmds with
  | `Error _ -> Caml.exit 1
  | _ -> Caml.exit 0
