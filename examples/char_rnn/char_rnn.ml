(* This example uses the tinyshakespeare dataset which can be downloaded at:
   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

   It has been heavily inspired by https://github.com/karpathy/char-rnn
*)
open Core_kernel.Std
open Tensorflow

let epochs = 100000
let size_c = 256
let gen_len = 500
let sample_size = 25
let temperature = 0.5

type t =
  { train_err              : [ `float ] Node.t
  ; train_output_mem       : [ `float ] Node.t
  ; train_placeholder_mem  : [ `float ] Node.t
  ; train_placeholder_x    : [ `float ] Node.t
  ; train_placeholder_y    : [ `float ] Node.t
  ; sample_output          : [ `float ] Node.t
  ; sample_output_mem      : [ `float ] Node.t
  ; sample_placeholder_mem : [ `float ] Node.t
  ; sample_placeholder_x   : [ `float ] Node.t
  }

let rnn ~size_c ~sample_size ~alphabet_size =
  let train_placeholder_mem  = Ops.placeholder ~type_:Float [] in
  let train_placeholder_x    = Ops.placeholder ~type_:Float [] in
  let train_placeholder_y    = Ops.placeholder ~type_:Float [] in
  let sample_placeholder_mem = Ops.placeholder ~type_:Float [] in
  let sample_placeholder_x   = Ops.placeholder ~type_:Float [] in
  (* Two LSTM specific code. *)
  let wy, by =
    Var.normalf [ size_c; alphabet_size ] ~stddev:0.1, Var.f [ alphabet_size ] 0.
  in
  let lstm1 = Staged.unstage (Cell.lstm ~size_c ~size_x:alphabet_size) in
  let lstm2 = Staged.unstage (Cell.lstm ~size_c ~size_x:size_c) in
  let two_lstm ~mem:(h1, c1, h2, c2) ~x =
    let `h h1, `c c1 = lstm1 ~h:h1 ~c:c1 ~x in
    let `h h2, `c c2 = lstm2 ~h:h2 ~c:c2 ~x:h1 in
    let y_bar = Ops.(h2 *^ wy + by) |> Ops.softmax in
    y_bar, (h1, c1, h2, c2)
  in
  let split node =
    Ops.split Ops.zero32 node ~num_split:sample_size
    |> List.map ~f:(fun n ->
      Ops.reshape n (Ops.const_int ~type_:Int32 [ 1; alphabet_size ]))
  in
  let x_and_ys =
    List.zip_exn (split train_placeholder_x) (split train_placeholder_y)
  in
  let mem_split mem =
    match Ops.split Ops.one32 mem ~num_split:4 with
    | [ h1; c1; h2; c2 ] -> h1, c1, h2, c2
    | _ -> assert false
  in
  let train_err, train_output_mem =
    List.fold x_and_ys
      ~init:([], mem_split train_placeholder_mem)
      ~f:(fun (errs, mem) (x, y) ->
        let y_bar, mem = two_lstm ~x ~mem in
        let err = Ops.(neg (y * log y_bar)) in
        err :: errs, mem)
  in
  let train_err =
    match train_err with
    | [] -> failwith "sample_size is 0"
    | [ err ] -> err
    | errs -> Ops.concat Ops.one32 errs |> Ops.reduce_sum
  in
  let mem_concat (h1, c1, h2, c2) = Ops.concat Ops.one32 [ h1; c1; h2; c2 ] in
  let sample_output, sample_output_mem =
    two_lstm ~mem:(mem_split sample_placeholder_mem) ~x:sample_placeholder_x
  in
  { train_err
  ; train_output_mem = mem_concat train_output_mem
  ; train_placeholder_mem
  ; train_placeholder_x
  ; train_placeholder_y
  ; sample_output
  ; sample_output_mem = mem_concat sample_output_mem
  ; sample_placeholder_mem
  ; sample_placeholder_x
  }

let tensor_zero size =
  let tensor = Tensor.create2 Float32 1 size in
  Tensor.fill tensor 0.;
  tensor

let print_sample t all_chars =
  let alphabet_size = Array.length all_chars in
  Staged.stage (fun ~prev_y ~prev_mem_data ->
    let init = [], prev_y, prev_mem_data in
    let ys, _, _ =
      List.fold (List.range 0 gen_len) ~init ~f:(fun (acc_y, prev_y, prev_mem_data) _ ->
        let inputs =
          Session.Input.
            [ float t.sample_placeholder_x prev_y
            ; float t.sample_placeholder_mem prev_mem_data
            ]
        in
        let y_res, mem_data =
          Session.run ~inputs
            Session.Output.(both (float t.sample_output) (float t.sample_output_mem))
        in
        let dist =
          Array.init alphabet_size ~f:(fun i ->
            (Tensor.get y_res [| 0; i |]) ** (1. /. temperature))
        in
        let p = Random.float (Array.reduce_exn dist ~f:(+.)) in
        let acc = ref 0. in
        let y = ref 0 in
        for i = 0 to alphabet_size - 1 do
          if !acc <= p then y := i;
          acc := !acc +. dist.(i)
        done;
        let y_res = tensor_zero alphabet_size in
        Tensor.set y_res [| 0; !y |] 1.;
        !y :: acc_y, y_res, mem_data)
    in
    List.rev ys
    |> List.map ~f:(fun i -> all_chars.(i))
    |> String.of_char_list
    |> printf "%s\n\n%!")

let fit_and_evaluate data all_chars ~checkpoint =
  ignore checkpoint;
  let alphabet_size = Array.length all_chars in
  let input_size = (Tensor.dims data).(0) in
  let t = rnn ~size_c ~sample_size ~alphabet_size in
  let gd = Optimizers.adam_minimizer t.train_err ~learning_rate:(Ops.f 0.004) in
  let print_sample = Staged.unstage (print_sample t all_chars) in
  let zero = tensor_zero (4 * size_c) in
  List.fold (List.range 1 epochs)
    ~init:(log (float alphabet_size) *. float sample_size, zero)
    ~f:(fun (smooth_error, prev_mem_data) i ->
      let start_idx = (i * sample_size) % (input_size - sample_size - 1) in
      let x_data = Tensor.sub_left data start_idx sample_size in
      let y_data = Tensor.sub_left data (start_idx + 1) sample_size in
      let inputs =
        Session.Input.
          [ float t.train_placeholder_x x_data
          ; float t.train_placeholder_y y_data
          ; float t.train_placeholder_mem prev_mem_data
          ]
      in
      let err, mem_data =
        Session.run
          ~inputs
          ~targets:gd
          Session.Output.(both (scalar_float t.train_err) (float t.train_output_mem))
      in
      let smooth_error = 0.999 *. smooth_error +. 0.001 *. err in
      if i % 500 = 0 then begin
        printf "Epoch: %d %f\n%!" i smooth_error;
        let prev_y = tensor_zero alphabet_size in
        Tensor.set prev_y [| 0; 0 |] 1.;
        print_sample ~prev_y ~prev_mem_data:mem_data
      end;
      smooth_error, mem_data)
  |> ignore

let read_file filename =
  let input = In_channel.read_all filename |> String.to_array in
  let all_chars = Char.Set.of_array input |> Set.to_array in
  let index_by_char =
    Array.mapi all_chars ~f:(fun i c -> c, i)
    |> Array.to_list
    |> Char.Table.of_alist_exn
  in
  let input_length = Array.length input in
  let alphabet_size = Array.length all_chars in
  let data = Tensor.create2 Float32 input_length alphabet_size in
  Tensor.fill data 0.;
  for x = 0 to input_length - 1 do
    let c = Hashtbl.find_exn index_by_char input.(x) in
    Tensor.set data [| x; c |] 1.
  done;
  data, all_chars

let train filename checkpoint =
  let data, all_chars = read_file filename in
  fit_and_evaluate data all_chars ~checkpoint

let () =
  Random.init 42;
  let open Cmdliner in
  let train_cmd =
    let train_filename =
      let doc = "Data file to use for training." in
      Arg.(value & opt file "data/input.txt"
        & info [ "train-file" ] ~docv:"FILE" ~doc)
    in
    let checkpoint =
      let doc = "Checkpoint file to store the current state." in
      Arg.(value & opt string "out.cpkt"
        & info [ "checkpoint" ] ~docv:"FILE" ~doc)
    in
    let doc = "Train a char based RNN on a given file" in
    let man =
      [ `S "DESCRIPTION"
      ; `P "Train a char based RNN on a given file"
      ]
    in
    Term.(const train
      $ train_filename
      $ checkpoint),
    Term.info "train" ~sdocs:"" ~doc ~man
  in
  let default_cmd =
    let doc = "char based RNN" in
    Term.(ret (const (`Help (`Pager, None)))),
    Term.info "char_rnn" ~version:"0" ~sdocs:"" ~doc
  in
  let cmds = [ train_cmd ] in
  match Term.eval_choice default_cmd cmds with
  | `Error _ -> exit 1
  | _ -> exit 0
