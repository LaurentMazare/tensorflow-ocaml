(* This example uses the tinyshakespeare dataset which can be downloaded at:
   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
*)
open Core_kernel.Std
open Tensorflow

let epochs = 100000
let size_c = 100
let gen_len = 220
let sample_size = 25

let lstm ~size_c ~size_x ~size_y ~prev_mem x_and_ys =
  let wy, by = Var.normalf [ size_c; size_y ] ~stddev:0.1, Var.f [ size_y ] 0. in
  let lstm1 = Staged.unstage (Cell.lstm ~size_c ~size_x) in
  let lstm2 = Staged.unstage (Cell.lstm ~size_c ~size_x:size_c) in
  let two_lstm ~mem ~x =
    let h1, c1, h2, c2 = mem in
    let `h h1, `c c1 = lstm1 ~h:h1 ~c:c1 ~x in
    let `h h2, `c c2 = lstm2 ~h:h2 ~c:c2 ~x:h1 in
    let y_bar = Ops.(h2 *^ wy + by) |> Ops.softmax in
    y_bar, (h1, c1, h2, c2)
  in
  List.fold x_and_ys ~init:([], prev_mem) ~f:(fun (errs, mem) (x, y) ->
    let y_bar, mem = two_lstm ~mem ~x in
    let err = Ops.(neg (y * log y_bar)) in
    err :: errs, mem)
  |> fun (errs, mem) ->
  match errs with
  | [] -> failwith "Empty input list"
  | [ err ] -> err, mem, two_lstm
  | errs ->
    let err = Ops.concat Ops.one32 errs |> Ops.reduce_sum in
    err, mem, two_lstm

let tensor_zero size =
  let tensor = Tensor.create2 Float32 1 size in
  Tensor.fill tensor 0.;
  tensor

let print_sample two_lstm all_chars =
  let alphabet_size = Array.length all_chars in
  let placeholder_h1 = Ops.placeholder ~type_:Float [] in
  let placeholder_c1 = Ops.placeholder ~type_:Float [] in
  let placeholder_h2 = Ops.placeholder ~type_:Float [] in
  let placeholder_c2 = Ops.placeholder ~type_:Float [] in
  let prev_mem = placeholder_h1, placeholder_c1, placeholder_h2, placeholder_c2 in
  let x = Ops.placeholder [] ~type_:Float in
  let y_bar, sample_mem = two_lstm ~mem:prev_mem ~x in
  Staged.stage (fun ~prev_y ~prev_mem_data ->
    let init = [], prev_y, prev_mem_data in
    let ys, _, _ =
      List.fold (List.range 0 gen_len) ~init ~f:(fun (acc_y, prev_y, prev_mem_data) _ ->
        let inputs =
          let prev_h1, prev_c1, prev_h2, prev_c2 = prev_mem_data in
          Session.Input.
            [ float x prev_y
            ; float placeholder_h1 prev_h1
            ; float placeholder_c1 prev_c1
            ; float placeholder_h2 prev_h2
            ; float placeholder_c2 prev_c2
            ]
        in
        let h1_out, c1_out, h2_out, c2_out = sample_mem in
        let y_res, h1_res, c1_res, h2_res, c2_res =
          Session.run ~inputs
            Session.Output.(five (float y_bar) (float h1_out) (float c1_out) (float h2_out) (float c2_out))
        in
        let p = Random.float 1. in
        let acc = ref 0. in
        let y = ref 0 in
        for i = 0 to alphabet_size - 1 do
          if !acc <= p then y := i;
          acc := !acc +. Tensor.get y_res [| 0; i |]
        done;
        let y_res = tensor_zero alphabet_size in
        Tensor.set y_res [| 0; !y |] 1.;
        !y :: acc_y, y_res, (h1_res, c1_res, h2_res, c2_res))
    in
    List.rev ys
    |> List.map ~f:(fun i -> all_chars.(i))
    |> String.of_char_list
    |> printf "%s\n%!")

let fit_and_evaluate data all_chars =
  let alphabet_size = Array.length all_chars in
  let input_size = (Tensor.dims data).(0) in
  let placeholder_x = Ops.placeholder ~type_:Float [] in
  let placeholder_y = Ops.placeholder ~type_:Float [] in
  let split node =
    Ops.split Ops.zero32 node ~num_split:sample_size
    |> List.map ~f:(fun n ->
      Ops.reshape n (Ops.const_int ~type_:Int32 [ 1; alphabet_size ]))
  in
  let xs = split placeholder_x in
  let ys = split placeholder_y in
  let x_and_ys = List.zip_exn xs ys in
  let placeholder_h1 = Ops.placeholder ~type_:Float [] in
  let placeholder_c1 = Ops.placeholder ~type_:Float [] in
  let placeholder_h2 = Ops.placeholder ~type_:Float [] in
  let placeholder_c2 = Ops.placeholder ~type_:Float [] in
  let prev_mem = placeholder_h1, placeholder_c1, placeholder_h2, placeholder_c2 in
  let err, train_mem, two_lstm =
    lstm ~size_c ~size_x:alphabet_size ~size_y:alphabet_size ~prev_mem x_and_ys
  in
  let gd = Optimizers.adam_minimizer err ~learning_rate:(Ops.f 0.004) in
  let print_sample = Staged.unstage (print_sample two_lstm all_chars) in
  let zero = tensor_zero size_c in
  List.fold (List.range 1 epochs)
    ~init:(log (float alphabet_size) *. float sample_size, (zero, zero, zero, zero))
    ~f:(fun (smooth_error, prev_mem_data) i ->
      let start_idx = (i * sample_size) % (input_size - sample_size - 1) in
      let x_data = Tensor.sub_left data start_idx sample_size in
      let y_data = Tensor.sub_left data (start_idx + 1) sample_size in
      let inputs =
        let prev_h1, prev_c1, prev_h2, prev_c2 = prev_mem_data in
        Session.Input.
          [ float placeholder_x  x_data
          ; float placeholder_y  y_data
          ; float placeholder_h1 prev_h1
          ; float placeholder_c1 prev_c1
          ; float placeholder_h2 prev_h2
          ; float placeholder_c2 prev_c2
          ]
      in
      let h1_out, c1_out, h2_out, c2_out = train_mem in
      let err, h1_res, c1_res, h2_res, c2_res =
        Session.run
          ~inputs
          ~targets:gd
          Session.Output.(five
            (scalar_float err) (float h1_out) (float c1_out) (float h2_out) (float c2_out))
      in
      let mem_data = h1_res, c1_res, h2_res, c2_res in
      let smooth_error = 0.999 *. smooth_error +. 0.001 *. err in
      if i % 50 = 0 then begin
        printf "\n%d %f\n%!" i smooth_error;
        let prev_y = tensor_zero alphabet_size in
        Tensor.set prev_y [| 0; 0 |] 1.;
        print_sample ~prev_y ~prev_mem_data:mem_data
      end;
      smooth_error, mem_data)
  |> ignore

let read_file ?(filename = "data/input.txt") () =
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

let () =
  Random.init 42;
  let data, all_chars = read_file () in
  fit_and_evaluate data all_chars
