open Core_kernel.Std
open Tensorflow

let epochs = 100000
let seq_len = 3
let batch_size = 128
let hidden_units = 50
let depth = 2

let all_vars_with_names node =
  Var.get_all_vars node
  |> List.mapi ~f:(fun i var -> sprintf "V%d" i, var)

let generate_data ~samples =
  let xs = Tensor.create3 Float32 samples (3 + 2*seq_len) 37 in
  let ys = Tensor.create2 Float32 samples 10 in
  Tensor.fill xs 0.;
  Tensor.fill ys 0.;
  for sample_idx = 0 to samples - 1 do
    let letters = Array.init 26 ~f:Fn.id in
    let numbers = Array.init seq_len ~f:(fun _ -> Random.int 10) in
    (* Fisher-Yates like shuffling. *)
    for i = 0 to seq_len - 1 do
      let tmp = letters.(i) in
      let j = i + Random.int (26 - i) in
      letters.(i) <- letters.(j);
      letters.(j) <- tmp;
      Tensor.set xs [| sample_idx; 2*i; letters.(i) |] 1.;
      Tensor.set xs [| sample_idx; 2*i+1; 26 + numbers.(i) |] 1.;
    done;
    Tensor.set xs [| sample_idx; 2*seq_len; 36 |] 1.;
    Tensor.set xs [| sample_idx; 2*seq_len+1; 36 |] 1.;
    let target = Random.int seq_len in
    Tensor.set xs [| sample_idx; 2*seq_len+2; letters.(target) |] 1.;
    Tensor.set ys [| sample_idx; numbers.(target) |] 1.;
  done;
  xs, ys

let const_i32 = Ops.const_int ~type_:Int32

(* Adapted from https://github.com/ajarai/fast-weights. *)
let model ~input_len ~input_dim ~output_dim =
  let type_ = Node.Type.Float in
  let xs_placeholder = Ops.placeholder ~type_ [ batch_size; input_len; input_dim ] in
  let ys_placeholder = Ops.placeholder ~type_ [ batch_size; output_dim ] in
  let l_placeholder = Ops.placeholder ~type_ [ 1 ] in
  let e_placeholder = Ops.placeholder ~type_ [ 1 ] in
  let l = Ops.Placeholder.to_node l_placeholder in
  let e = Ops.Placeholder.to_node e_placeholder in
  let xs = Ops.Placeholder.to_node xs_placeholder in
  let ys = Ops.Placeholder.to_node ys_placeholder in
  let w_x = Var.normal ~type_ [ input_dim; hidden_units ] ~stddev:0.1 in
  let b_x = Var.f_or_d [ hidden_units ] 0. ~type_ in
  let w_h = Var.normal ~type_ [ hidden_units; hidden_units ] ~stddev:0.1 in
  let w_softmax = Var.normal ~type_ [ hidden_units; output_dim ] ~stddev:0.1 in
  let b_softmax = Var.normal ~type_ [ output_dim ] ~stddev:0.1 in
  let gain = Var.f_or_d [ hidden_units ] 1. ~type_ in
  let bias = Var.f_or_d [ hidden_units ] 0. ~type_ in
  let fw_a = Ops.f_or_d 0. ~shape:[ batch_size; hidden_units; hidden_units ] ~type_ in
  let fw_h = Ops.f_or_d 0. ~shape:[ batch_size; hidden_units ] ~type_ in
  let shape_fw_h_s_3 = const_i32 ~shape:[ 3 ] [ batch_size; 1; hidden_units ] in
  let shape_fw_h_s_2 = const_i32 ~shape:[ 3 ] [ batch_size; hidden_units ] in
  let zero_two_one = const_i32 ~shape:[ 3 ] [ 0; 2; 1 ] in
  let fw_a, fw_h =
    List.init input_len ~f:Fn.id
    |> List.fold ~init:(fw_a, fw_h) ~f:(fun (fw_a, fw_h) input_idx ->
        let xs = Ops.slice xs (const_i32 [ 0; input_idx; 0 ]) (const_i32 [ batch_size; 1; input_dim ]) in
        let fw_h = Ops.(xs *^ w_x + fw_h *^ w_h + b_x |> relu) in
        let fw_h_s = Ops.reshape fw_h shape_fw_h_s_3 in
        let fw_a = Ops.(l * fw_a + e * batchMatMul (transpose fw_h_s zero_two_one) fw_h_s) in
        let fw_h_s =
          List.init depth ~f:Fn.id
          |> List.fold ~init:fw_h_s ~f:(fun fw_h_s _idx ->
              let fw_h_s =
                Ops.(
                  reshape (xs *^ w_x + b_x) shape_fw_h_s_3
                  + reshape (fw_h *^ w_h) shape_fw_h_s_3
                  + batchMatMul fw_h_s fw_a)
              in
              let mu = Ops.reduce_mean fw_h_s in
              let sigma = Ops.(square (fw_h_s - mu) |> reduce_mean |> sqrt) in
              Ops.(gain * (fw_h_s - mu) / sigma + bias |> relu))
        in
        let fw_h = Ops.reshape fw_h_s shape_fw_h_s_2 in
        fw_a, fw_h
      )
  in
  let y_hats = Ops.(fw_h *^ w_softmax + b_softmax |> softmax) in
  let cross_entropy = Cell.cross_entropy ~ys ~y_hats in
  xs_placeholder, ys_placeholder, cross_entropy

let () =
  let test_xs, test_ys = generate_data ~samples:512_000 in
  let valid_xs, valid_ys = generate_data ~samples:128_000 in
  ()
