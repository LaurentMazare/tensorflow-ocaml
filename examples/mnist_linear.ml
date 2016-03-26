open Core_kernel.Std
module H = Helper

let train_size = 1000
let validation_size = 1000
let image_dim = 28 * 28
let label_count = 10
let epochs = 1000

let slice1 data start_idx n =
  let slice =
    Bigarray.Array1.create (Bigarray.Array1.kind data) Bigarray.c_layout n
  in
  for i = 0 to n - 1 do
    Bigarray.Array1.set slice i (Bigarray.Array1.get data (start_idx + i))
  done;
  slice

let slice2 data start_idx n =
  let dim2 = Bigarray.Array2.dim2 data in
  let slice =
    Bigarray.Array2.create (Bigarray.Array2.kind data) Bigarray.c_layout n dim2
  in
  for i = 0 to n - 1 do
    for j = 0 to dim2 - 1 do
      Bigarray.Array2.set slice i j (Bigarray.Array2.get data (start_idx + i) j)
    done;
  done;
  slice

let one_hot labels =
  let nsamples = Bigarray.Array1.dim labels in
  let one_hot =
    Bigarray.Genarray.create
      Bigarray.float32
      Bigarray.c_layout
      [| nsamples; label_count |]
  in
  for idx = 0 to nsamples - 1 do
    for lbl = 0 to 9 do
      Bigarray.Genarray.set one_hot [| idx; lbl |] 0.
    done;
    let lbl = Bigarray.Array1.get labels idx |> Int32.to_int_exn in
    Bigarray.Genarray.set one_hot [| idx; lbl |] 1.
  done;
  one_hot

let () =
  let all_images =
    Mnist.read_images "data/train-images-idx3-ubyte"
      ~nsamples:(train_size + validation_size)
  in
  let all_labels =
    Mnist.read_labels "data/train-labels-idx1-ubyte"
      ~nsamples:(train_size + validation_size)
  in
  let train_images = slice2 all_images 0 train_size in
  let train_labels = slice1 all_labels 0 train_size in
  let validation_images = slice2 all_images train_size validation_size in
  let validation_labels = slice1 all_labels train_size validation_size in
  let xs = Ops_m.placeholder [] ~type_:Float in
  let ys = Ops_m.placeholder [] ~type_:Float in
  let w = Ops_m.varf [ image_dim; label_count ] in
  let b = Ops_m.varf [ label_count ] in
  let w_assign = Ops.assign w (Ops_m.f ~shape:[ image_dim; label_count ] 0.) in
  let b_assign = Ops.assign b (Ops_m.f ~shape:[ label_count ] 0.) in
  let ys_ = Ops_m.(xs *^ w + b) |> Ops.softmax in
  let cross_entropy = Ops.neg Ops_m.(reduce_sum (ys * Ops.log ys_)) in
  let accuracy =
    Ops.equal (Ops.argMax ys_ Ops_m.one32) (Ops.argMax ys Ops_m.one32)
    |> Ops.cast ~type_:Float
    |> Ops_m.reduce_mean
  in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:0.001 ~varsf:[ w; b ]
      cross_entropy
  in
  let session =
    H.create_session
      (Node.[ P accuracy; P w_assign; P b_assign ] @ gd)
  in
  let _output =
    H.run session
      ~outputs:[]
      ~targets:[ w_assign; b_assign ] 
  in
  let train_inputs =
    [ xs, Tensor.P (Bigarray.genarray_of_array2 train_images)
    ; ys, Tensor.P (one_hot train_labels)
    ]
  in
  let validation_inputs =
    [ xs, Tensor.P (Bigarray.genarray_of_array2 validation_images)
    ; ys, Tensor.P (one_hot validation_labels)
    ]
  in
  let print_err n =
    let output =
      H.run session
        ~inputs:validation_inputs
        ~outputs:[ accuracy; cross_entropy ]
        ~targets:[ accuracy; cross_entropy ]
    in
    match output with
    | [ accuracy; cross_entropy ] ->
      H.print_tensors [ accuracy; cross_entropy ] ~names:[ sprintf "acc %d" n; "ce" ];
    | _ -> assert false
  in
  let train_inputs =
    List.map train_inputs ~f:(fun (n, tensor) -> n.Node.name |> Node.Name.to_string, tensor)
  in
  for i = 0 to epochs do
    print_err i;
    let output =
      Wrapper.Session.run session
        ~inputs:train_inputs
        ~targets:(List.map gd ~f:(fun n -> Node.packed_name n |> Node.Name.to_string))
    in
    ignore output;
  done
