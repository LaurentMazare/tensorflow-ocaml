open Core_kernel.Std
module H = Helper

let train_size = 1000
let image_dim = 28 * 28
let label_count = 10
let epochs = 1000

let () =
  let train_images =
    Mnist.read_images "data/train-images-idx3-ubyte" ~nsamples:train_size
  in
  let train_labels =
    Mnist.read_labels "data/train-labels-idx1-ubyte" ~nsamples:train_size
  in
  let nsamples = Bigarray.Array1.dim train_labels in
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
  let inputs =
    let train_labels_p =
      Bigarray.Genarray.create
        Bigarray.float32
        Bigarray.c_layout
        [| nsamples; label_count |]
    in
    for idx = 0 to nsamples - 1 do
      for lbl = 0 to 9 do
        Bigarray.Genarray.set train_labels_p [| idx; lbl |] 0.
      done;
      let lbl = Bigarray.Array1.get train_labels idx |> Int32.to_int_exn in
      Bigarray.Genarray.set train_labels_p [| idx; lbl |] 1.
    done;
    [ xs, Tensor.P (Bigarray.genarray_of_array2 train_images)
    ; ys, Tensor.P train_labels_p
    ]
  in
  let print_err n =
    let output =
      H.run session
        ~inputs
        ~outputs:[ accuracy; cross_entropy ]
        ~targets:[ accuracy; cross_entropy ]
    in
    match output with
    | [ accuracy; cross_entropy ] ->
      H.print_tensors [ accuracy; cross_entropy ] ~names:[ sprintf "acc %d" n; "ce" ];
    | _ -> assert false
  in
  let inputs =
    List.map inputs ~f:(fun (n, tensor) -> n.Node.name |> Node.Name.to_string, tensor)
  in
  for i = 0 to epochs do
    print_err i;
    let output =
      Wrapper.Session.run session
        ~inputs
        ~targets:(List.map gd ~f:(fun n -> Node.packed_name n |> Node.Name.to_string))
    in
    ignore output;
  done
