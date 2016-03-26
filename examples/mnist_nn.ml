open Core_kernel.Std
module H = Helper

let image_dim = 28 * 28
let label_count = 10
let hidden_nodes = 64
let epochs = 1000

let () =
  let train_images = Mnist.read_images "data/train-images-idx3-ubyte" in
  let train_labels = Mnist.read_labels "data/train-labels-idx1-ubyte" in
  let nsamples = Bigarray.Array1.dim train_labels in
  let xs = Ops_m.placeholder [] ~type_:Float in
  let ys = Ops_m.placeholder [] ~type_:Float in
  let w1 = Ops_m.varf [ image_dim; hidden_nodes ] in
  let b1 = Ops_m.varf [ hidden_nodes ] in
  let w2 = Ops_m.varf [ hidden_nodes; label_count ] in
  let b2 = Ops_m.varf [ label_count ] in
  let w1_assign = Ops.assign w1 (Ops_m.f ~shape:[ image_dim; hidden_nodes ] 0.) in
  let b1_assign = Ops.assign b1 (Ops_m.f ~shape:[ hidden_nodes ] 0.) in
  let w2_assign = Ops.assign w2 (Ops_m.f ~shape:[ hidden_nodes; label_count ] 0.) in
  let b2_assign = Ops.assign b2 (Ops_m.f ~shape:[ label_count ] 0.) in
  let ys_ = Ops_m.(Ops.sigmoid (xs *^ w1 + b1) *^ w2 + b2) |> Ops.softmax in
  let cross_entropy = Ops.neg Ops_m.(reduce_sum (ys * Ops.log ys_)) in
  let accuracy =
    Ops.equal (Ops.argMax ys_ Ops_m.one32) (Ops.argMax ys Ops_m.one32)
    |> Ops.cast ~type_:Float
    |> Ops_m.reduce_mean
  in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:0.01 ~varsf:[ w1; w2; b1; b2 ]
      cross_entropy
  in
  let session =
    H.create_session
      (Node.[ P accuracy; P w1_assign; P b1_assign; P w2_assign; P b2_assign ] @ gd)
  in
  let _output =
    H.run session
      ~outputs:[]
      ~targets:[ w1_assign; b1_assign; w2_assign; b2_assign ] 
  in
  let results = ref [] in
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
        ~outputs:[ accuracy; ys_ ]
        ~targets:[ accuracy ]
    in
    match output with
    | [ accuracy; ys_ ] ->
      H.print_tensors [ accuracy ] ~names:[ sprintf "ce %d" n ];
      results := (n, Tensor.to_float_list ys_) :: !results
    | _ -> assert false
  in
  let inputs =
    List.map inputs ~f:(fun (n, tensor) -> n.Node.name |> Node.Name.to_string, tensor)
  in
  for i = 0 to epochs do
    let output =
      Wrapper.Session.run session
        ~inputs
        ~targets:(List.map gd ~f:(fun n -> Node.packed_name n |> Node.Name.to_string))
    in
    ignore output;
    print_err i
  done;
  ignore !results

