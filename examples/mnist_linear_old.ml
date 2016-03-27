open Core_kernel.Std
module H = Helper

let train_size = 1000
let validation_size = 1000
let image_dim = Mnist.image_dim
let label_count = Mnist.label_count
let epochs = 10

let () =
  let { Mnist.train_images; train_labels; validation_images; validation_labels } =
    Mnist.read_files ~train_size ~validation_size ()
  in
  let xs = Ops_m.placeholder [] ~type_:Float in
  let ys = Ops_m.placeholder [] ~type_:Float in
  let zero_w = Ops_m.f ~shape:[ image_dim; label_count ] 0. in
  let w = Ops_m.varf [ image_dim; label_count ] in
  let w_assign = Ops.assign w zero_w in
  let zero_b = Ops_m.f ~shape:[ label_count ] 0. in
  let b = Ops_m.varf [ label_count ] in
  let b_assign = Ops.assign b zero_b in
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
  let train_inputs = [ xs, Tensor.P train_images; ys, Tensor.P train_labels ] in
  let validation_inputs =
    [ xs, Tensor.P validation_images; ys, Tensor.P validation_labels ]
  in
  let print_err n =
    let output =
      H.run session
        ~inputs:validation_inputs
        ~outputs:[ accuracy ]
        ~targets:[ accuracy ]
    in
    match output with
    | [ accuracy ] ->
      H.print_tensors [ accuracy ] ~names:[ sprintf "acc %d" n ];
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
