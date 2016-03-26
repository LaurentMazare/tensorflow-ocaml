open Core_kernel.Std
module H = Helper

let verbose = false

let rnd ~shape =
  Ops.randomStandardNormal (Ops_m.const_int ~type_:Int32 shape) ~type_:Float
  |> Ops.mul (Ops_m.f 0.1)

let one_layer ~samples ~size_xs ~size_ys ~xs ~ys ~hidden_nodes ~epochs =
  let xs = List.concat xs in
  let ys = List.concat ys in
  let xs = Ops_m.cf ~shape:[samples; size_xs] xs in
  let y  = Ops_m.cf ~shape:[samples; size_ys] ys in
  let w1 = Ops_m.varf [ size_xs; hidden_nodes ] in
  let b1 = Ops_m.varf [ hidden_nodes ] in
  let w2 = Ops_m.varf [ hidden_nodes; size_ys ] in
  let b2 = Ops_m.varf [ size_ys ] in
  let w1_assign = Ops.assign w1 (rnd ~shape:[ size_xs; hidden_nodes ]) in
  let b1_assign = Ops.assign b1 (Ops_m.f ~shape:[ hidden_nodes ] 0.) in
  let w2_assign = Ops.assign w2 (rnd ~shape:[ hidden_nodes; size_ys ]) in
  let b2_assign = Ops.assign b2 (Ops_m.f ~shape:[ size_ys ] 0.) in
  let y_ = Ops_m.(Ops.sigmoid (xs *^ w1 + b1) *^ w2 + b2) in
  let err = Ops_m.(Ops.square (y_ - y) |> reduce_mean) in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:0.4 ~varsf:[ w1; w2; b1; b2 ] err
  in
  let session =
    H.create_session (Node.[ P err; P w1_assign; P b1_assign; P w2_assign; P b2_assign ] @ gd)
  in
  let _output =
    H.run session
      ~outputs:[]
      ~targets:[ w1_assign; b1_assign; w2_assign; b2_assign ] 
  in
  let results = ref [] in
  let print_err n =
    let output =
      H.run session
        ~outputs:[ err; y_; w1; w2; b1; b2 ]
        ~targets:[ err ]
    in
    match output with
    | [ err; y_; w1; w2; b1; b2 ] ->
      if verbose
      then
        H.print_tensors
          [ err; w1; w2; b1; b2 ]
          ~names:[ sprintf "err %d" n; "w1"; "w2"; "b1"; "b2" ]
      else H.print_tensors [ err ] ~names:[ sprintf "err %d" n ];
      results := (n, Tensor.to_float_list y_) :: !results
    | _ -> assert false
  in
  for i = 0 to epochs do
    let output =
      Wrapper.Session.run session
        ~targets:(List.map gd ~f:(fun n -> Node.packed_name n |> Node.Name.to_string))
    in
    ignore output;
    if i % 1000 = 0 then print_err i
  done;
  !results
