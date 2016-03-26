open Core_kernel.Std
module H = Helper

let run ~samples ~size_xs ~size_y ~xs ~y =
  let xs = List.concat xs in
  let xs = Ops_m.cf ~shape:[samples; size_xs] xs in
  let y  = Ops_m.cf ~shape:[samples; size_y]  y in
  let w = Ops_m.varf [ size_xs; size_y ] in
  let b = Ops_m.varf [ size_y ] in
  let w_assign = Ops.assign w (Ops_m.f ~shape:[ size_xs; size_y ] 0.) in
  let b_assign = Ops.assign b (Ops_m.f ~shape:[ size_y ] 0.) in
  let y_ = Ops_m.(xs *^ w + b) in
  let err = Ops_m.(Ops.square (y_ - y) |> reduce_mean) in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:0.04 ~varsf:[ w; b ] err
  in
  let session =
    H.create_session (Node.[ P err; P w_assign; P b_assign ] @ gd)
  in
  let _output =
    H.run session
      ~outputs:[]
      ~targets:[ w_assign; b_assign ] 
  in
  let results = ref [] in
  let print_err n =
    let output =
      H.run session
        ~outputs:[ err; y_ ]
        ~targets:[ err ]
    in
    match output with
    | [ err; y_ ] ->
      H.print_tensors [ err ] ~names:[ sprintf "err %d" n ];
      results := (n, Tensor.to_float_list y_) :: !results
    | _ -> assert false
  in
  for i = 0 to 500 do
    let output =
      Wrapper.Session.run session
        ~targets:(List.map gd ~f:(fun n -> Node.packed_name n |> Node.Name.to_string))
    in
    ignore output;
    if i % 100 = 0 then print_err i
  done;
  !results
