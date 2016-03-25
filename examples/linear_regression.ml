open Core_kernel.Std
module H = Helper

let () =
  let samples = 100 in
  let size_y = 1 in
  let size_xs = 3 in
  let x = List.init samples ~f:(fun i -> 3.14 *. float i /. float samples) in
  let xs = List.concat_map x ~f:(fun x -> [ x; x *. x ]) in
  let y = List.map x ~f:sin in
  let xs = Ops_m.cf ~shape:[samples; size_xs] xs in
  let y  = Ops_m.cf ~shape:[samples; size_y]  y in
  let w = Ops_m.varf [ size_xs; size_y ] in
  let b = Ops_m.varf [ size_y ] in
  let w_assign = Ops.assign w (Ops_m.f ~shape:[ size_xs; size_y ] 0.) in
  let b_assign = Ops.assign b (Ops_m.f ~shape:[ size_y ] 0.) in
  let err =
    let open Ops_m in
    Ops.square (xs *^ w + b - y) |> reduce_mean
  in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:0.0004 ~varsf:[ w; b ] err
  in
  let session =
    H.create_session (Node.[ P err; P w_assign; P b_assign ] @ gd)
  in
  let _output =
    H.run session
      ~outputs:[]
      ~targets:[ w_assign; b_assign ] 
  in
  let print_err () =
    let output =
      H.run session
        ~outputs:[ err ]
        ~targets:[ err ]
    in
    H.print_tensors output ~names:[ "err" ]
  in
  print_err ();
  for i = 0 to 1000 do
    let output =
      Wrapper.Session.run session
        ~targets:(List.map gd ~f:(fun n -> Node.packed_name n |> Node.Name.to_string))
    in
    ignore output;
    if i % 100 = 0 then print_err ()
  done;
  print_err ()
