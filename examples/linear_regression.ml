open Core.Std
module H = Helper
module Tensor = Wrapper.Tensor

let () =
  let n = 3 in (* size of y *)
  let m = 2 in (* size of x *)
  let x = Ops_m.cf ~shape:[1; m] [ 1.; 2. ] in
  let y = Ops_m.cf ~shape:[1; n] [ 5.; 8.; 12. ] in
  let w = Ops_m.varf [ m; n ] in
  let b = Ops_m.varf [ n ] in
  let w_assign = Ops.assign w (H.const_float [ m; n ] 0.) in
  let b_assign = Ops.assign b (H.const_float [ n ] 0.) in
  let err =
    let open Ops_m in
    let diff = x *^ w + b - y in
    diff *. diff
  in
  let gd =
    Optimizers.gradient_descent_minimizer ~alpha:0.0004 ~varsf:[ w; b ] err
  in
  let session =
    H.create_session (Node.[ P err; P w_assign; P b_assign ] @ gd)
  in
  let output =
    H.run session
      ~outputs:[ w_assign ]
      ~targets:[ w_assign; b_assign ] 
  in
  H.print_tensors output ~names:[ "init" ];
  let output =
    H.run session
      ~outputs:[ w; b ]
      ~targets:[ w; b ]
  in
  H.print_tensors output ~names:[ "w"; "b" ];
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

