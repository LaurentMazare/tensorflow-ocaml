open Core.Std
module CArray = Ctypes.CArray
module H = Helper
module Tensor = Wrapper.Tensor

let () =
  Ops_gradients.register_all ();
  let n = 3 in (* size of y *)
  let m = 2 in (* size of x *)
  let x = Ops_m.const_float ~type_:Float ~shape:[1; m] [ 1.; 2. ] in
  let y = Ops_m.const_float ~type_:Float ~shape:[1; n] [ 5.; 8.; 12. ] in
  let w =
    Ops.variable ()
      ~type_:Float
      ~shape:[ { size = m; name = None }; { size = n; name = None } ]
  in
  let b =
    Ops.variable ()
      ~type_:Float
      ~shape:[ { size = n; name = None } ]
  in
  let w_assign = Ops.assign w (H.const_float [ m; n ] 0.) in
  let b_assign = Ops.assign b (H.const_float [ n ] 0.) in
  let diff = Ops.matMul x w |> Ops.add b |> Ops.sub y in
  let err = Ops.matMul diff diff ~transpose_b:true in
  let gradient_w, gradient_b =
    Gradients.gradient err
      ~with_respect_to_float:[ w; b ]
      ~with_respect_to_double:[]
    |> function
    | [ gradient_w; gradient_b ], [] -> gradient_w, gradient_b
    | _ -> assert false
  in
  let alpha = Ops_m.f 0.0004 in
  let gradient_w = Ops.reshape gradient_w (Ops.shape w) in
  let gradient_b = Ops.reshape gradient_b (Ops.shape b) in
  let gd_w = Ops.applyGradientDescent w alpha gradient_w in
  let gd_b = Ops.applyGradientDescent b alpha gradient_b in
  let session =
    H.create_session
      [ P err; P w_assign; P b_assign; P gradient_w; P gradient_b; P gd_w; P gd_b ]
  in
  let output =
    H.run session
      ~inputs:[]
      ~outputs:[ w_assign ]
      ~targets:[ w_assign; b_assign ] 
  in
  H.print_tensors output ~names:[ "init" ];
  let output =
    H.run session
      ~inputs:[]
      ~outputs:[ w; b ]
      ~targets:[ w; b ]
  in
  H.print_tensors output ~names:[ "w"; "b" ];
  let print_err () =
    let output =
      H.run session
        ~inputs:[]
        ~outputs:[ err ]
        ~targets:[ err ]
    in
    H.print_tensors output ~names:[ "err" ]
  in
  print_err ();
  for i = 0 to 1000 do
    let output =
      H.run session
        ~inputs:[]
        ~outputs:[ gd_w; gd_b ]
        ~targets:[ gd_w; gd_b ]
    in
    ignore output;
    if i % 10 = 0 then print_err ()
  done;
  print_err ()

