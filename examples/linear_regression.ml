open Core.Std
module CArray = Ctypes.CArray
module H = Helper
module Tensor = Wrapper.Tensor

let () =
  Ops_gradients.register_all ();
  let n = 3 in (* size of y *)
  let m = 2 in (* size of x *)
  let x_tensor = Tensor.create2d TF_FLOAT 1 m in
  let x_data = Tensor.data x_tensor Ctypes.float m in
  for i = 0 to m - 1 do
    CArray.set x_data i (float (i+1));
  done;
  let y_tensor = Tensor.create2d TF_FLOAT 1 n in
  let y_data = Tensor.data y_tensor Ctypes.float n in
  for i = 0 to n - 1 do
    CArray.set y_data i (float (4*(i+1)));
  done;
  let x = Ops.placeholder ~name:"x" ~type_:Float () in
  let y = Ops.placeholder ~name:"y" ~type_:Float () in
  let w = Ops_m.varf [ m; n ] in
  let b = Ops_m.varf [ n ] in
  let w_assign = Ops.assign w (H.const_float [ m; n ] 0.) in
  let b_assign = Ops.assign b (H.const_float [ n ] 0.) in
  let diff = Ops_m.(x *^ w + b - y) in
  let err = Ops.matMul diff diff ~transpose_b:true in
  let gradient_w, gradient_b =
    Gradients.gradient err
      ~with_respect_to_float:[ w; b ]
      ~with_respect_to_double:[]
    |> function
    | [ gradient_w; gradient_b ], [] -> gradient_w, gradient_b
    | _ -> assert false
  in
  let alpha = Ops_m.f 0.4 in
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
        ~inputs:[ x, x_tensor; y, y_tensor ]
        ~outputs:[ err; gradient_w; gradient_b ]
        ~targets:[ err; gradient_w; gradient_b ]
    in
    H.print_tensors output ~names:[ "err"; "grad-w"; "grad-b" ]
  in
  print_err ();
  let _output =
    H.run session
      ~inputs:[ x, x_tensor; y, y_tensor ]
      ~outputs:[ gd_w; gd_b ]
      ~targets:[ gd_w; gd_b ]
  in
  print_err ()

