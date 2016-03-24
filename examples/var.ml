open Wrapper
module CArray = Ctypes.CArray
module H = Helper
module Tensor = Wrapper.Tensor

let () =
  let input_tensor = Tensor.create1d TF_FLOAT 3 in
  let data = Tensor.data input_tensor Ctypes.float 3 in
  CArray.set data 0 1.;
  CArray.set data 1 2.;
  CArray.set data 2 6.;
  let placeholder = Ops.placeholder ~name:"x" ~type_:Float () in
  let variable = Ops_m.varf [ 3 ] in
  let assign =
    Ops.assign variable (Ops_m.const_float ~type_:Float [ 8.; 0.; 1. ])
  in
  let node =
    Ops.sub assign placeholder
    |> Ops.abs
  in
  let session = H.create_session [ P node ] in
  let output =
    H.run
      session
      ~inputs:[ placeholder, input_tensor ]
      ~outputs:[ node ]
      ~targets:[ node ]
  in
  H.print_tensors output ~names:[ "var" ]
