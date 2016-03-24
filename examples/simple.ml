open Wrapper
module CArray = Ctypes.CArray
module H = Helper
module Tensor = Wrapper.Tensor
module Session = Wrapper.Session

let () =
  let input_tensor = Tensor.create1d TF_FLOAT 3 in
  let data = Tensor.data input_tensor Ctypes.float 3 in
  CArray.set data 0 1.;
  CArray.set data 1 2.;
  CArray.set data 2 6.;
  let placeholder = Ops.placeholder ~name:"x" ~type_:Float () in
  let node =
    Ops_m.(cf [ 2.; 1.; 4. ] - placeholder)
    |> Ops.abs
  in
  let session = H.create_session [ Node.P node ] in
  let output =
    H.run
      session
      ~inputs:[ placeholder, input_tensor ]
      ~outputs:[ node ]
      ~targets:[ node ]
  in
  H.print_tensors output ~names:[ "simple" ]
