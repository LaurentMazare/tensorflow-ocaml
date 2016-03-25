open Wrapper
module H = Helper
module Tensor = Wrapper.Tensor

let () =
  let input_tensor = Tensor.create1d TF_FLOAT 3 in
  let data = Tensor.data input_tensor Bigarray.float32 3 in
  Bigarray.Array1.set data 0 1.;
  Bigarray.Array1.set data 1 2.;
  Bigarray.Array1.set data 2 6.;
  let placeholder = Ops.placeholder ~name:"x" ~type_:Float () in
  let variable = Ops_m.varf [ 3 ] in
  let assign =
    Ops.assign variable (Ops_m.cf [ 8.; 0.; 1. ])
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
