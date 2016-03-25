open Wrapper
module H = Helper
module Tensor = Wrapper.Tensor
module Session = Wrapper.Session

let () =
  let input_tensor = Tensor.create1d TF_FLOAT 3 in
  let data = Tensor.data input_tensor Bigarray.float32 3 in
  Bigarray.Array1.set data 0 1.;
  Bigarray.Array1.set data 1 2.;
  Bigarray.Array1.set data 2 6.;
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
