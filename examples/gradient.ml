open Core.Std
module H = Helper
module Tensor = Wrapper.Tensor

let () =
  let input_tensor = Tensor.create1d TF_FLOAT 3 in
  let data = Tensor.data input_tensor Bigarray.float32 3 in
  Bigarray.Array1.set data 0 1.;
  Bigarray.Array1.set data 1 2.;
  Bigarray.Array1.set data 2 6.;
  let placeholder = Ops.placeholder ~name:"x" ~type_:Float () in
  let node =
    let open Ops_m in
    cf [ 2.; 1.; 4. ] - placeholder - placeholder
  in
  let gradient =
    Gradients.gradient node
      ~with_respect_to_float:[ placeholder ]
      ~with_respect_to_double:[]
    |> function
    | [ float ], [] -> float
    | _ -> assert false
  in
  let session = H.create_session [ P node; P gradient ] in
  let output =
    H.run
      session
      ~inputs:[ placeholder, input_tensor ]
      ~outputs:[ gradient ]
      ~targets:[ gradient ]
  in
  H.print_tensors output ~names:[ "grad" ]
