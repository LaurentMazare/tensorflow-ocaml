module H = Helper

let () =
  let data = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 3 |] in
  Bigarray.Genarray.set data [| 0 |] 1.;
  Bigarray.Genarray.set data [| 1 |] 2.;
  Bigarray.Genarray.set data [| 2 |] 6.;
  let input_tensor = Tensor.P data in
  let placeholder = Ops.placeholder ~name:"x" ~type_:Float () in
  let variable = Variable.float [ 3 ] ~init:(Ops_m.cf [ 8.; 0.; 1. ]) in
  let node =
    Ops.sub variable placeholder
    |> Ops.abs
  in
  let sum = Ops_m.reduce_sum node in
  let session = H.create_session [ P node; P sum ] in
  let output =
    H.run
      session
      ~inputs:[ placeholder, input_tensor ]
      ~outputs:[ node; sum ]
      ~targets:[ node; sum ]
  in
  H.print_tensors output ~names:[ "var"; "sum" ]
