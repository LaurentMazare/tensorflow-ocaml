module H = Helper

let () =
  let data = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 3 |] in
  Bigarray.Genarray.set data [| 0 |] 1.;
  Bigarray.Genarray.set data [| 1 |] 2.;
  Bigarray.Genarray.set data [| 2 |] 6.;
  let input_tensor = Tensor.P { data; kind = Bigarray.float32 } in
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
