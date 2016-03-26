module H = Helper
module Session = Wrapper.Session

let () =
  let data = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 3 |] in
  Bigarray.Genarray.set data [| 0 |] 1.;
  Bigarray.Genarray.set data [| 1 |] 2.;
  Bigarray.Genarray.set data [| 2 |] 6.;
  let input_tensor = Tensor.P data in
  let ph = Ops.placeholder ~name:"x" ~type_:Float () in
  let node = Ops_m.(cf [ 2.; 1.; 4. ] - ph) |> Ops.abs in
  let session = H.create_session [ Node.P node ] in
  let output =
    H.run
      session
      ~inputs:[ ph, input_tensor ]
      ~outputs:[ node ]
      ~targets:[ node ]
  in
  H.print_tensors output ~names:[ "simple" ]
