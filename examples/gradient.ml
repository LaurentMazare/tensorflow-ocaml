open Core_kernel.Std
module H = Helper

let () =
  let data = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 3 |] in
  Bigarray.Genarray.set data [| 0 |] 1.;
  Bigarray.Genarray.set data [| 1 |] 2.;
  Bigarray.Genarray.set data [| 2 |] 6.;
  let input_tensor = Tensor.P data in
  let ph = Ops.placeholder ~name:"x" ~type_:Float () in
  let node =
    let open Ops_m in
    cf [ 2.; 1.; 4. ] - ph - ph
  in
  let gradient =
    Gradients.gradient node
      ~with_respect_to_float:[ ph ]
      ~with_respect_to_double:[]
    |> function
    | [ float ], [] -> float
    | _ -> assert false
  in
  let session = H.create_session [ P node; P gradient ] in
  let output =
    H.run
      session
      ~inputs:[ ph, input_tensor ]
      ~outputs:[ gradient ]
      ~targets:[ gradient ]
  in
  H.print_tensors output ~names:[ "grad" ]
