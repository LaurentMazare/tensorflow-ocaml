open Core.Std
module H = Helper

let () =
  let data = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 3 |] in
  Bigarray.Genarray.set data [| 0 |] 1.;
  Bigarray.Genarray.set data [| 1 |] 2.;
  Bigarray.Genarray.set data [| 2 |] 6.;
  let input_tensor = Tensor.P data in
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
