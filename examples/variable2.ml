module H = Helper

let () =
  let data = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 3 |] in
  Bigarray.Genarray.set data [| 0 |] 1.;
  Bigarray.Genarray.set data [| 1 |] 2.;
  Bigarray.Genarray.set data [| 2 |] 6.;
  let placeholder = Ops.placeholder ~name:"x" ~type_:Float () in
  let variable = Var.float [ 3 ] ~init:(Ops_m.cf [ 8.; 0.; 1. ]) in
  let node =
    Ops.sub variable placeholder
    |> Ops.abs
  in
  let sum = Ops_m.reduce_sum node in
  let session = Session.create () in
  let node, sum =
    Session.(run
      session
      ~inputs:[ Input.float placeholder data ]
       (Output.(both (float node) (float sum))))
  in
  H.print_tensors [Tensor.P node; P sum] ~names:[ "node"; "sum" ]
