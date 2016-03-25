open Wrapper

let ok_exn (result : 'a Session.result) ~context =
  match result with
  | Ok result -> result
  | Error status ->
    Printf.sprintf "Error in %s: %s" context (Status.message status)
    |> failwith

let () =
  let data = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [| 3 |] in
  Bigarray.Genarray.set data [| 0 |] 1.;
  Bigarray.Genarray.set data [| 1 |] 2.;
  Bigarray.Genarray.set data [| 2 |] 6.;
  let input_tensor = Tensor.P data in
  let session =
    Session.create ()
    |> ok_exn ~context:"session creation"
  in
  Session.extend_graph
    session
    (Protobuf.read_file "examples/load.pb")
    |> ok_exn ~context:"extending graph";
  let output =
    Session.run
      session
      ~inputs:[ "x", input_tensor ]
      ~outputs:[ "add" ]
      ~targets:[ "add" ]
    |> ok_exn ~context:"session run"
  in
  Helper.print_tensors output ~names:[ "load" ]
