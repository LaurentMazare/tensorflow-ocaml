open Tensorflow
open Wrapper

let ok_exn (result : 'a Session.result) ~context =
  match result with
  | Ok result -> result
  | Error status ->
    Printf.sprintf "Error in %s: %s" context (Status.message status)
    |> failwith

let () =
  let data = Tensor.create1 Float32 3 in
  Tensor.set data [| 0 |] 1.;
  Tensor.set data [| 1 |] 2.;
  Tensor.set data [| 2 |] 6.;
  let input_tensor = Tensor.P data in
  let session =
    Session.create ()
    |> ok_exn ~context:"session creation"
  in
  Session.extend_graph
    session
    (Protobuf.read_file "examples/load/load.pb")
    |> ok_exn ~context:"extending graph";
  let output =
    Session.run
      session
      ~inputs:[ "x", input_tensor ]
      ~outputs:[ "add" ]
      ~targets:[ "add" ]
    |> ok_exn ~context:"session run"
  in
  match output with
  | [ output ] -> Tensor.print output
  | _ -> assert false
