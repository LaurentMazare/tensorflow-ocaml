open Wrapper
module CArray = Ctypes.CArray

let ok_exn (result : 'a Session.result) ~context =
  match result with
  | Ok result -> result
  | Error status ->
    Printf.sprintf "Error in %s: %s" context (Status.message status)
    |> failwith

let () =
  let graph =
    Graph.add
      (Graph.const [ 4.; 16. ])
      (Graph.const [ 38.; 16. ])
  in
  let session_options = Session_options.create () in
  let session =
    Session.create session_options
    |> ok_exn ~context:"session creation"
  in
  Session.extend_graph
    session
    (Graph.Protobuf.to_protobuf graph)
    |> ok_exn ~context:"extending graph";
  let output =
    Session.run
      session
      ~inputs:[]
      ~outputs:[ Graph.name graph ]
      ~targets:[ Graph.name graph ]
    |> ok_exn ~context:"session run"
  in
  match output with
  | [ output_tensor ] ->
    let data = Tensor.data output_tensor Ctypes.float 2 in
    Printf.printf "%f %f\n%!" (CArray.get data 0) (CArray.get data 1)
  | [] | _ :: _ :: _ -> assert false

