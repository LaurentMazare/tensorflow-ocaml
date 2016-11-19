open Tensorflow
open Wrapper

let ok_exn (result : 'a Status.result) ~context =
  match result with
  | Ok result -> result
  | Error status ->
    Printf.sprintf "Error in %s: %s" context (Status.message status)
    |> failwith

let () =
  let data = Tensor.create1 Float32 3 in
  Tensor.copy_elt_list data [ 1.; 2.; 6. ];
  let input_tensor = Tensor.P data in
  let graph = Graph.create () in
  let session =
    Session.create graph
    |> ok_exn ~context:"session creation"
  in
  Graph.import
    graph
    (Protobuf.read_file "examples/load/load.pb" |> Protobuf.to_string)
    |> ok_exn ~context:"extending graph";
  let find_operation name =
    match Graph.find_operation graph name with
    | Some operation -> operation
    | None -> failwith (Printf.sprintf "Cannot find op %s" name)
  in
  let output =
    Session.run
      session
      ~inputs:[ Graph.create_port (find_operation "x") ~index:0, input_tensor ]
      ~outputs:[ Graph.create_port (find_operation "add") ~index:0 ]
      ~targets:[ find_operation "add" ]
    |> ok_exn ~context:"session run"
  in
  match output with
  | [ output ] -> Tensor.print output
  | _ -> assert false
