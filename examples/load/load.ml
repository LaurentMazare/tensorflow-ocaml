open Tensorflow_core
open Wrapper
module Graph = Wrapper.Graph
module Session = Wrapper.Session

let ok_exn (result : 'a Status.result) ~context =
  match result with
  | Ok result -> result
  | Error status ->
    Printf.sprintf "Error in %s: %s" context (Status.message status)
    |> failwith

let () =
  let data = Tensor.create3 Float32 1 2 3 in
  Tensor.copy_elt_list data [ 1.; 2.; 6.; 1.; 2.; 6. ];
  let input_tensor = Tensor.P data in
  let graph = Graph.create () in
  let session =
    Session.create graph
    |> ok_exn ~context:"session creation"
  in
  Graph.import
    graph
    (Protobuf.read_file "examples/load/lstm.pb" |> Protobuf.to_string)
    |> ok_exn ~context:"extending graph";
  let find_operation name =
    match Graph.find_operation graph name with
    | Some operation -> operation
    | None -> failwith (Printf.sprintf "Cannot find op %s" name)
  in
  let output =
    Session.run
      session
      ~inputs:[]
      ~outputs:[]
      ~targets:
        [ find_operation "dense_1_b/Assign"
        ; find_operation "dense_1_W/Assign"
        ; find_operation "lstm_1_W/Assign"
        ; find_operation "lstm_1_b/Assign"
        ; find_operation "lstm_1_U/Assign"
        ]
    |> ok_exn ~context:"session run"
  in
  begin
    match output with
    | [] -> ()
    | _ -> assert false
  end;
  let output =
    Session.run
      session
      ~inputs:[ Graph.create_output (find_operation "lstm_input_1") ~index:0, input_tensor ]
      ~outputs:[ Graph.create_output (find_operation "Sigmoid") ~index:0 ]
      ~targets:[ find_operation "Variable/Assign" ]
    |> ok_exn ~context:"session run"
  in
  begin
    match output with
    | [ output ] -> Tensor.print output
    | _ -> assert false
  end
