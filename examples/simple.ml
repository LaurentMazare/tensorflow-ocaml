open Wrapper
module CArray = Ctypes.CArray

let ok_exn (result : 'a Session.result) ~context =
  match result with
  | Ok result -> result
  | Error status ->
    Printf.sprintf "Error in %s: %s" context (Status.message status)
    |> failwith

let () =
  let input_tensor = Tensor.create1d Ctypes.float 3 in
  let data = Tensor.data input_tensor Ctypes.float 3 in
  CArray.set data 0 1.;
  CArray.set data 1 2.;
  CArray.set data 2 6.;
  let placeholder = Ops.placeholder ~name:"x" ~type_:Float () in
  let node =
    Ops.sub
      (Ops_m.const_float ~type_:Float [ 2.; 1.; 4. ])
      placeholder
    |> Ops.abs
  in
  let session_options = Session_options.create () in
  let session =
    Session.create session_options
    |> ok_exn ~context:"session creation"
  in
  Session.extend_graph
    session
    (Protobuf.of_node node)
    |> ok_exn ~context:"extending graph";
  let output =
    Session.run
      session
      ~inputs:[ placeholder.name |> Node.Name.to_string, input_tensor ]
      ~outputs:[ node.name |> Node.Name.to_string ]
      ~targets:[ node.name |> Node.Name.to_string ]
    |> ok_exn ~context:"session run"
  in
  match output with
  | [ output_tensor ] ->
    let dim = Tensor.dim output_tensor 0 in
    let data = Tensor.data output_tensor Ctypes.float dim in
    for d = 0 to dim - 1 do
      Printf.printf "%d %f\n%!" d (CArray.get data d)
    done
  | [] | _ :: _ :: _ -> assert false
