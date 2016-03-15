open Wrapper
module CArray = Ctypes.CArray

let () =
  let graph =
    Graph.add
      (Graph.const [ 4. ])
      (Graph.const [ 38. ])
  in
  let session_options = Session_options.create () in
  let status = Status.create () in
  let session = Session.create session_options status in
  Printf.printf "%d %s\n%!" (Status.code status) (Status.message status);
  Session.extend_graph
    session
    (Graph.Protobuf.to_string graph)
    status;
  Printf.printf "%d %s\n%!" (Status.code status) (Status.message status);
  let output =
    Session.run
      session
      ~inputs:[]
      ~outputs:[ Graph.name graph ]
      ~targets:[ Graph.name graph ]
  in
  match output with
  | `Ok [ output_tensor ] ->
    let data = Tensor.data output_tensor Ctypes.float 1 in
    Printf.printf "%f\n%!" (CArray.get data 0)
  | `Ok ([] | _ :: _ :: _) -> assert false
  | `Error error -> Printf.printf "Error: %s\n%!" error

