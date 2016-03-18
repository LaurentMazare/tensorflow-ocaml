open Wrapper
module CArray = Ctypes.CArray

let () =
  let node =
    Ops.add
      (Ops_m.const_float_1d ~type_:Float [ 2.; 9.; 12. ])
      (Ops_m.const_float_1d ~type_:Float [ 1.; 3.; 5. ])
  in
  let session_options = Session_options.create () in
  let status = Status.create () in
  let session = Session.create session_options status in
  assert (Status.code status = TF_OK);
  Session.extend_graph
    session
    (Protobuf.of_node (P node))
    status;
  begin
    match Status.code status with
    | TF_OK -> ()
    | _ ->
      Printf.sprintf "Error building graph: %s\n%!" (Status.message status)
      |> failwith
  end;
  let output =
    Session.run
      session
      ~inputs:[]
      ~outputs:[ node.name |> Node.Name.to_string ]
      ~targets:[ node.name |> Node.Name.to_string ]
  in
  match output with
  | `Ok [ output_tensor ] ->
    let data = Tensor.data output_tensor Ctypes.float 2 in
    Printf.printf "%f %f\n%!" (CArray.get data 0) (CArray.get data 1)
  | `Ok ([] | _ :: _ :: _) -> assert false
  | `Error (_code, error) ->
      Printf.printf "Error: %s\n%!" error


