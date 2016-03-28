open Core_kernel.Std
open Tf_ocaml
open Wrapper

let ok_exn (result : 'a Session.result) ~context =
  match result with
  | Ok result -> result
  | Error status ->
    Printf.sprintf "Error in %s: %s" context (Status.message status)
    |> failwith

let print_tensors tensors ~names =
  List.zip_exn names tensors
  |> List.iter ~f:(fun (name, tensor) ->
      Printf.printf "%s:\n%!" name;
      Tensor.print tensor)

let run ?(inputs = []) ?(outputs = []) ?(targets = []) session =
  let f n = Node.Name.to_string n.Node.name in
  Session.run
    session
    ~inputs:(List.map inputs ~f:(fun (name, tensor) -> f name, tensor))
    ~outputs:(List.map outputs ~f)
    ~targets:(List.map targets ~f)
  |> ok_exn ~context:"session_run"

let create_session nodes =
  let session =
    Session.create ()
    |> ok_exn ~context:"session creation"
  in
  Session.extend_graph
    session
    (Node_protobuf.of_nodes nodes)
  |> ok_exn ~context:"extending graph";
  session
