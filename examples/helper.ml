open Core.Std
open Wrapper
module CArray = Ctypes.CArray

(* TODO: move the interesting functions to src/. *)
let ok_exn (result : 'a Session.result) ~context =
  match result with
  | Ok result -> result
  | Error status ->
    Printf.sprintf "Error in %s: %s" context (Status.message status)
    |> failwith

let const_float shape f =
  let size = List.fold shape ~init:1 ~f:(( * )) in
  Ops_m.const_float
    ~type_:Float
    ~shape
    (List.init size ~f:(const f))

let print_one_tensor (name, tensor) =
  Printf.printf "%s:\n%!" name;
  match Tensor.num_dims tensor with
  | 1 ->
    let dim = Tensor.dim tensor 0 in
    let data = Tensor.data tensor Ctypes.float dim in
    for d = 0 to dim - 1 do
      Printf.printf "%d %f\n%!" d (CArray.get data d)
    done
  | 2 ->
    let d0 = Tensor.dim tensor 0 in
    let d1 = Tensor.dim tensor 1 in
    let data = Tensor.data tensor Ctypes.float (d0 * d1) in
    for x = 0 to d0 - 1 do
      Printf.printf "%d " x;
      for y = 0 to d1 - 1 do
        Printf.printf "%f " (CArray.get data (x+d0*y))
      done;
      Printf.printf "\n%!";
    done
  | n -> Printf.printf "%d dims\n%!" n

let print_tensors tensors ~names =
  List.zip_exn names tensors
  |> List.iter ~f:print_one_tensor

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
