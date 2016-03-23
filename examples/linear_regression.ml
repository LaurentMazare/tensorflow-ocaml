open Core.Std
open Wrapper
module CArray = Ctypes.CArray

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
  let dim = Tensor.dim tensor 0 in
  let data = Tensor.data tensor Ctypes.float dim in
  for d = 0 to dim - 1 do
    Printf.printf "%d %f\n%!" d (CArray.get data d)
  done

let print_tensors tensors ~names =
  List.zip_exn names tensors
  |> List.iter ~f:print_one_tensor

let run session ~inputs ~outputs ~targets =
  let f n = Node.Name.to_string n.Node.name in
  Session.run
    session
    ~inputs:(List.map inputs ~f:(fun (name, tensor) -> f name, tensor))
    ~outputs:(List.map outputs ~f)
    ~targets:(List.map targets ~f)
  |> ok_exn ~context:"session_run"

let () =
  let n = 3 in (* size of y *)
  let m = 2 in (* size of x *)
  let x_tensor = Tensor.create2d Ctypes.float 1 m in
  let x_data = Tensor.data x_tensor Ctypes.float m in
  for i = 0 to m - 1 do
    CArray.set x_data i (float (i+1));
  done;
  let y_tensor = Tensor.create2d Ctypes.float 1 n in
  let y_data = Tensor.data y_tensor Ctypes.float n in
  for i = 0 to n - 1 do
    CArray.set y_data i (float (4*(i+1)));
  done;
  let x = Ops.placeholder ~name:"x" ~type_:Float () in
  let y = Ops.placeholder ~name:"y" ~type_:Float () in
  let w =
    Ops.variable ()
      ~type_:Float
      ~shape:[ { size = m; name = None }; { size = n; name = None } ]
  in
  let b =
    Ops.variable ()
      ~type_:Float
      ~shape:[ { size = n; name = None } ]
  in
  let w_assign = Ops.assign w (const_float [ m; n ] 0.) in
  let b_assign = Ops.assign b (const_float [ n ] 0.) in
  let diff = Ops.matMul x w |> Ops.add b |> Ops.sub y in
  let err = Ops.matMul diff diff ~transpose_b:true in
  let session =
    Session.create ()
    |> ok_exn ~context:"session creation"
  in
  Session.extend_graph
    session
    (Node_protobuf.of_nodes [ P err; P w_assign; P b_assign ])
    |> ok_exn ~context:"extending graph";
  let output =
    run session
      ~inputs:[]
      ~outputs:[ w_assign ]
      ~targets:[ w_assign; b_assign ] 
  in
  print_tensors output ~names:[ "init" ];
  let output =
    run session
      ~inputs:[]
      ~outputs:[ w; b ]
      ~targets:[ w; b ]
  in
  print_tensors output ~names:[ "w"; "b" ];
  let output =
    run session
      ~inputs:[ x, x_tensor; y, y_tensor ]
      ~outputs:[ err ]
      ~targets:[ err ]
  in
  print_tensors output ~names:[ "err" ]

