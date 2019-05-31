open Tensorflow_core
open Tensorflow

let filename = "./mytensors"

let () =
  let pi = Ops.cf [ 3.; 1.; 4.; 1.; 5.; 9. ] in
  let e = Ops.cf [ 2.; 7.; 1.; 8.; 2.; 8. ] in
  (* Save the two tensors in the same file. *)
  Session.run
    ~targets:[ Node.P (Ops.save ~filename [ "pi", Node.P pi; "e", Node.P e ]) ]
    Session.Output.empty;
  let var_pi = Var.load_f [ 6 ] ~filename ~tensor:"pi" in
  let var_e = Var.load_f [ 6 ] ~filename ~tensor:"e" in
  let tensor_pi, tensor_e =
    Session.run Session.Output.(both (float var_pi) (float var_e))
  in
  Tensor.print (Tensor.P tensor_pi);
  Tensor.print (Tensor.P tensor_e)
