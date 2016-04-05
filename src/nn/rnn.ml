open Core_kernel.Std

let gru ~shape =
  let open Nn.Shared_var in
  let denseR = Staged.unstage (dense ~shape) in
  let denseZ = Staged.unstage (dense ~shape) in
  let denseH = Staged.unstage (dense ~shape) in
  Staged.stage
    (fun ~h ~x ->
       let open Nn in
       let h_and_x = concat h x in
       let rh = sigmoid (denseR h_and_x) * h in
       let rh_and_x = concat rh x in
       (* the new value of h *)
       let nh = tanh (denseH rh_and_x) in
       (* How do we mix th new h and the old h *)
       (* CR noury: check if it is nh_and_x *)
       let z = sigmoid (denseZ h_and_x) in
       (* we mix the old h and the new h *)
       let h = z * nh + (f 1.0 ~shape:(Nn.shape z) - z) * h in
       h)
