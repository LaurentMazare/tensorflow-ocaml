open Core_kernel.Std

let () =
  let samples = 100 in
  let size_ys = 1 in
  let size_xs = 2 in
  let x = List.init samples ~f:(fun i -> 3.14 *. float i /. float samples) in
  let xs = List.map x ~f:(fun x -> [ x; x *. x ]) in
  let ys = List.map x ~f:(fun x -> [ sin x ]) in
  ignore (Linear_regression_common.run ~samples ~size_xs ~size_ys ~xs ~ys)

