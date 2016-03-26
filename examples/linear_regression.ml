open Core_kernel.Std

let () =
  let samples = 100 in
  let size_y = 1 in
  let size_xs = 2 in
  let x = List.init samples ~f:(fun i -> 3.14 *. float i /. float samples) in
  let xs = List.map x ~f:(fun x -> [ x; x *. x ]) in
  let y = List.map x ~f:sin in
  ignore (Linear_regression_common.run ~samples ~size_xs ~size_y ~xs ~y)

