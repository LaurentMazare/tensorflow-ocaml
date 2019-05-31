open Base
open Float.O_dot

let () =
  let samples = 100 in
  let size_ys = 1 in
  let size_xs = 2 in
  let x =
    List.init samples ~f:(fun i -> 3.14 *. Float.of_int i /. Float.of_int samples)
  in
  let xs = List.map x ~f:(fun x -> [ x; x *. x ]) in
  let ys = List.map x ~f:(fun x -> [ Float.sin x ]) in
  ignore (Linear_regression_common.run ~samples ~size_xs ~size_ys ~xs ~ys)
