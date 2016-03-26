open Core_kernel.Std
open Gnuplot

let () =
  let samples = 100 in
  let size_y = 1 in
  let size_xs = 2 in
  let x = List.init samples ~f:(fun i -> 3.14 *. float i /. float samples) in
  let xs = List.map x ~f:(fun x -> [ x; x *. x ]) in
  let y = List.map x ~f:sin in
  let ys = Linear_regression_common.run ~samples ~size_xs ~size_y ~xs ~y in
  let gp = Gp.create () in
  let series =
    List.map (List.tl_exn (List.rev ys)) ~f:(fun (n, y) ->
      let title = sprintf "%d epochs" n in
      Series.lines_xy (List.zip_exn x y) ~title)
  in
  Gp.plot_many gp ~output:(Output.create `Qt)
    (Series.points_xy (List.zip_exn x y) ~color:`Blue ~title:"target" :: series);
  Gp.close gp
