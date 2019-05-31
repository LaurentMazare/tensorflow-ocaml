open Base
open Gnuplot

let () =
  let samples = 100 in
  let size_ys = 1 in
  let size_xs = 2 in
  let x = List.init samples ~f:(fun i -> 3.14 *. float i /. float samples) in
  let xs = List.map x ~f:(fun x -> [ x; x *. x ]) in
  let ys = List.map x ~f:(fun x -> [ sin x ]) in
  let trained_ys =
    Nn_common.one_layer ~samples ~size_xs ~size_ys ~xs ~ys ~epochs:10000 ~hidden_nodes:2
  in
  let gp = Gp.create () in
  let series =
    List.map
      (List.tl_exn (List.rev trained_ys))
      ~f:(fun (n, y) ->
        let title = sprintf "%d epochs" n in
        Series.lines_xy (List.zip_exn x y) ~title)
  in
  Gp.plot_many
    gp
    ~output:(Output.create `Qt)
    (Series.points_xy (List.zip_exn x (List.concat ys)) ~color:`Blue ~title:"target"
    :: series);
  Gp.close gp
