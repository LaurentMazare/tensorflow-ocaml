open Tensorflow

let () =
  let forty_two = Ops.(f 40. + f 2.) in
  let v = Session.run (Session.Output.scalar_float forty_two) in
  Printf.printf "%f\n%!" v
