open Tensorflow

let () =
  let forty_two = Ops.(f 40. + f 2.) in
  let tensor = Session.run (Session.Output.float forty_two) in
  Tensor.print (Tensor.P tensor)
