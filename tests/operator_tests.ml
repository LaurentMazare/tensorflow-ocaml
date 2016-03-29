open Core_kernel.Std
open Tensorflow
module O = Ops

let assert_scalar tensor ~expected_value =
  let index =
    match Bigarray.Genarray.dims tensor with
    | [||] -> [||]
    | [| 1 |] -> [| 0 |]
    | [| n |] -> failwithf "Single dimension tensor with %d elements" n ()
    | _ -> failwith "Multi-dimensional tensor."
  in
  let value = Bigarray.Genarray.get tensor index in
  if value <> expected_value
  then failwithf "Got %f, expected %f" value expected_value ()

let test_scalar () =
  List.iter ~f:(fun (ops, expected_value) ->
    let tensor = Session.run (Session.Output.float ops) in
    assert_scalar tensor ~expected_value)
    [ O.(f 40. + f 2.), 42.
    ; O.(f 12. * f 3.), 36.
    ; O.(f 7. / (neg (f 2.))), -3.5
    ; O.(pow (f 2.) (f 10.) - square (f 10.)), 924.
    ]

let () =
  test_scalar ()
