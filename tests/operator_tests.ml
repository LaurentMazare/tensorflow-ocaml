open Core_kernel.Std
open Tensorflow
module O = Ops

let assert_scalar tensor ~expected_value ~tol =
  let index =
    match Bigarray.Genarray.dims tensor with
    | [||] -> [||]
    | [| 1 |] -> [| 0 |]
    | [| n |] -> failwithf "Single dimension tensor with %d elements" n ()
    | _ -> failwith "Multi-dimensional tensor."
  in
  let value = Bigarray.Genarray.get tensor index in
  if Float.abs (value -. expected_value) > tol
  then failwithf "Got %f, expected %f" value expected_value ()

let test_scalar () =
  List.iter ~f:(fun (tol, ops, expected_value) ->
    let tensor = Session.run (Session.Output.float ops) in
    assert_scalar tensor ~expected_value ~tol)
    [ 0.,   O.(f 40. + f 2.), 42.
    ; 0.,   O.(f 12. * f 3.), 36.
    ; 0.,   O.(f 7. / (neg (f 2.))), -3.5
    ; 0.,   O.(pow (f 2.) (f 10.) - square (f 10.)), 924.
    ; 1e-7, O.(sin (f 1.) + cos (f 2.) - tanh (f 3.)), sin 1. +. cos 2. -. tanh 3.
    ; 0.,   O.(reduce_mean (pow (cf (List.init 100 ~f:float)) (f 3.))), 245025.
    ; 0.,   O.(reduce_mean (range (const_int ~shape:[] ~type_:Int32 [ 10 ])
              |> cast ~type_:Float)), 4.5
    ]

let () =
  test_scalar ()
