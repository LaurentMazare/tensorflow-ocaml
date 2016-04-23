open Core_kernel.Std
open Tensorflow
module O = Ops

let assert_equal value ~expected_value ~tol =
  if Float.abs (value -. expected_value) > tol
  then failwithf "Got %f, expected %f" value expected_value ()

let assert_scalar tensor ~expected_value ~tol =
  let index =
    match Tensor.dims tensor with
    | [||] -> [||]
    | [| 1 |] -> [| 0 |]
    | [| n |] -> failwithf "Single dimension tensor with %d elements" n ()
    | _ -> failwith "Multi-dimensional tensor."
  in
  assert_equal (Tensor.get tensor index) ~expected_value ~tol

let assert_vector tensor ~expected_value ~tol =
  match Tensor.dims tensor with
  | [||] -> failwith "Scalar rather than vector"
  | [| n |] ->
    List.init n ~f:(fun i -> Tensor.get tensor [| i |])
    |> List.iter2_exn expected_value ~f:(fun expected_value value ->
      assert_equal value ~expected_value ~tol)
  | _ -> failwith "Multi-dimensional tensor."

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
    ; 0.,   O.(reduce_sum (range (const_int ~shape:[] ~type_:Int32 [ 10 ])
              |> cast ~type_:Float)), 45.
    ; 0.,   O.(maximum (f 2.) (f 3.) - minimum (f 5.) (f (-5.))), 8.
    ; 1e-6, O.(matrixDeterminant (cf ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ])), -2.
    ]

let test_vector () =
  List.iter ~f:(fun (tol, ops, expected_value) ->
    let tensor = Session.run (Session.Output.double ops) in
    assert_vector tensor ~expected_value ~tol)
    [ 0., O.(range (const_int ~shape:[] ~type_:Int32 [ 3 ]) |> cast ~type_:Double), [ 0.; 1.; 2. ]
    ; 0., O.(concat zero32
        [ floor (cd [ 1.0; 1.1; 1.9; 2.0 ])
        ; ceil (cd [ 1.0; 1.1; 1.9; 2.0 ])
        ])
      , [ 1.; 1.; 1.; 2.; 1.; 2.; 2.; 2. ]
    ; 0., O.((cd ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ]) *^ (cd ~shape:[ 2; 1 ] [ 5.; 6. ])
            |> reduce_sum ~dims:[ 1 ])
      , [ 17.; 39. ]
    ; 1e-8, O.(matrixSolve (cd ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ])
                           (cd ~shape:[ 2; 1 ] [ 4.; 10. ])
            |> reduce_sum ~dims:[ 1 ])
      , [ 2.; 1. ]
    ; 0., O.(split ~num_split:2 one32 (cd ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ])
          |> function
          | [ o1; o2 ] ->
            concat zero32
              [ reshape o2 (const_int ~type_:Int32 [ 2 ])
              ; reshape o1 (const_int ~type_:Int32 [ 2 ])
              ]
          | _ -> assert false)
      , [ 2.; 4.; 1.; 3. ]
    ; 0., O.(split ~num_split:2 zero32 (cd ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ])
          |> function
          | [ o1; o2 ] ->
            concat zero32
              [ reshape o2 (const_int ~type_:Int32 [ 2 ])
              ; reshape o1 (const_int ~type_:Int32 [ 2 ])
              ]
          | _ -> assert false)
      , [ 3.; 4.; 1.; 2. ]
    ]

let () =
  test_scalar ();
  test_vector ()
