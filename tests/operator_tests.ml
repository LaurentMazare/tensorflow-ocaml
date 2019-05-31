open Base
open Tensorflow_core
open Tensorflow
module O = Ops

let%expect_test _ =
  let eval_and_print ops =
    Session.run (Session.Output.float ops) |> Tensor.print_;
    Caml.Format.print_flush ()
  in
  eval_and_print O.(f 40. + f 2.);
  [%expect {| 42.000000 |}];
  eval_and_print O.(f 12. * f 3.);
  [%expect {| 36.000000 |}];
  eval_and_print O.(f 7. / neg (f 2.));
  [%expect {| -3.500000 |}];
  eval_and_print O.(pow (f 2.) (f 10.) - square (f 10.));
  [%expect {| 924.000000 |}];
  eval_and_print O.(sin (f 1.) + cos (f 2.) - tanh (f 3.));
  [%expect {| -0.569731 |}];
  eval_and_print O.(reduce_mean (pow (cf (List.init 100 ~f:Float.of_int)) (f 3.)));
  [%expect {| 245025.000000 |}];
  eval_and_print
    O.(reduce_sum (range (const_int ~shape:[] ~type_:Int32 [ 10 ]) |> cast ~type_:Float));
  [%expect {| 45.000000 |}];
  eval_and_print O.(maximum (f 2.) (f 3.) - minimum (f 5.) (f (-5.)));
  [%expect {|
    8.000000
    |}];
  eval_and_print O.(matrixDeterminant (cf ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ]));
  [%expect {| -2.000000 |}];
  eval_and_print O.(moments (cf ~shape:[ 5 ] [ 1.; 2.; 3.; 4.; 5. ]) ~dims:[ 0 ]).mean;
  [%expect {| 3.000000 |}];
  eval_and_print
    O.(moments (cf ~shape:[ 5 ] [ 1.; 2.; 3.; 4.; 5. ]) ~dims:[ 0 ]).variance;
  [%expect {| 2.000000 |}]

let%expect_test _ =
  let eval_and_print ops = Session.run (Session.Output.double ops) |> Tensor.print_ in
  eval_and_print O.(range (const_int ~shape:[] ~type_:Int32 [ 3 ]) |> cast ~type_:Double);
  [%expect {|
    0 0.000000
    1 1.000000
    2 2.000000 |}];
  eval_and_print
    O.(
      concat
        zero32
        [ floor (cd [ 1.0; 1.1; 1.9; 2.0 ]); ceil (cd [ 1.0; 1.1; 1.9; 2.0 ]) ]);
  [%expect
    {|
    0 1.000000
    1 1.000000
    2 1.000000
    3 2.000000
    4 1.000000
    5 2.000000
    6 2.000000
    7 2.000000
    |}];
  eval_and_print
    O.(
      cd ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ] *^ cd ~shape:[ 2; 1 ] [ 5.; 6. ]
      |> reduce_sum ~dims:[ 1 ]);
  [%expect {|
    0 17.000000
    1 39.000000
    |}];
  eval_and_print
    O.(
      matrixSolve
        (cd ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ])
        (cd ~shape:[ 2; 1 ] [ 4.; 10. ])
      |> reduce_sum ~dims:[ 1 ]);
  [%expect {|
    0 2.000000
    1 1.000000
    |}];
  eval_and_print
    O.(
      split ~num_split:2 one32 (cd ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ])
      |> function
      | [ o1; o2 ] ->
        concat
          zero32
          [ reshape o2 (const_int ~type_:Int32 [ 2 ])
          ; reshape o1 (const_int ~type_:Int32 [ 2 ])
          ]
      | _ -> assert false);
  [%expect {|
    0 2.000000
    1 4.000000
    2 1.000000
    3 3.000000
    |}];
  eval_and_print
    O.(
      split ~num_split:2 zero32 (cd ~shape:[ 2; 2 ] [ 1.; 2.; 3.; 4. ])
      |> function
      | [ o1; o2 ] ->
        concat
          zero32
          [ reshape o2 (const_int ~type_:Int32 [ 2 ])
          ; reshape o1 (const_int ~type_:Int32 [ 2 ])
          ]
      | _ -> assert false);
  [%expect {|
    0 3.000000
    1 4.000000
    2 1.000000
    3 2.000000
    |}];
  eval_and_print
    O.(
      moments
        (cd ~shape:[ 2; 5 ] [ 1.; 2.; 3.; 4.; 5.; 8.; 10.; 8.; 10.; 9. ])
        ~dims:[ 1 ])
      .mean;
  [%expect {|
    0 3.000000
    1 9.000000
    |}];
  eval_and_print
    O.(
      moments
        (cd ~shape:[ 2; 5 ] [ 1.; 2.; 3.; 4.; 5.; 8.; 10.; 8.; 10.; 9. ])
        ~dims:[ 1 ])
      .variance;
  [%expect {|
    0 2.000000
    1 0.800000
    |}]

let%expect_test _ =
  let batch = Ops.placeholder ~type_:Double [ 3; 4 ] in
  let is_training = Ops.placeholder ~type_:Bool [ 1 ] in
  let update_ops_store = Layer.Update_ops_store.create () in
  let ops =
    Layer.batch_norm
      (Ops.Placeholder.to_node batch)
      ~is_training:(Ops.Placeholder.to_node is_training)
      ~update_ops_store
      ~decay:0.5
  in
  let ops = Ops.reduce_sum ops ~dims:[ 0 ] in
  let batch_tensor = Tensor.create2 Float64 3 4 in
  let is_training_tensor = Tensor.create0 Int8_unsigned in
  Tensor.copy_elt_list is_training_tensor [ 1 ];
  for i = 0 to 4 do
    let training = i < 3 in
    Tensor.copy_elt_list is_training_tensor [ (if training then 1 else 0) ];
    let tensor =
      Tensor.copy_elt_list batch_tensor [ 0.; 4.; 0.; 8.; 0.; 4.; 9.; 8.; 0.; 4.; 3.; 3. ];
      Session.run
        Session.Output.(double ops)
        ~inputs:
          [ Session.Input.double batch batch_tensor
          ; Session.Input.bool is_training is_training_tensor
          ]
    in
    Tensor.print_ tensor;
    if training
    then
      Session.run
        ~targets:(Layer.Update_ops_store.ops update_ops_store)
        ~inputs:
          [ Session.Input.double batch batch_tensor
          ; Session.Input.bool is_training is_training_tensor
          ]
        Session.Output.empty
  done;
  [%expect
    {|
    0 0.000000
    1 0.000000
    2 0.000000
    3 0.000000
    0 0.000000
    1 0.000000
    2 0.000000
    3 0.000000
    0 0.000000
    1 0.000000
    2 0.000000
    3 0.000000
    0 0.000000
    1 4.242641
    2 0.426401
    3 1.063611
    0 0.000000
    1 4.242641
    2 0.426401
    3 1.063611
    |}]

let%expect_test _ =
  let run true_false =
    let testing = Ops.placeholder ~type_:Bool [] in
    let true_false = if true_false then 1 else 0 in
    let int32_with_control_inputs ~control_inputs v =
      Ops.const_int ~shape:[] ~type_:Int32 ~control_inputs [ v ]
    in
    let cond =
      Ops.cond_with_control_inputs
        (Ops.Placeholder.to_node testing)
        ~if_true:(int32_with_control_inputs 1)
        ~if_false:(int32_with_control_inputs 0)
    in
    let testing_tensor = Tensor.create0 Int8_unsigned in
    Tensor.copy_elt_list testing_tensor [ true_false ];
    let tensor =
      Session.run
        Session.Output.(int32 cond)
        ~inputs:[ Session.Input.bool testing testing_tensor ]
    in
    let index =
      match Tensor.dims tensor with
      | [||] -> [||]
      | [| 1 |] -> [| 0 |]
      | [| n |] -> Printf.failwithf "Single dimension tensor with %d elements" n ()
      | _ -> failwith "Multi-dimensional tensor."
    in
    let value = Tensor.get tensor index |> Int32.to_int_exn in
    Stdio.printf "%d\n" value
  in
  run true;
  [%expect {| 1 |}];
  run false;
  [%expect {| 0 |}]
