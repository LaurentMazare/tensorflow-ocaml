open Base
open Float.O_dot
open Tensorflow_core
open Tensorflow
module O = Ops

let assert_equal value ~expected_value ~tol =
  if Float.(abs (value - expected_value) > tol)
  then Printf.failwithf "Got %f, expected %f" value expected_value ()

let assert_scalar tensor ~expected_value ~tol =
  let index =
    match Tensor.dims tensor with
    | [||] -> [||]
    | [| 1 |] -> [| 0 |]
    | [| n |] -> Printf.failwithf "Single dimension tensor with %d elements" n ()
    | _ -> failwith "Multi-dimensional tensor."
  in
  assert_equal (Tensor.get tensor index) ~expected_value ~tol

let test_scalar kind =
  let concat_or_pack =
    match kind with
    | `caml -> fun l -> O.concat O.zero32 l
    | `tf -> fun l -> O.pack l
  in
  List.iter
    ~f:(fun (tol, ops, x, expected_value, expected_gradient) ->
      let var = Var.create [ 1 ] ~type_:Float ~init:(O.const_float ~type_:Float [ x ]) in
      let ops = ops var in
      let gradient_f, gradient_d =
        match kind with
        | `caml ->
          Gradients.gradient_caml
            ops
            ~with_respect_to_float:[ var ]
            ~with_respect_to_double:[]
        | `tf ->
          Gradients.gradient_tf
            ops
            ~with_respect_to_float:[ var ]
            ~with_respect_to_double:[]
      in
      assert (List.is_empty gradient_d);
      let gradient =
        match gradient_f with
        | [] | _ :: _ :: _ -> assert false
        | [ gradient ] -> gradient
      in
      let tensor_value, tensor_gradient =
        Session.run Session.Output.(both (float ops) (float gradient))
      in
      assert_scalar tensor_value ~expected_value ~tol;
      assert_scalar tensor_gradient ~expected_value:expected_gradient ~tol)
    [ 0., (fun v -> O.((v * v) + (f 3. * v) + v)), 5., 45., 14.
    ; 0., (fun v -> O.(f 1. / v)), 2., 0.5, -0.25
    ; 0., (fun v -> O.(f 1. / (v * v * v))), 2., 0.125, -3. /. 16.
    ; ( 0.
      , (fun v -> O.(reduce_sum (concat_or_pack [ v * v; v; f 1. / v ])))
      , 2.
      , 6.5
      , 4.75 )
    ; 0., (fun v -> O.(reduce_sum (v + f ~shape:[ 5 ] 1.))), 2., 15., 5.
    ; 0., (fun v -> O.(reduce_sum (v * f ~shape:[ 5 ] 2.))), 2., 20., 10.
    ; 0., (fun v -> O.(reduce_sum (v * v * f ~shape:[ 5 ] 2.))), 2., 40., 40.
    ]

let () =
  test_scalar `caml;
  test_scalar `tf
