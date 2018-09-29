(* TODO: support checkpointing based on time rather than on number of iterations. *)
open Base

let latest_index_and_filename ~checkpoint_base =
  let dirname = Caml.Filename.dirname checkpoint_base in
  let basename = Caml.Filename.basename checkpoint_base in
  Caml.Sys.readdir dirname
  |> Array.to_list
  |> List.filter_map ~f:(fun filename ->
      match String.chop_prefix filename ~prefix:basename with
      | None -> None
      | Some suffix ->
        try
          Some (Int.of_string suffix, Caml.Filename.concat dirname filename)
        with _ -> None)
  |> List.sort ~compare:Caml.Pervasives.compare
  |> List.last

let loop
      ~start_index
      ~end_index
      ~save_vars_from
      ~checkpoint_base
      ?(checkpoint_every = 100)
      f
  =
  if start_index < 0
  then raise (Invalid_argument (Printf.sprintf "negative start_index %d" start_index));
  let named_vars =
    Var.get_all_vars save_vars_from
    |> List.map ~f:(fun var ->
      "V" ^ (Node.packed_id var |> Node.Id.to_string), var)
  in
  let temp_checkpoint = Caml.Filename.temp_file "tf-ocaml" "ckpt" in
  let save_op = Ops.save ~filename:temp_checkpoint named_vars in
  let latest_index_and_filename = latest_index_and_filename ~checkpoint_base in
  let load_ops =
    Option.map latest_index_and_filename ~f:(fun (latest_index, filename) ->
      Stdio.eprintf "Restoring checkpoint for index %d from %s\n%!" latest_index filename;
      let filename = Ops.const_string0 filename in
      List.map named_vars ~f:(fun (var_name, Node.P var) ->
        Ops.assign
          var
          (Ops.restore ~type_:(Node.output_type var) filename (Ops.const_string0 var_name))
        |> fun node -> Node.P node
      )
    )
  in

  (* From this point, no op should be added to the graph anymore as we may call
     [Session.run]. *)
  Option.iter load_ops ~f:(fun load_ops ->
    Session.run ~targets:load_ops Session.Output.empty);

  let start_index =
    Option.value_map latest_index_and_filename ~default:start_index ~f:fst
  in
  let save ~suffix =
    Session.run ~targets:[Node.P save_op] Session.Output.empty;
    Caml.Sys.rename
      temp_checkpoint
      (Printf.sprintf "%s.%s" checkpoint_base suffix)
  in
  for index = start_index to end_index do
    f ~index;
    if index % checkpoint_every = 0
    then save ~suffix:(Int.to_string index)
  done;
  save ~suffix:"final"
