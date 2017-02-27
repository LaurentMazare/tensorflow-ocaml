open Core_kernel.Std
open Tensorflow

let epochs = 1000
let learning_rate = 1.
let cpkt_filename = Sys.getcwd () ^ "/vgg19.cpkt"

let conv2d node ~in_channels ~out_channels =
  let w = Var.f_or_d [ 3; 3; in_channels; out_channels ] ~type_:Float 0. in
  let b = Var.f_or_d [ out_channels ] ~type_:Float 0. in
  let conv2d = Ops.conv2D ~strides:[ 1; 1; 1; 1 ] ~padding:"SAME" node w in
  Ops.(+) conv2d b, w, b

let max_pool node =
  Ops.maxPool node ~ksize:[ 1; 2; 2; 1 ] ~strides:[ 1; 2; 2; 1 ] ~padding:"SAME"

let load vars ~filename =
  let filename = Ops.const_string [ filename ] in
  let load_and_assign_nodes =
    List.map vars ~f:(fun (var_name, var) ->
      Ops.restore
        ~type_:(Node.output_type var)
        filename
        (Ops.const_string [ var_name ])
      |> Ops.assign var
      |> fun node -> Node.P node)
  in
  Session.run
    ~inputs:[]
    ~targets:load_and_assign_nodes
    Session.Output.empty

let style_grams_and_content_nodes input ~img_h ~img_w =
  let var_by_name = String.Table.create () in
  let block iter ~block_idx ~in_channels ~out_channels (node, acc) =
    List.init iter ~f:Fn.id
    |> List.fold ~init:(node, []) ~f:(fun (node, acc_relus) idx ->
      let name = sprintf "conv%d_%d" block_idx (idx+1) in
      let in_channels = if idx = 0 then in_channels else out_channels in
      let conv2d, w, b = conv2d node ~in_channels ~out_channels in
      Hashtbl.set var_by_name ~key:(name ^ "/" ^ name ^ "_filters") ~data:w;
      Hashtbl.set var_by_name ~key:(name ^ "/" ^ name ^ "_biases") ~data:b;
      let relu = Ops.relu conv2d in
      relu, (relu, out_channels) :: acc_relus)
    |> fun (node, acc_relus) -> max_pool node, List.rev acc_relus :: acc
  in
  let _model, acc_relus =
    (Ops.reshape input (Ops.const_int ~type_:Int32 [ 1; img_h; img_w; 3 ]), [])
    |> block 2 ~block_idx:1 ~in_channels:3   ~out_channels:64
    |> block 2 ~block_idx:2 ~in_channels:64  ~out_channels:128
    |> block 4 ~block_idx:3 ~in_channels:128 ~out_channels:256
    |> block 4 ~block_idx:4 ~in_channels:256 ~out_channels:512
    |> block 4 ~block_idx:5 ~in_channels:512 ~out_channels:512
  in
  let acc_relus = List.rev acc_relus in
  load (Hashtbl.to_alist var_by_name) ~filename:cpkt_filename;
  let style_grams =
    List.map acc_relus ~f:(fun relus ->
      let node, out_channels = List.hd_exn relus in
      let node = Ops.reshape node (Ops.const_int ~type_:Int32 [ -1; out_channels ]) in
      let size_ = float (out_channels * out_channels) in
      Ops.(matMul ~transpose_a:true node node / f size_))
  in
  let content_nodes =
    (* Block 4, conv 2. *)
    List.map [ 4, 2 ] ~f:(fun (block_idx, conv_idx) ->
      let block = List.nth_exn acc_relus (block_idx - 1) in
      List.nth_exn block (conv_idx - 1) |> fst)
  in
  style_grams, content_nodes

let imagenet_mean = function
  | `blue -> 103.939
  | `green -> 116.779
  | `red -> 123.68

let normalize x ~channel =
  float x -. imagenet_mean channel

let unnormalize x ~channel =
  min 255 (int_of_float (x +. imagenet_mean channel))
  |> max 0

let load_image ~filename ~resize =
  let image = OImages.load filename [] in
  let image = OImages.rgb24 image in
  let width = image # width in
  let height = image # height in
  let img_w, img_h, image =
    match resize with
    | None -> width, height, image
    | Some (img_h, img_w) ->
      let ratio = float img_h /. float img_w in
      let target_height = Float.min (float width *. ratio) (float height) in
      let target_width = target_height /. ratio |> Float.to_int in
      let target_height = Float.to_int target_height in
      let image =
        image # sub
          ((width-target_width) / 2)
          ((height-target_height) / 2)
          target_width
          target_height
      in
      img_w, img_h, image # resize None img_w img_h
  in
  let tensor = Tensor.create3 Float32 img_h img_w 3 in
  for i = 0 to img_h - 1 do
    for j = 0 to img_w - 1 do
      let { Color.r; g; b } = image # get j i in
      Tensor.set tensor [| i; j; 0 |] (normalize b ~channel:`blue);
      Tensor.set tensor [| i; j; 1 |] (normalize g ~channel:`green);
      Tensor.set tensor [| i; j; 2 |] (normalize r ~channel:`red);
    done;
  done;
  tensor

let save_image tensor ~filename ~img_h ~img_w =
  let total_size = Array.fold (Tensor.dims tensor) ~init:1 ~f:( * ) in
  if total_size <> img_h * img_w * 3
  then failwith "Incorrect tensor size";
  let image = new OImages.rgb24 img_w img_h in
  for i = 0 to img_h - 1 do
    for j = 0 to img_w - 1 do
      let b = Tensor.get tensor [| i; j; 0 |] |> unnormalize ~channel:`blue in
      let g = Tensor.get tensor [| i; j; 1 |] |> unnormalize ~channel:`green in
      let r = Tensor.get tensor [| i; j; 2 |] |> unnormalize ~channel:`red in
      image # set j i { Color.r; g; b }
    done;
  done;
  image # save filename None []

let compute_grams ~filename ~img_h ~img_w =
  let input_placeholder = Ops.placeholder ~type_:Float [ img_h; img_w; 3 ] in
  let style_grams, _ =
    style_grams_and_content_nodes (Ops.Placeholder.to_node input_placeholder)
      ~img_h ~img_w
  in
  let input_tensor = load_image ~filename ~resize:(Some (img_h, img_w)) in
  save_image input_tensor ~filename:"cropped.png" ~img_h ~img_w;
  List.map style_grams ~f:(fun node ->
    Session.run
      ~inputs:[ Session.Input.float input_placeholder input_tensor ]
      ~targets:[ Node.P node ]
      (Session.Output.float node))

let total_variation_loss input ~img_h ~img_w =
  let input =
    Ops.reshape input (Ops.const_int ~type_:Int32 [ img_h; img_w; 3 ])
  in
  let axis1_diff =
    Ops.(-)
      (Ops.slice input (Ops.ci32 [ 0; 0; 0 ]) (Ops.ci32 [ img_h-1; img_w; 3 ]))
      (Ops.slice input (Ops.ci32 [ 1; 0; 0 ]) (Ops.ci32 [ img_h-1; img_w; 3 ]))
  in
  let axis2_diff =
    Ops.(-)
      (Ops.slice input (Ops.ci32 [ 0; 0; 0 ]) (Ops.ci32 [ img_h; img_w-1; 3 ]))
      (Ops.slice input (Ops.ci32 [ 0; 1; 0 ]) (Ops.ci32 [ img_h; img_w-1; 3 ]))
  in
  Ops.(reduce_sum (axis1_diff * axis1_diff) + reduce_sum (axis2_diff * axis2_diff))

let create_and_set_var tensor =
  let dims = Tensor.dims tensor |> Array.to_list in
  let input_var = Var.f_or_d dims ~type_:Float 0. in
  let placeholder = Ops.placeholder dims ~type_:Float in
  let assign = Ops.assign input_var (Ops.Placeholder.to_node placeholder) in
  Session.run
    ~inputs:[ Session.Input.float placeholder tensor ]
    ~targets:[ Node.P assign ]
    Session.Output.empty;
  input_var

let () =
  let input_tensor = load_image ~filename:"input.png" ~resize:None in
  let img_h, img_w =
    match Tensor.dims input_tensor with
    | [| img_h; img_w; _ |] -> img_h, img_w
    | _ -> assert false
  in
  printf "Computing target features...\n%!";
  let target_grams = compute_grams ~filename:"style.png" ~img_h ~img_w in
  printf "Done computing target features.\n%!";
  let input_var = create_and_set_var input_tensor in
  let style_grams, content_nodes =
    style_grams_and_content_nodes input_var ~img_h ~img_w
  in
  let style_losses, style_inputs =
    List.map2_exn style_grams target_grams ~f:(fun gram_node target_gram ->
      let dims = Tensor.dims target_gram |> Array.to_list in
      let placeholder = Ops.placeholder ~type_:Float dims in
      let diff = Ops.(-) gram_node (Ops.Placeholder.to_node placeholder) in
      let total_dims = List.reduce_exn dims ~f:( * ) |> float in
      Ops.(reduce_sum (diff * diff) / f total_dims),
      Session.Input.float placeholder target_gram)
    |> List.unzip
  in
  let content_losses, content_inputs =
    List.map content_nodes ~f:(fun content_node ->
      let content_target = Session.run Session.Output.(float content_node) in
      let dims = Tensor.dims content_target |> Array.to_list in
      let placeholder = Ops.placeholder ~type_:Float dims in
      let diff = Ops.(-) content_node (Ops.Placeholder.to_node placeholder) in
      let total_dims = List.reduce_exn dims ~f:( * ) |> float in
      Ops.(reduce_sum (diff * diff) / f total_dims),
      Session.Input.float placeholder content_target)
    |> List.unzip
  in
  let loss =
    Ops.(List.reduce_exn style_losses ~f:(+)
      + List.reduce_exn content_losses ~f:(+)
      + total_variation_loss input_var ~img_h ~img_w)
  in
  let gd =
    Optimizers.adam_minimizer loss
      ~learning_rate:(Ops.f learning_rate)
      ~varsf:[ input_var ]
  in
  for epoch = 1 to epochs do
    let output_tensor, loss =
      Session.run
        ~inputs:(style_inputs @ content_inputs)
        ~targets:gd
        Session.Output.(both (float input_var) (scalar_float loss))
    in
    printf "epoch: %d   loss: %f\n%!" epoch loss;
    save_image output_tensor ~filename:(sprintf "out_%d.png" epoch) ~img_h ~img_w;
  done
