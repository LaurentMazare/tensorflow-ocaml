open Core_kernel.Std
open Tensorflow

let img_size = 224
let epochs = 10
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
    Hashtbl.to_alist vars
    |> List.map ~f:(fun (var_name, var) ->
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

let style_grams input =
  let style_layer_nodes = ref [] in
  let var_by_name = String.Table.create () in
  let block iter ~block_idx ~in_channels ~out_channels node =
    List.init iter ~f:Fn.id
    |> List.fold ~init:node ~f:(fun acc idx ->
      let name = sprintf "conv%d_%d" block_idx (idx+1) in
      let in_channels =
        if idx = 0
        then in_channels
        else out_channels
      in
      let conv2d, w, b = conv2d acc ~in_channels ~out_channels in
      Hashtbl.set var_by_name ~key:(name ^ "/" ^ name ^ "_filters") ~data:w;
      Hashtbl.set var_by_name ~key:(name ^ "/" ^ name ^ "_biases") ~data:b;
      let relu = Ops.relu conv2d in
      if idx = 0
      then style_layer_nodes := relu :: !style_layer_nodes;
      relu)
    |> max_pool
  in
  let _model =
    input
    |> block 2 ~block_idx:1 ~in_channels:3   ~out_channels:64
    |> block 2 ~block_idx:2 ~in_channels:64  ~out_channels:128
    |> block 4 ~block_idx:3 ~in_channels:128 ~out_channels:256
    |> block 4 ~block_idx:4 ~in_channels:256 ~out_channels:512
    |> block 4 ~block_idx:5 ~in_channels:512 ~out_channels:512
  in
  load var_by_name ~filename:cpkt_filename;
  List.map !style_layer_nodes ~f:(fun node ->
    let node = Ops.reshape node (Ops.const_int ~type_:Int32 [ -1 ]) in
    Ops.(matMul ~transpose_a:true node node))

let imagenet_mean = function
  | `blue -> 103.939
  | `green -> 116.779
  | `red -> 123.68

let normalize x ~channel =
  float x -. imagenet_mean channel

let unnormalize x ~channel =
  min 255 (int_of_float (x +. imagenet_mean channel))

let load_image ~filename =
  let image = OImages.load filename [] in
  let image = OImages.rgb24 image in
  let width = image # width in
  let height = image # height in
  let min_edge = min width height in
  let image =
    image # sub
      ((width-min_edge) / 2)
      ((height-min_edge) / 2)
      min_edge
      min_edge
  in
  let image = image # resize None img_size img_size in
  let tensor = Tensor.create3 Float32 img_size img_size 3 in
  for i = 0 to img_size - 1 do
    for j = 0 to img_size - 1 do
      let { Color.r; g; b } = image # get j i in
      Tensor.set tensor [| i; j; 0 |] (normalize b ~channel:`blue);
      Tensor.set tensor [| i; j; 1 |] (normalize g ~channel:`green);
      Tensor.set tensor [| i; j; 2 |] (normalize r ~channel:`red);
    done;
  done;
  tensor

let save_image tensor ~filename =
  let total_size = Array.fold (Tensor.dims tensor) ~init:1 ~f:( * ) in
  if total_size <> img_size * img_size * 3
  then failwith "Incorrect tensor size";
  let image = new OImages.rgb24 img_size img_size in
  for i = 0 to img_size - 1 do
    for j = 0 to img_size - 1 do
      let b = Tensor.get tensor [| i; j; 0 |] |> unnormalize ~channel:`blue in
      let g = Tensor.get tensor [| i; j; 1 |] |> unnormalize ~channel:`green in
      let r = Tensor.get tensor [| i; j; 2 |] |> unnormalize ~channel:`red in
      image # set j i { Color.r; g; b }
    done;
  done;
  image # save filename None []

let compute_grams ~filename =
  let input_placeholder = Ops.placeholder ~type_:Float [ img_size; img_size; 3 ] in
  let style_grams = style_grams (Ops.Placeholder.to_node input_placeholder) in
  let input_tensor = load_image ~filename in
  save_image input_tensor ~filename:"cropped.jpg";
  List.map style_grams ~f:(fun node ->
    Session.run
      ~inputs:[ Session.Input.float input_placeholder input_tensor ]
      (Session.Output.return node))

let () =
  let target_grams = compute_grams ~filename:"style.jpg" in
  let input_var = Var.f_or_d [ img_size; img_size; 3 ] ~type_:Float 0. in
  let style_loss =
    List.map2_exn (style_grams input_var) target_grams ~f:(fun gram_node target_gram ->
      let diff = Ops.(-) gram_node target_gram in
      Ops.(diff * diff)
    )
    |> List.reduce_exn ~f:Ops.(+)
  in
  let gd = Optimizers.adam_minimizer ~learning_rate:(Ops.f 1e-3) style_loss in
  for epoch = 1 to epochs do
    let style_loss =
      Session.run
        ~inputs:[]
        ~targets:gd
        Session.Output.(scalar_float style_loss)
    in
    printf "epoch: %d   loss: %f\n%!" epoch style_loss
  done
