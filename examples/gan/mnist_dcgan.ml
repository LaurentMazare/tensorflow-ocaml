(* Deep Convolutional Generative Adverserial Networks trained on the MNIST dataset. *)
open Base
open Tensorflow
open Tensorflow_core
module O = Ops

let image_dim = Mnist_helper.image_dim
let latent_dim = 100
let batch_size = 256
let learning_rate = 1e-5
let batches = 10 ** 8

(* No need to keep running averages as this is only used in training mode. *)
let batch_norm xs =
  let nb_dims = Node.shape xs |> List.length in
  let batch_moments = Ops.moments xs ~dims:(List.init (nb_dims - 1) ~f:Fn.id) in
  Ops.normalize xs batch_moments ~epsilon:1e-8

(** [create_generator rand_input] creates a Generator network taking as
    input [rand_input]. This returns both the network and the variables
    that it contains.
*)
let create_generator rand_input =
  let linear1 = Layer.Linear.create 1024 in
  let linear2 = Layer.Linear.create (7 * 7 * 128) in
  let conv2dt1 =
    Layer.Conv2DTranspose.create ~ksize:(5, 5) ~strides:(2, 2) ~padding:Same 64
  in
  let conv2dt2 =
    Layer.Conv2DTranspose.create ~ksize:(5, 5) ~strides:(2, 2) ~padding:Same 1
  in
  let output =
    Layer.Linear.apply linear1 rand_input ~activation:Relu
    |> Layer.Linear.apply linear2
    |> batch_norm
    |> O.relu
    |> Layer.reshape ~shape:[ -1; 7; 7; 128 ]
    |> Layer.Conv2DTranspose.apply conv2dt1
    |> batch_norm
    |> O.relu
    |> Layer.Conv2DTranspose.apply conv2dt2
    |> Layer.flatten
    |> O.tanh
  in
  let vars =
    List.concat_map [ linear1; linear2 ] ~f:Layer.Linear.vars
    @ List.concat_map [ conv2dt1; conv2dt2 ] ~f:Layer.Conv2DTranspose.vars
  in
  output, vars

(** [create_discriminator xs1 xs2] creates two Discriminator networks taking as
    input [xs1] and [xs2], the two networks share the same weights.
    This returns the two networks as well as their (shared) variables.
*)
let create_discriminator xs1 xs2 =
  let conv2d1 = Layer.Conv2D.create ~ksize:(5, 5) ~strides:(2, 2) ~padding:Same 16 in
  let conv2d2 = Layer.Conv2D.create ~ksize:(5, 5) ~strides:(2, 2) ~padding:Same 32 in
  let linear1 = Layer.Linear.create 1 in
  let linear2 = Layer.Linear.create 1 in
  let model xs =
    Layer.reshape xs ~shape:[ -1; 28; 28; 1 ]
    |> Layer.Conv2D.apply conv2d1
    |> O.leaky_relu ~alpha:0.1
    |> Layer.Conv2D.apply conv2d2
    |> batch_norm
    |> O.leaky_relu ~alpha:0.1
    |> Layer.reshape ~shape:[ -1; 7 * 7 * 32 ]
    |> Layer.Linear.apply linear1
    |> batch_norm
    |> O.leaky_relu ~alpha:0.1
    |> Layer.Linear.apply linear2 ~activation:Sigmoid
  in
  let ys1 = model xs1 in
  let ys2 = model xs2 in
  let vars =
    List.concat_map [ linear1; linear2 ] ~f:Layer.Linear.vars
    @ List.concat_map [ conv2d1; conv2d2 ] ~f:Layer.Conv2D.vars
  in
  ys1, ys2, vars

let write_samples samples ~filename =
  Stdio.Out_channel.with_file filename ~f:(fun channel ->
      for sample_index = 0 to 99 do
        List.init image_dim ~f:(fun pixel_index ->
            Tensorflow_core.Tensor.get samples [| sample_index; pixel_index |]
            |> Printf.sprintf "%.2f")
        |> String.concat ~sep:", "
        |> Printf.sprintf "data%d = [%s]\n" sample_index
        |> Stdio.Out_channel.output_string channel
      done)

let () =
  let mnist = Mnist_helper.read_files () in
  (* Create a placeholder for random latent data used by the generator and for the actual
     MNIST data used by the discriminator. *)
  let rand_data_ph = O.placeholder [ batch_size; latent_dim ] ~type_:Float in
  let real_data_ph = O.placeholder [ batch_size; image_dim ] ~type_:Float in
  (* Create the Generator and Discriminator networks. *)
  let generated, gen_variables = create_generator (O.Placeholder.to_node rand_data_ph) in
  let real_doutput, fake_doutput, discriminator_variables =
    create_discriminator O.((Placeholder.to_node real_data_ph * f 2.) - f 1.) generated
  in
  (* The Generator loss is based on the Discriminator making mistakes on the generated
     output. The Discriminator loss is based on being right on both real and generated
     data. *)
  let real_loss =
    O.binary_cross_entropy ~labels:(O.f 0.9) ~model_values:real_doutput `mean
  in
  let fake_loss =
    O.binary_cross_entropy ~labels:(O.f 0.) ~model_values:fake_doutput `mean
  in
  let discriminator_loss = O.(real_loss + fake_loss) in
  let generator_loss =
    O.binary_cross_entropy ~labels:(O.f 1.) ~model_values:fake_doutput `mean
  in
  let learning_rate = O.f learning_rate in
  let discriminator_opt =
    Optimizers.adam_minimizer
      ~learning_rate
      discriminator_loss
      ~varsf:discriminator_variables
  in
  let generator_opt =
    Optimizers.adam_minimizer ~learning_rate generator_loss ~varsf:gen_variables
  in
  (* Create tensor for random data both for training and validation. *)
  let batch_rand = Tensor.create2 Float32 batch_size latent_dim in
  let samples_rand = Tensor.create2 Float32 batch_size latent_dim in
  (* Always reuse the same random latent space for validation samples. *)
  Tensor.fill_uniform samples_rand ~lower_bound:(-1.) ~upper_bound:1.;
  Checkpointing.loop
    ~start_index:1
    ~end_index:batches
    ~save_vars_from:discriminator_opt
    ~checkpoint_base:"./tf-mnist-dcgan.ckpt"
    (fun ~index:batch_idx ->
      let batch_images, _ = Mnist_helper.train_batch mnist ~batch_size ~batch_idx in
      let discriminator_loss =
        Tensor.fill_uniform batch_rand ~lower_bound:(-1.) ~upper_bound:1.;
        Session.run
          ~inputs:
            Session.Input.
              [ float real_data_ph batch_images; float rand_data_ph batch_rand ]
          ~targets:discriminator_opt
          (Session.Output.scalar_float discriminator_loss)
      in
      let generator_loss =
        Tensor.fill_uniform batch_rand ~lower_bound:(-1.) ~upper_bound:1.;
        Session.run
          ~inputs:Session.Input.[ float rand_data_ph batch_rand ]
          ~targets:generator_opt
          (Session.Output.scalar_float generator_loss)
      in
      if batch_idx % 100 = 0
      then
        Stdio.printf
          "batch %4d    d-loss: %12.6f    g-loss: %12.6f\n%!"
          batch_idx
          discriminator_loss
          generator_loss;
      if batch_idx % 100000 = 0 || (batch_idx < 100000 && batch_idx % 25000 = 0)
      then
        Session.run
          (Session.Output.float generated)
          ~inputs:Session.Input.[ float rand_data_ph samples_rand ]
        |> write_samples ~filename:(Printf.sprintf "out%d.txt" batch_idx))
