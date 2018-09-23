(* Generative Adverserial Networks trained on the MNIST dataset. *)
open Base
open Tensorflow
module O = Ops

let image_dim = Mnist_helper.image_dim
let latent_dim = 100

let generator_hidden_nodes = 128
let discriminator_hidden_nodes = 128

let batch_size = 64
let learning_rate = 1e-3
let epochs = 1000

let create_generator () =
  let hidden_layer =
    O.(randomUniform ~type_:Float (ci32 [batch_size; latent_dim]) * f 2. - f 1.)
    |> Layer.linear_with_vars ~activation:(Leaky_relu 0.01) ~output_dim:generator_hidden_nodes
  in
  let final_layer =
    Layer.linear_with_vars (Layer.linear_output hidden_layer)
      ~activation:Tanh ~output_dim:image_dim
  in
  Layer.linear_output final_layer, (Layer.linear_vars hidden_layer @ Layer.linear_vars final_layer)

let create_discriminator xs1 xs2 =
  let hidden_layer =
    Layer.linear_with_vars xs1
      ~activation:(Leaky_relu 0.01) ~output_dim:discriminator_hidden_nodes
  in
  let final_layer =
    Layer.linear_with_vars (Layer.linear_output hidden_layer)
      ~activation:Sigmoid ~output_dim:image_dim
  in
  Layer.linear_output final_layer,
  Layer.linear_apply final_layer (Layer.linear_apply hidden_layer xs2),
  (Layer.linear_vars hidden_layer @ Layer.linear_vars final_layer)

let () =
  let mnist = Mnist_helper.read_files () in
  let real_data_ph = O.placeholder [batch_size; image_dim] ~type_:Float in
  let real_data = O.(Placeholder.to_node real_data_ph * f 2. - f 1.) in
  let generated, generator_variables = create_generator () in
  let real_doutput, fake_doutput, discriminator_variables =
    create_discriminator real_data generated
  in
  let real_loss =
    O.cross_entropy `mean ~ys:(O.f 1.) ~y_hats:real_doutput
  in
  let fake_loss =
    O.cross_entropy `mean ~ys:(O.f 0.) ~y_hats:fake_doutput
  in
  let discriminator_loss = O.(real_loss + fake_loss) in
  let learning_rate = O.f learning_rate in
  let discriminator_opt =
    Optimizers.adam_minimizer ~learning_rate discriminator_loss
      ~varsf:discriminator_variables
      ~varsd:[] (* TODO: remove this. *)
  in
  let generator_opt =
    Optimizers.adam_minimizer ~learning_rate fake_loss
      ~varsf:generator_variables
      ~varsd:[] (* TODO: remove this. *)
  in
  for batch_idx = 1 to epochs do
    let batch_images, _ = Mnist_helper.train_batch mnist ~batch_size ~batch_idx in
    Session.run
      ~inputs:Session.Input.[ float real_data_ph batch_images ]
      ~targets:discriminator_opt
      Session.Output.empty;
    Session.run
      ~inputs:[]
      ~targets:generator_opt
      Session.Output.empty;
    Stdio.printf "epoch %d\n%!" batch_idx
  done
