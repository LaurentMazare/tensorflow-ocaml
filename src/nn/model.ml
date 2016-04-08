open Core_kernel.Std

type t =
  { session : Session.t
  ; net : Nn._1d Nn.t
  ; placeholder : [ `float ] Node.t
  }

module Optimizer = struct
  (* We should use some inline records here when they will be available. *)
  type t =
    | Gradient_descent of float
    | Adam of float * float option * float option * float option
    | Momentum of float * float

  let gradient_descent ~alpha = Gradient_descent alpha

  let adam ~alpha ?beta1 ?beta2 ?epsilon () =
    Adam (alpha, beta1, beta2, epsilon)

  let momentum ~alpha ~momentum =
    Momentum (alpha, momentum)

  let get t ~loss =
    match t with
    | Gradient_descent alpha ->
      Optimizers.gradient_descent_minimizer ~alpha:(Ops.f alpha) loss
    | Adam (alpha, beta1, beta2, epsilon) ->
      Optimizers.adam_minimizer loss
        ~alpha:(Ops.f alpha)
        ?beta1:(Option.map beta1 ~f:Ops.f)
        ?beta2:(Option.map beta2 ~f:Ops.f)
        ?epsilon:(Option.map epsilon ~f:Ops.f)
    | Momentum (alpha, momentum) ->
      Optimizers.momentum_minimizer loss
        ~alpha:(Ops.f alpha)
        ~momentum:(Ops.f momentum)
end

module Loss = struct
  type t =
    | Cross_entropy of [ `sum | `mean ]
    | L2 of [ `sum | `mean ]

  let cross_entropy sum_mean = Cross_entropy sum_mean
  let l2 sum_mean = L2 sum_mean

  let get t ~sample_ys ~model_ys =
    let reduce = function
      | `sum -> Ops.reduce_sum
      | `mean -> Ops.reduce_mean
    in
    match t with
    | Cross_entropy sum_mean ->
      Ops.(neg (reduce sum_mean (sample_ys * log model_ys)))
    | L2 sum_mean ->
      Ops.(reduce sum_mean (square (sample_ys - model_ys)))
end

let create net =
  let session = Session.create () in
  let placeholder = Ops.placeholder ~type_:Float (Nn.shape net |> Nn.Shape.dim_list) in
  { session
  ; net
  ; placeholder
  }

let all_inputs ?(named_inputs=[]) t xs =
  let inputs =
    List.map named_inputs ~f:(fun (input_name, value) ->
      let node = Nn.Input_name.to_node input_name in
      Session.Input.float node value)
  in
  match Nn.default_input t.net with
  | None -> inputs
  | Some input_name ->
    let node = Nn.Input_name.to_node input_name in
    Session.Input.float node xs :: inputs

let fit ?named_inputs ?batch_size ?on_epoch ~loss ~optimizer ~epochs ~xs ~ys t =
  let loss = Loss.get loss ~sample_ys:t.placeholder ~model_ys:(Nn.node t.net) in
  let optimizer = Optimizer.get optimizer ~loss in
  let samples = (Bigarray.Genarray.dims xs).(0) in
  let batch_size =
    match batch_size with
    | None -> None
    | Some batch_size when batch_size > samples -> None
    | Some _ as s -> s
  in
  let inputs ~xs ~ys =
    (Session.Input.float t.placeholder ys)
    :: all_inputs ?named_inputs t xs
  in
  let on_epoch =
    Option.value on_epoch
      ~default:(fun (_:int) ~err:_ ~loss:_ -> `print_err)
  in
  for epoch = 1 to epochs do
    let inputs =
      match batch_size with
      | None -> inputs ~xs ~ys
      | Some batch_size ->
        let offset = ((epoch-1) * batch_size) mod (samples - batch_size) in
        let xs = Bigarray.Genarray.sub_left xs offset batch_size in
        let ys = Bigarray.Genarray.sub_left ys offset batch_size in
        inputs ~xs ~ys
    in
    let err =
      Session.run
        ~inputs
        ~targets:optimizer
        ~session:t.session
        (Session.Output.scalar_float loss)
    in
    match on_epoch epoch ~err ~loss with
    | `print_err -> printf "Epoch: %6d/%-6d   Loss: %.2f\n%!" epoch epochs err
    | `do_nothing -> ()
  done

let evaluate ?named_inputs ?batch_size ?node t xs =
  let nsamples = Bigarray.Genarray.nth_dim xs 0 in
  let nbatchs, batch_size =
    match batch_size with
    | None -> 1, nsamples
    | Some batch_size -> 1 + (nsamples - 1) / batch_size, batch_size
  in
  let node =
    match node with
    | None -> Nn.node t.net
    | Some node -> node
  in
  if nbatchs = 1
  then
    Session.run
      ~inputs:(all_inputs ?named_inputs t xs)
      ~session:t.session
      (Session.Output.float node)
  else begin
    let dims = Nn.shape t.net |> Nn.Shape.dim_list in
    let ys = Tensor.create Float32 (Array.of_list (nsamples :: dims)) in
    for batch_idx = 0 to nbatchs - 1 do
      let samples_idx = batch_idx * batch_size in
      let samples_count = min batch_size (nsamples - samples_idx) in
      let xs = Bigarray.Genarray.sub_left xs samples_idx samples_count in
      let batch_results =
        Session.run
          ~inputs:(all_inputs ?named_inputs t xs)
          ~session:t.session
          (Session.Output.float node)
      in
      Bigarray.Genarray.blit
        batch_results
        (Bigarray.Genarray.sub_left ys samples_idx samples_count)
    done;
    ys
  end
