open Base
open Float.O_dot

let create shape ~type_ ~init =
  let node =
    Ops_generated.variable
      ()
      ~type_
      ~shape:(List.map shape ~f:(fun size -> { Node.Dim.size; name = None }))
  in
  let assign_node = Ops_generated.assign node init in
  Node.Session.add_variable_initialization (Node.operation assign_node);
  node

let load ~type_ shape ~filename ~tensor =
  let init =
    Ops.restore ~type_ (Ops.const_string0 filename) (Ops.const_string0 tensor)
  in
  create shape ~type_ ~init

let load_f = load ~type_:Float
let load_d = load ~type_:Double
let float shape ~init = create shape ~type_:Float ~init
let double shape ~init = create shape ~type_:Double ~init
let f_or_d shape x ~type_ = create shape ~type_ ~init:(Ops.f_or_d x ~shape ~type_)
let f shape x = f_or_d shape x ~type_:Float
let d shape x = f_or_d shape x ~type_:Double

let gen node shape ~type_ ~scale =
  let init =
    node (Ops.const_int ~type_:Int32 shape) ~type_ |> Ops.mul (Ops.f_or_d scale ~type_)
  in
  create shape ~init ~type_

let normal shape ~stddev ~type_ =
  gen (fun shape -> Ops.randomStandardNormal shape) shape ~type_ ~scale:stddev

let normalf = normal ~type_:Float
let normald = normal ~type_:Double

let truncated_normal shape ~stddev ~type_ =
  gen (fun shape -> Ops.truncatedNormal shape) shape ~type_ ~scale:stddev

let truncated_normalf = truncated_normal ~type_:Float
let truncated_normald = truncated_normal ~type_:Double

let uniform shape ~lo ~hi ~type_ =
  gen (fun shape -> Ops.randomUniform shape) shape ~type_ ~scale:(hi -. lo)
  |> Ops.add (Ops.f_or_d lo ~type_)

let uniformf = uniform ~type_:Float
let uniformd = uniform ~type_:Double

let get_all_vars nodes =
  let processed_nodes = Hash_set.create (module Node.Id) in
  (* Using references here make the following code quite consise. *)
  let all_vars = ref [] in
  let rec vars (Node.P node) =
    if not (Hash_set.mem processed_nodes (Node.id node))
    then (
      Hash_set.add processed_nodes (Node.id node);
      if Node.Op_name.( = ) (Node.op_name node) Ops.Op_names.variable
      then all_vars := Node.P node :: !all_vars
      else List.iter (Node.flat_inputs node) ~f:vars)
  in
  List.iter nodes ~f:vars;
  !all_vars
