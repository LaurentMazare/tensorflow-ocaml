open Core_kernel.Std

let init_table = Node.Weak_table.create ()

let create shape ~type_ ~init =
  let node =
    Ops_generated.variable ()
      ~type_
      ~shape:(List.map shape ~f:(fun size -> { Node.Dim.size; name = None }))
  in
  Node.Weak_table.set init_table ~key:node ~data:init;
  node

let load ~type_ shape ~filename ~tensor =
  let init =
    Ops.restore ~type_ (Ops.const_string [ filename ]) (Ops.const_string [ tensor ])
  in
  create shape ~type_ ~init
let load_f = load ~type_:Float
let load_d = load ~type_:Double

let float shape ~init = create shape ~type_:Float ~init
let double shape ~init = create shape ~type_:Double ~init

let f_or_d shape x ~type_ = create shape ~type_ ~init:(Ops.f_or_d x ~shape ~type_)
let f shape x = f_or_d shape x ~type_:Float
let d shape x = f_or_d shape x ~type_:Double

let normal_gen node shape ~type_ ~stddev =
  let init =
    node (Ops.const_int ~type_:Int32 shape) ~type_
    |> Ops.mul (Ops.f_or_d stddev ~type_)
  in
  create shape ~init ~type_

let normal shape ~stddev ~type_ =
  normal_gen (fun shape -> Ops.randomStandardNormal shape) shape ~type_ ~stddev

let normalf = normal ~type_:Float
let normald = normal ~type_:Double

let truncated_normal shape ~stddev ~type_ =
  normal_gen (fun shape -> Ops.truncatedNormal shape) shape ~type_ ~stddev

let truncated_normalf = truncated_normal ~type_:Float
let truncated_normald = truncated_normal ~type_:Double

let get_init p = Node.Weak_table.find init_table p

let get_all_vars node =
  let processed_nodes = Node.Id.Hash_set.create () in
  (* Using references here make the following code quite consise. *)
  let all_vars = ref [] in
  let rec vars (Node.P node) =
    if not (Hash_set.mem processed_nodes (Node.id node))
    then begin
      Hash_set.add processed_nodes (Node.id node);
      if Node.Op_name.(=) (Node.op_name node) Ops.Op_names.variable
      then all_vars := (Node.P node) :: !all_vars
      else List.iter (Node.inputs node) ~f:vars
    end
  in
  vars (Node.P node);
  !all_vars
