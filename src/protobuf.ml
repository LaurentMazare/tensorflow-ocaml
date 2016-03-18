type t = string

let of_string x = x
let to_string x = x

open Node
open Graph_piqi

let default_attr_value = default_attr_value ()
let default_tensor_proto = default_tensor_proto ()

let of_attribute name value =
  let value =
    match value with
    | Type type_ ->
      let type_ = Type.to_dt_type type_ in
      Some { default_attr_value with Attr_value.type_ = Some type_ }
    | Tensor_float tensor_float ->
      let tensor =
        { default_tensor_proto with
          dtype = Some `dt_float (* TODO: use the output type. *)
        ; float_val = tensor_float.values
        ; tensor_shape =
          Some
            { dim =
              List.map
              (fun d -> { Tensor_shape_proto_dim.size = Some (Int64.of_int d); name = None })
              tensor_float.shape
            ; unknown_rank = None
            }
        }
      in
      Some { default_attr_value with tensor = Some tensor }
   (* TODO *)
    | _ -> None
  in
  { Node_def_attr_entry.key = Some name
  ; value
  }

let of_node t =
  let nodes = Hashtbl.create 128 in
  let rec walk p =
    let P t = p in
    if Hashtbl.mem nodes t.name
    then ()
    else begin
      let attr = List.map (fun (name, value) -> of_attribute name value) t.attributes in
      let node =
        { Node_def.name = Some (Node.Name.to_string t.name)
        ; op = Some t.op_name
        ; input = List.map (fun (P input) -> Node.Name.to_string input.name) t.inputs
        ; device = None
        ; attr
        }
      in
      Hashtbl.add nodes t.name node;
      List.iter walk t.inputs
    end
  in
  walk t;
  let nodes = Hashtbl.fold (fun _ v acc -> v :: acc) nodes [] in
  let graph_def =
    { Graph_def.node = nodes
    ; versions =
      Some
        { Version_def.producer = Some (Int32.of_int 8)
        ; min_consumer = None
        ; bad_consumers = []
        }
    ; version = None
    ; library = None
    }
  in
  gen_graph_def graph_def
  |> Piqirun.to_string
