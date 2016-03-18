open Node

let nary op ~input ~name =
  let open Graph_piqi in
  let default_attr_value = default_attr_value () in
  { Node_def.name = Some name
  ; op = Some op
  ; input
  ; device = None
  ; attr =
    [ { key = Some "T"
      ; value =
        Some { default_attr_value with Attr_value.type_ = Some `dt_float }
      }
    ]
  }

let to_protobuf t =
  let open Graph_piqi in
  let default_attr_value = default_attr_value () in
  let walk (type a) (t : a t) =
    let attr =
      List.map
        (fun (name, value) ->
          let value =
            match value with
            | Type type_ ->
              let type_ = Type.to_dt_type type_ in
              Some { default_attr_value with Attr_value.type_ = Some type_ }
            (* TODO *)
            | _ -> None
          in
          { Node_def_attr_entry.key = Some name
          ; value
          }
        )
        t.attributes
    in
    let current_node =
      { Node_def.name = Some (Node.Name.to_string t.name)
      ; op = Some t.op_name
      ; input = List.map (fun (P input) -> Node.Name.to_string input.name) t.inputs
      ; device = None
      ; attr
      }
    in
    current_node :: [] (* List.concat (List.map walk t.inputs) *)
  in
  let graph_def =
    { Graph_def.node = walk t
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
  |> Protobuf.of_string
