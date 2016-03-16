type t =
  { t_: t_
  ; name : string
  }
and t_ =
  | Const of float list
  | Add of t * t
  | Sub of t * t
  | Mul of t * t
  | Div of t * t
  | Exp of t

let name t = t.name

let fresh_name =
  let cnt = ref 0 in
  fun () ->
    incr cnt;
    Printf.sprintf "node%d" !cnt

let const floats =
  { name = fresh_name ()
  ; t_ = Const floats
  }

let add lhs rhs =
  { name = fresh_name ()
  ; t_ = Add (lhs, rhs)
  }

let sub lhs rhs =
  { name = fresh_name ()
  ; t_ = Add (lhs, rhs)
  }

let mul lhs rhs =
  { name = fresh_name ()
  ; t_ = Mul (lhs, rhs)
  }

let div lhs rhs =
  { name = fresh_name ()
  ; t_ = Div (lhs, rhs)
  }

let exp arg =
  { name = fresh_name ()
  ; t_ = Exp arg
  }


module Protobuf = struct
  let const_1d floats ~name =
    let open Graph_piqi in
    let default_attr_value = default_attr_value () in
    let default_tensor_proto = default_tensor_proto () in
    { Node_def.name = Some name
    ; op = Some "Const"
    ; input = []
    ; device = None
    ; attr =
      [ { Node_def_attr_entry.key = Some "dtype"
        ; value =
          Some { default_attr_value with Attr_value.type_ = Some `dt_float }
        }
      ; { Node_def_attr_entry.key = Some "value"
        ; value =
          Some
            { default_attr_value with
              Attr_value.tensor =
                Some
                  { default_tensor_proto with
                    Tensor_proto.dtype = Some `dt_float
                  ; float_val = floats
                  ; tensor_shape =
                    Some
                      { Tensor_shape_proto.dim =
                        [ { Tensor_shape_proto_dim.size = Some (List.length floats |> Int64.of_int); name = None } ]
                      ; unknown_rank = None
                      }
                  }
            }
        }
      ]
    }
  
  let nary op ~input ~name =
    let open Graph_piqi in
    let default_attr_value = default_attr_value () in
    { Node_def.name = Some name
    ; op = Some op
    ; input
    ; device = None
    ; attr =
      [ { Node_def_attr_entry.key = Some "T"
        ; value =
          Some { default_attr_value with Attr_value.type_ = Some `dt_float }
        }
      ]
    }
  
  let binary_op op ~lhs ~rhs ~name =
    nary op ~input:[ lhs; rhs ] ~name
  
  let unary_op op ~arg ~name =
    nary op ~input:[ arg ] ~name

  let to_protobuf t =
    let rec walk { name; t_ } =
      match t_ with
      | Const floats ->
        [ const_1d floats ~name ]
      | Add (lhs, rhs) ->
        binary_op "Add" ~lhs:lhs.name ~rhs:rhs.name ~name ::
        (walk lhs @ walk rhs)
      | Sub (lhs, rhs) ->
        binary_op "Sub" ~lhs:lhs.name ~rhs:rhs.name ~name ::
        (walk lhs @ walk rhs)
      | Mul (lhs, rhs) ->
        binary_op "Mul" ~lhs:lhs.name ~rhs:rhs.name ~name ::
        (walk lhs @ walk rhs)
      | Div (lhs, rhs) ->
        binary_op "Div" ~lhs:lhs.name ~rhs:rhs.name ~name ::
        (walk lhs @ walk rhs)
      | Exp arg ->
        unary_op "Exp" ~arg:arg.name ~name :: walk arg
    in
    let open Graph_piqi in
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
end
