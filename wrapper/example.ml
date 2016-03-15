open Wrapper
module CArray = Ctypes.CArray

let read_file filename =
  let lines = ref [] in
  let chan = open_in filename in
  try
    while true; do
      lines := input_line chan :: !lines
    done;
    assert false
  with End_of_file ->
    close_in chan;
    String.concat "\n" (List.rev !lines)

let char_list_of_string s =
  let list = ref [] in
  for i = 0 to String.length s - 1 do
    list := s.[i] :: !list
  done;
  List.rev !list

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

let binary_op op ~lhs ~rhs ~name =
  let open Graph_piqi in
  let default_attr_value = default_attr_value () in
  { Node_def.name = Some name
  ; op = Some op
  ; input = [ lhs; rhs ]
  ; device = None
  ; attr =
    [ { Node_def_attr_entry.key = Some "T"
      ; value =
        Some { default_attr_value with Attr_value.type_ = Some `dt_float }
      }
    ]
  }

let generate_protobuf () =
  let open Graph_piqi in
  let graph_def =
    let nodes =
      [ const_1d [ 3. ] ~name:"Const1"
      ; const_1d [ 2. ] ~name:"Const2"
      ; binary_op "Add" ~name:"Add1" ~lhs:"Const1" ~rhs:"Const2"
      ]
    in
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
  let obuf = gen_graph_def graph_def in
  Piqirun.to_string obuf

let () =
  let graph_def_str = generate_protobuf () in
  let vector = Tensor.create1d Ctypes.float 10 in
  let session_options = Session_options.create () in
  let status = Status.create () in
  let session = Session.create session_options status in
  Printf.printf "%d %s\n%!" (Status.code status) (Status.message status);
  let simple_pbtxt = read_file "test.pbtxt" in
  let carray = char_list_of_string simple_pbtxt |> CArray.of_list Ctypes.char in
  Session.extend_graph
    session
    carray
    (String.length simple_pbtxt)
    status;
  Printf.printf "%d %s\n%!" (Status.code status) (Status.message status);
  let output_tensors =
    Session.run
      session
      ~inputs:[]
      ~outputs:[ "add" ]
      ~targets:[ "add" ]
      status
  in
  Printf.printf "%d %s\n%!" (Status.code status) (Status.message status);
  match output_tensors with
  | [ output_tensor ] ->
    let data = Tensor.data output_tensor Ctypes.float 1 in
    Printf.printf "%f\n%!" (CArray.get data 0)
  | [] | _ :: _ :: _ -> assert false

