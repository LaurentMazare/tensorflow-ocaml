module rec Graph_piqi:
  sig
    type float32 = float
    type float64 = float
    type protobuf_int64 = int64
    type protobuf_int32 = int32
    type binary = string
    type data_type =
      [
        | `dt_invalid
        | `dt_float
        | `dt_double
        | `dt_int32
        | `dt_uint8
        | `dt_int16
        | `dt_int8
        | `dt_string
        | `dt_complex64
        | `dt_int64
        | `dt_bool
        | `dt_qint8
        | `dt_quint8
        | `dt_qint32
        | `dt_bfloat16
        | `dt_qint16
        | `dt_quint16
        | `dt_uint16
        | `dt_complex128
        | `dt_half
        | `dt_float_ref
        | `dt_double_ref
        | `dt_int32_ref
        | `dt_uint8_ref
        | `dt_int16_ref
        | `dt_int8_ref
        | `dt_string_ref
        | `dt_complex64_ref
        | `dt_int64_ref
        | `dt_bool_ref
        | `dt_qint8_ref
        | `dt_quint8_ref
        | `dt_qint32_ref
        | `dt_bfloat16_ref
        | `dt_qint16_ref
        | `dt_quint16_ref
        | `dt_uint16_ref
        | `dt_complex128_ref
        | `dt_half_ref
      ]
    type tensor_shape_proto = Tensor_shape_proto.t
    type tensor_shape_proto_dim = Tensor_shape_proto_dim.t
    type tensor_proto = Tensor_proto.t
    type attr_value = Attr_value.t
    type attr_value_list_value = Attr_value_list_value.t
    type name_attr_list = Name_attr_list.t
    type name_attr_list_attr_entry = Name_attr_list_attr_entry.t
    type op_def = Op_def.t
    type op_def_arg_def = Op_def_arg_def.t
    type op_def_attr_def = Op_def_attr_def.t
    type op_deprecation = Op_deprecation.t
    type op_list = Op_list.t
    type function_def_library = Function_def_library.t
    type function_def = Function_def.t
    type function_def_node = Function_def_node.t
    type function_def_node_attr_entry = Function_def_node_attr_entry.t
    type gradient_def = Gradient_def.t
    type version_def = Version_def.t
    type graph_def = Graph_def.t
    type node_def = Node_def.t
    type node_def_attr_entry = Node_def_attr_entry.t
  end = Graph_piqi
and Tensor_shape_proto:
  sig
    type t = {
      mutable dim: Graph_piqi.tensor_shape_proto_dim list;
      mutable unknown_rank: bool option;
    }
  end = Tensor_shape_proto
and Tensor_shape_proto_dim:
  sig
    type t = {
      mutable size: Graph_piqi.protobuf_int64 option;
      mutable name: string option;
    }
  end = Tensor_shape_proto_dim
and Tensor_proto:
  sig
    type t = {
      mutable dtype: Graph_piqi.data_type option;
      mutable tensor_shape: Graph_piqi.tensor_shape_proto option;
      mutable version_number: Graph_piqi.protobuf_int32 option;
      mutable tensor_content: Graph_piqi.binary option;
      mutable half_val: Graph_piqi.protobuf_int32 list;
      mutable float_val: Graph_piqi.float32 list;
      mutable double_val: Graph_piqi.float64 list;
      mutable int_val: Graph_piqi.protobuf_int32 list;
      mutable string_val: Graph_piqi.binary list;
      mutable scomplex_val: Graph_piqi.float32 list;
      mutable int64_val: Graph_piqi.protobuf_int64 list;
      mutable bool_val: bool list;
      mutable dcomplex_val: Graph_piqi.float64 list;
    }
  end = Tensor_proto
and Attr_value:
  sig
    type t = {
      mutable s: Graph_piqi.binary option;
      mutable i: Graph_piqi.protobuf_int64 option;
      mutable f: Graph_piqi.float32 option;
      mutable b: bool option;
      mutable type_: Graph_piqi.data_type option;
      mutable shape: Graph_piqi.tensor_shape_proto option;
      mutable tensor: Graph_piqi.tensor_proto option;
      mutable list: Graph_piqi.attr_value_list_value option;
      mutable func: Graph_piqi.name_attr_list option;
      mutable placeholder: string option;
    }
  end = Attr_value
and Attr_value_list_value:
  sig
    type t = {
      mutable s: Graph_piqi.binary list;
      mutable i: Graph_piqi.protobuf_int64 list;
      mutable f: Graph_piqi.float32 list;
      mutable b: bool list;
      mutable type_: Graph_piqi.data_type list;
      mutable shape: Graph_piqi.tensor_shape_proto list;
      mutable tensor: Graph_piqi.tensor_proto list;
    }
  end = Attr_value_list_value
and Name_attr_list:
  sig
    type t = {
      mutable name: string option;
      mutable attr: Graph_piqi.name_attr_list_attr_entry list;
    }
  end = Name_attr_list
and Name_attr_list_attr_entry:
  sig
    type t = {
      mutable key: string option;
      mutable value: Graph_piqi.attr_value option;
    }
  end = Name_attr_list_attr_entry
and Op_def:
  sig
    type t = {
      mutable name: string option;
      mutable input_arg: Graph_piqi.op_def_arg_def list;
      mutable output_arg: Graph_piqi.op_def_arg_def list;
      mutable attr: Graph_piqi.op_def_attr_def list;
      mutable deprecation: Graph_piqi.op_deprecation option;
      mutable summary: string option;
      mutable description: string option;
      mutable is_commutative: bool option;
      mutable is_aggregate: bool option;
      mutable is_stateful: bool option;
      mutable allows_uninitialized_input: bool option;
    }
  end = Op_def
and Op_def_arg_def:
  sig
    type t = {
      mutable name: string option;
      mutable description: string option;
      mutable type_: Graph_piqi.data_type option;
      mutable type_attr: string option;
      mutable number_attr: string option;
      mutable type_list_attr: string option;
      mutable is_ref: bool option;
    }
  end = Op_def_arg_def
and Op_def_attr_def:
  sig
    type t = {
      mutable name: string option;
      mutable type_: string option;
      mutable default_value: Graph_piqi.attr_value option;
      mutable description: string option;
      mutable has_minimum: bool option;
      mutable minimum: Graph_piqi.protobuf_int64 option;
      mutable allowed_values: Graph_piqi.attr_value option;
    }
  end = Op_def_attr_def
and Op_deprecation:
  sig
    type t = {
      mutable version: Graph_piqi.protobuf_int32 option;
      mutable explanation: string option;
    }
  end = Op_deprecation
and Op_list:
  sig
    type t = {
      mutable op: Graph_piqi.op_def list;
    }
  end = Op_list
and Function_def_library:
  sig
    type t = {
      mutable function_: Graph_piqi.function_def list;
      mutable gradient: Graph_piqi.gradient_def list;
    }
  end = Function_def_library
and Function_def:
  sig
    type t = {
      mutable signature: Graph_piqi.op_def option;
      mutable node: Graph_piqi.function_def_node list;
    }
  end = Function_def
and Function_def_node:
  sig
    type t = {
      mutable ret: string list;
      mutable op: string option;
      mutable arg: string list;
      mutable dep: string list;
      mutable attr: Graph_piqi.function_def_node_attr_entry list;
    }
  end = Function_def_node
and Function_def_node_attr_entry:
  sig
    type t = {
      mutable key: string option;
      mutable value: Graph_piqi.attr_value option;
    }
  end = Function_def_node_attr_entry
and Gradient_def:
  sig
    type t = {
      mutable function_name: string option;
      mutable gradient_func: string option;
    }
  end = Gradient_def
and Version_def:
  sig
    type t = {
      mutable producer: Graph_piqi.protobuf_int32 option;
      mutable min_consumer: Graph_piqi.protobuf_int32 option;
      mutable bad_consumers: Graph_piqi.protobuf_int32 list;
    }
  end = Version_def
and Graph_def:
  sig
    type t = {
      mutable node: Graph_piqi.node_def list;
      mutable versions: Graph_piqi.version_def option;
      mutable version: Graph_piqi.protobuf_int32 option;
      mutable library: Graph_piqi.function_def_library option;
    }
  end = Graph_def
and Node_def:
  sig
    type t = {
      mutable name: string option;
      mutable op: string option;
      mutable input: string list;
      mutable device: string option;
      mutable attr: Graph_piqi.node_def_attr_entry list;
    }
  end = Node_def
and Node_def_attr_entry:
  sig
    type t = {
      mutable key: string option;
      mutable value: Graph_piqi.attr_value option;
    }
  end = Node_def_attr_entry


let rec parse_int64 x = Piqirun.int64_of_zigzag_varint x
and packed_parse_int64 x = Piqirun.int64_of_packed_zigzag_varint x

and parse_int32 x = Piqirun.int32_of_zigzag_varint x
and packed_parse_int32 x = Piqirun.int32_of_packed_zigzag_varint x

and parse_bool x = Piqirun.bool_of_varint x
and packed_parse_bool x = Piqirun.bool_of_packed_varint x

and parse_protobuf_int64 x = Piqirun.int64_of_signed_varint x
and packed_parse_protobuf_int64 x = Piqirun.int64_of_packed_signed_varint x

and parse_string x = Piqirun.string_of_block x

and parse_protobuf_int32 x = Piqirun.int32_of_signed_varint x
and packed_parse_protobuf_int32 x = Piqirun.int32_of_packed_signed_varint x

and parse_binary x = Piqirun.string_of_block x

and parse_float32 x = Piqirun.float_of_fixed32 x
and packed_parse_float32 x = Piqirun.float_of_packed_fixed32 x

and parse_float64 x = Piqirun.float_of_fixed64 x
and packed_parse_float64 x = Piqirun.float_of_packed_fixed64 x

and parse_tensor_shape_proto x =
  let x = Piqirun.parse_record x in
  let _dim, x = Piqirun.parse_repeated_field 2 parse_tensor_shape_proto_dim x in
  let _unknown_rank, x = Piqirun.parse_optional_field 3 parse_bool x in
  Piqirun.check_unparsed_fields x;
  {
    Tensor_shape_proto.dim = _dim;
    Tensor_shape_proto.unknown_rank = _unknown_rank;
  }

and parse_tensor_shape_proto_dim x =
  let x = Piqirun.parse_record x in
  let _size, x = Piqirun.parse_optional_field 1 parse_protobuf_int64 x in
  let _name, x = Piqirun.parse_optional_field 2 parse_string x in
  Piqirun.check_unparsed_fields x;
  {
    Tensor_shape_proto_dim.size = _size;
    Tensor_shape_proto_dim.name = _name;
  }

and parse_tensor_proto x =
  let x = Piqirun.parse_record x in
  let _dtype, x = Piqirun.parse_optional_field 1 parse_data_type x in
  let _tensor_shape, x = Piqirun.parse_optional_field 2 parse_tensor_shape_proto x in
  let _version_number, x = Piqirun.parse_optional_field 3 parse_protobuf_int32 x in
  let _tensor_content, x = Piqirun.parse_optional_field 4 parse_binary x in
  let _float_val, x = Piqirun.parse_packed_repeated_field 5 packed_parse_float32 parse_float32 x in
  let _double_val, x = Piqirun.parse_packed_repeated_field 6 packed_parse_float64 parse_float64 x in
  let _int_val, x = Piqirun.parse_packed_repeated_field 7 packed_parse_protobuf_int32 parse_protobuf_int32 x in
  let _string_val, x = Piqirun.parse_repeated_field 8 parse_binary x in
  let _scomplex_val, x = Piqirun.parse_packed_repeated_field 9 packed_parse_float32 parse_float32 x in
  let _int64_val, x = Piqirun.parse_packed_repeated_field 10 packed_parse_protobuf_int64 parse_protobuf_int64 x in
  let _bool_val, x = Piqirun.parse_packed_repeated_field 11 packed_parse_bool parse_bool x in
  let _dcomplex_val, x = Piqirun.parse_packed_repeated_field 12 packed_parse_float64 parse_float64 x in
  let _half_val, x = Piqirun.parse_packed_repeated_field 13 packed_parse_protobuf_int32 parse_protobuf_int32 x in
  Piqirun.check_unparsed_fields x;
  {
    Tensor_proto.dtype = _dtype;
    Tensor_proto.tensor_shape = _tensor_shape;
    Tensor_proto.version_number = _version_number;
    Tensor_proto.tensor_content = _tensor_content;
    Tensor_proto.float_val = _float_val;
    Tensor_proto.double_val = _double_val;
    Tensor_proto.int_val = _int_val;
    Tensor_proto.string_val = _string_val;
    Tensor_proto.scomplex_val = _scomplex_val;
    Tensor_proto.int64_val = _int64_val;
    Tensor_proto.bool_val = _bool_val;
    Tensor_proto.dcomplex_val = _dcomplex_val;
    Tensor_proto.half_val = _half_val;
  }

and parse_attr_value x =
  let x = Piqirun.parse_record x in
  let _list, x = Piqirun.parse_optional_field 1 parse_attr_value_list_value x in
  let _s, x = Piqirun.parse_optional_field 2 parse_binary x in
  let _i, x = Piqirun.parse_optional_field 3 parse_protobuf_int64 x in
  let _f, x = Piqirun.parse_optional_field 4 parse_float32 x in
  let _b, x = Piqirun.parse_optional_field 5 parse_bool x in
  let _type_, x = Piqirun.parse_optional_field 6 parse_data_type x in
  let _shape, x = Piqirun.parse_optional_field 7 parse_tensor_shape_proto x in
  let _tensor, x = Piqirun.parse_optional_field 8 parse_tensor_proto x in
  let _placeholder, x = Piqirun.parse_optional_field 9 parse_string x in
  let _func, x = Piqirun.parse_optional_field 10 parse_name_attr_list x in
  Piqirun.check_unparsed_fields x;
  {
    Attr_value.list = _list;
    Attr_value.s = _s;
    Attr_value.i = _i;
    Attr_value.f = _f;
    Attr_value.b = _b;
    Attr_value.type_ = _type_;
    Attr_value.shape = _shape;
    Attr_value.tensor = _tensor;
    Attr_value.placeholder = _placeholder;
    Attr_value.func = _func;
  }

and parse_attr_value_list_value x =
  let x = Piqirun.parse_record x in
  let _s, x = Piqirun.parse_repeated_field 2 parse_binary x in
  let _i, x = Piqirun.parse_packed_repeated_field 3 packed_parse_protobuf_int64 parse_protobuf_int64 x in
  let _f, x = Piqirun.parse_packed_repeated_field 4 packed_parse_float32 parse_float32 x in
  let _b, x = Piqirun.parse_packed_repeated_field 5 packed_parse_bool parse_bool x in
  let _type_, x = Piqirun.parse_packed_repeated_field 6 packed_parse_data_type parse_data_type x in
  let _shape, x = Piqirun.parse_repeated_field 7 parse_tensor_shape_proto x in
  let _tensor, x = Piqirun.parse_repeated_field 8 parse_tensor_proto x in
  Piqirun.check_unparsed_fields x;
  {
    Attr_value_list_value.s = _s;
    Attr_value_list_value.i = _i;
    Attr_value_list_value.f = _f;
    Attr_value_list_value.b = _b;
    Attr_value_list_value.type_ = _type_;
    Attr_value_list_value.shape = _shape;
    Attr_value_list_value.tensor = _tensor;
  }

and parse_name_attr_list x =
  let x = Piqirun.parse_record x in
  let _name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _attr, x = Piqirun.parse_repeated_field 2 parse_name_attr_list_attr_entry x in
  Piqirun.check_unparsed_fields x;
  {
    Name_attr_list.name = _name;
    Name_attr_list.attr = _attr;
  }

and parse_name_attr_list_attr_entry x =
  let x = Piqirun.parse_record x in
  let _key, x = Piqirun.parse_optional_field 1 parse_string x in
  let _value, x = Piqirun.parse_optional_field 2 parse_attr_value x in
  Piqirun.check_unparsed_fields x;
  {
    Name_attr_list_attr_entry.key = _key;
    Name_attr_list_attr_entry.value = _value;
  }

and parse_op_def x =
  let x = Piqirun.parse_record x in
  let _name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _input_arg, x = Piqirun.parse_repeated_field 2 parse_op_def_arg_def x in
  let _output_arg, x = Piqirun.parse_repeated_field 3 parse_op_def_arg_def x in
  let _attr, x = Piqirun.parse_repeated_field 4 parse_op_def_attr_def x in
  let _summary, x = Piqirun.parse_optional_field 5 parse_string x in
  let _description, x = Piqirun.parse_optional_field 6 parse_string x in
  let _deprecation, x = Piqirun.parse_optional_field 8 parse_op_deprecation x in
  let _is_aggregate, x = Piqirun.parse_optional_field 16 parse_bool x in
  let _is_stateful, x = Piqirun.parse_optional_field 17 parse_bool x in
  let _is_commutative, x = Piqirun.parse_optional_field 18 parse_bool x in
  let _allows_uninitialized_input, x = Piqirun.parse_optional_field 19 parse_bool x in
  Piqirun.check_unparsed_fields x;
  {
    Op_def.name = _name;
    Op_def.input_arg = _input_arg;
    Op_def.output_arg = _output_arg;
    Op_def.attr = _attr;
    Op_def.summary = _summary;
    Op_def.description = _description;
    Op_def.deprecation = _deprecation;
    Op_def.is_aggregate = _is_aggregate;
    Op_def.is_stateful = _is_stateful;
    Op_def.is_commutative = _is_commutative;
    Op_def.allows_uninitialized_input = _allows_uninitialized_input;
  }

and parse_op_def_arg_def x =
  let x = Piqirun.parse_record x in
  let _name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _description, x = Piqirun.parse_optional_field 2 parse_string x in
  let _type_, x = Piqirun.parse_optional_field 3 parse_data_type x in
  let _type_attr, x = Piqirun.parse_optional_field 4 parse_string x in
  let _number_attr, x = Piqirun.parse_optional_field 5 parse_string x in
  let _type_list_attr, x = Piqirun.parse_optional_field 6 parse_string x in
  let _is_ref, x = Piqirun.parse_optional_field 16 parse_bool x in
  Piqirun.check_unparsed_fields x;
  {
    Op_def_arg_def.name = _name;
    Op_def_arg_def.description = _description;
    Op_def_arg_def.type_ = _type_;
    Op_def_arg_def.type_attr = _type_attr;
    Op_def_arg_def.number_attr = _number_attr;
    Op_def_arg_def.type_list_attr = _type_list_attr;
    Op_def_arg_def.is_ref = _is_ref;
  }

and parse_op_def_attr_def x =
  let x = Piqirun.parse_record x in
  let _name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _type_, x = Piqirun.parse_optional_field 2 parse_string x in
  let _default_value, x = Piqirun.parse_optional_field 3 parse_attr_value x in
  let _description, x = Piqirun.parse_optional_field 4 parse_string x in
  let _has_minimum, x = Piqirun.parse_optional_field 5 parse_bool x in
  let _minimum, x = Piqirun.parse_optional_field 6 parse_protobuf_int64 x in
  let _allowed_values, x = Piqirun.parse_optional_field 7 parse_attr_value x in
  Piqirun.check_unparsed_fields x;
  {
    Op_def_attr_def.name = _name;
    Op_def_attr_def.type_ = _type_;
    Op_def_attr_def.default_value = _default_value;
    Op_def_attr_def.description = _description;
    Op_def_attr_def.has_minimum = _has_minimum;
    Op_def_attr_def.minimum = _minimum;
    Op_def_attr_def.allowed_values = _allowed_values;
  }

and parse_op_deprecation x =
  let x = Piqirun.parse_record x in
  let _version, x = Piqirun.parse_optional_field 1 parse_protobuf_int32 x in
  let _explanation, x = Piqirun.parse_optional_field 2 parse_string x in
  Piqirun.check_unparsed_fields x;
  {
    Op_deprecation.version = _version;
    Op_deprecation.explanation = _explanation;
  }

and parse_op_list x =
  let x = Piqirun.parse_record x in
  let _op, x = Piqirun.parse_repeated_field 1 parse_op_def x in
  Piqirun.check_unparsed_fields x;
  {
    Op_list.op = _op;
  }

and parse_function_def_library x =
  let x = Piqirun.parse_record x in
  let _function_, x = Piqirun.parse_repeated_field 1 parse_function_def x in
  let _gradient, x = Piqirun.parse_repeated_field 2 parse_gradient_def x in
  Piqirun.check_unparsed_fields x;
  {
    Function_def_library.function_ = _function_;
    Function_def_library.gradient = _gradient;
  }

and parse_function_def x =
  let x = Piqirun.parse_record x in
  let _signature, x = Piqirun.parse_optional_field 1 parse_op_def x in
  let _node, x = Piqirun.parse_repeated_field 2 parse_function_def_node x in
  Piqirun.check_unparsed_fields x;
  {
    Function_def.signature = _signature;
    Function_def.node = _node;
  }

and parse_function_def_node x =
  let x = Piqirun.parse_record x in
  let _ret, x = Piqirun.parse_repeated_field 1 parse_string x in
  let _op, x = Piqirun.parse_optional_field 2 parse_string x in
  let _arg, x = Piqirun.parse_repeated_field 3 parse_string x in
  let _dep, x = Piqirun.parse_repeated_field 4 parse_string x in
  let _attr, x = Piqirun.parse_repeated_field 5 parse_function_def_node_attr_entry x in
  Piqirun.check_unparsed_fields x;
  {
    Function_def_node.ret = _ret;
    Function_def_node.op = _op;
    Function_def_node.arg = _arg;
    Function_def_node.dep = _dep;
    Function_def_node.attr = _attr;
  }

and parse_function_def_node_attr_entry x =
  let x = Piqirun.parse_record x in
  let _key, x = Piqirun.parse_optional_field 1 parse_string x in
  let _value, x = Piqirun.parse_optional_field 2 parse_attr_value x in
  Piqirun.check_unparsed_fields x;
  {
    Function_def_node_attr_entry.key = _key;
    Function_def_node_attr_entry.value = _value;
  }

and parse_gradient_def x =
  let x = Piqirun.parse_record x in
  let _function_name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _gradient_func, x = Piqirun.parse_optional_field 2 parse_string x in
  Piqirun.check_unparsed_fields x;
  {
    Gradient_def.function_name = _function_name;
    Gradient_def.gradient_func = _gradient_func;
  }

and parse_version_def x =
  let x = Piqirun.parse_record x in
  let _producer, x = Piqirun.parse_optional_field 1 parse_protobuf_int32 x in
  let _min_consumer, x = Piqirun.parse_optional_field 2 parse_protobuf_int32 x in
  let _bad_consumers, x = Piqirun.parse_repeated_field 3 parse_protobuf_int32 x in
  Piqirun.check_unparsed_fields x;
  {
    Version_def.producer = _producer;
    Version_def.min_consumer = _min_consumer;
    Version_def.bad_consumers = _bad_consumers;
  }

and parse_graph_def x =
  let x = Piqirun.parse_record x in
  let _node, x = Piqirun.parse_repeated_field 1 parse_node_def x in
  let _library, x = Piqirun.parse_optional_field 2 parse_function_def_library x in
  let _version, x = Piqirun.parse_optional_field 3 parse_protobuf_int32 x in
  let _versions, x = Piqirun.parse_optional_field 4 parse_version_def x in
  Piqirun.check_unparsed_fields x;
  {
    Graph_def.node = _node;
    Graph_def.library = _library;
    Graph_def.version = _version;
    Graph_def.versions = _versions;
  }

and parse_node_def x =
  let x = Piqirun.parse_record x in
  let _name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _op, x = Piqirun.parse_optional_field 2 parse_string x in
  let _input, x = Piqirun.parse_repeated_field 3 parse_string x in
  let _device, x = Piqirun.parse_optional_field 4 parse_string x in
  let _attr, x = Piqirun.parse_repeated_field 5 parse_node_def_attr_entry x in
  Piqirun.check_unparsed_fields x;
  {
    Node_def.name = _name;
    Node_def.op = _op;
    Node_def.input = _input;
    Node_def.device = _device;
    Node_def.attr = _attr;
  }

and parse_node_def_attr_entry x =
  let x = Piqirun.parse_record x in
  let _key, x = Piqirun.parse_optional_field 1 parse_string x in
  let _value, x = Piqirun.parse_optional_field 2 parse_attr_value x in
  Piqirun.check_unparsed_fields x;
  {
    Node_def_attr_entry.key = _key;
    Node_def_attr_entry.value = _value;
  }

and parse_data_type x =
  match Piqirun.int32_of_signed_varint x with
    | 0l -> `dt_invalid
    | 1l -> `dt_float
    | 2l -> `dt_double
    | 3l -> `dt_int32
    | 4l -> `dt_uint8
    | 5l -> `dt_int16
    | 6l -> `dt_int8
    | 7l -> `dt_string
    | 8l -> `dt_complex64
    | 9l -> `dt_int64
    | 10l -> `dt_bool
    | 11l -> `dt_qint8
    | 12l -> `dt_quint8
    | 13l -> `dt_qint32
    | 14l -> `dt_bfloat16
    | 15l -> `dt_qint16
    | 16l -> `dt_quint16
    | 17l -> `dt_uint16
    | 18l -> `dt_complex128
    | 19l -> `dt_half
    | 101l -> `dt_float_ref
    | 102l -> `dt_double_ref
    | 103l -> `dt_int32_ref
    | 104l -> `dt_uint8_ref
    | 105l -> `dt_int16_ref
    | 106l -> `dt_int8_ref
    | 107l -> `dt_string_ref
    | 108l -> `dt_complex64_ref
    | 109l -> `dt_int64_ref
    | 110l -> `dt_bool_ref
    | 111l -> `dt_qint8_ref
    | 112l -> `dt_quint8_ref
    | 113l -> `dt_qint32_ref
    | 114l -> `dt_bfloat16_ref
    | 115l -> `dt_qint16_ref
    | 116l -> `dt_quint16_ref
    | 117l -> `dt_uint16_ref
    | 118l -> `dt_complex128_ref
    | 119l -> `dt_half_ref
    | x -> Piqirun.error_enum_const x
and packed_parse_data_type x =
  match Piqirun.int32_of_packed_signed_varint x with
    | 0l -> `dt_invalid
    | 1l -> `dt_float
    | 2l -> `dt_double
    | 3l -> `dt_int32
    | 4l -> `dt_uint8
    | 5l -> `dt_int16
    | 6l -> `dt_int8
    | 7l -> `dt_string
    | 8l -> `dt_complex64
    | 9l -> `dt_int64
    | 10l -> `dt_bool
    | 11l -> `dt_qint8
    | 12l -> `dt_quint8
    | 13l -> `dt_qint32
    | 14l -> `dt_bfloat16
    | 15l -> `dt_qint16
    | 16l -> `dt_quint16
    | 17l -> `dt_uint16
    | 18l -> `dt_complex128
    | 19l -> `dt_half
    | 101l -> `dt_float_ref
    | 102l -> `dt_double_ref
    | 103l -> `dt_int32_ref
    | 104l -> `dt_uint8_ref
    | 105l -> `dt_int16_ref
    | 106l -> `dt_int8_ref
    | 107l -> `dt_string_ref
    | 108l -> `dt_complex64_ref
    | 109l -> `dt_int64_ref
    | 110l -> `dt_bool_ref
    | 111l -> `dt_qint8_ref
    | 112l -> `dt_quint8_ref
    | 113l -> `dt_qint32_ref
    | 114l -> `dt_bfloat16_ref
    | 115l -> `dt_qint16_ref
    | 116l -> `dt_quint16_ref
    | 117l -> `dt_uint16_ref
    | 118l -> `dt_complex128_ref
    | 119l -> `dt_half_ref
    | x -> Piqirun.error_enum_const x


let rec gen__int64 code x = Piqirun.int64_to_zigzag_varint code x
and packed_gen__int64 x = Piqirun.int64_to_packed_zigzag_varint x

and gen__int32 code x = Piqirun.int32_to_zigzag_varint code x
and packed_gen__int32 x = Piqirun.int32_to_packed_zigzag_varint x

and gen__bool code x = Piqirun.bool_to_varint code x
and packed_gen__bool x = Piqirun.bool_to_packed_varint x

and gen__protobuf_int64 code x = Piqirun.int64_to_signed_varint code x
and packed_gen__protobuf_int64 x = Piqirun.int64_to_packed_signed_varint x

and gen__string code x = Piqirun.string_to_block code x

and gen__protobuf_int32 code x = Piqirun.int32_to_signed_varint code x
and packed_gen__protobuf_int32 x = Piqirun.int32_to_packed_signed_varint x

and gen__binary code x = Piqirun.string_to_block code x

and gen__float32 code x = Piqirun.float_to_fixed32 code x
and packed_gen__float32 x = Piqirun.float_to_packed_fixed32 x

and gen__float64 code x = Piqirun.float_to_fixed64 code x
and packed_gen__float64 x = Piqirun.float_to_packed_fixed64 x

and gen__tensor_shape_proto code x =
  let _dim = Piqirun.gen_repeated_field 2 gen__tensor_shape_proto_dim x.Tensor_shape_proto.dim in
  let _unknown_rank = Piqirun.gen_optional_field 3 gen__bool x.Tensor_shape_proto.unknown_rank in
  Piqirun.gen_record code (_dim :: _unknown_rank :: [])

and gen__tensor_shape_proto_dim code x =
  let _size = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Tensor_shape_proto_dim.size in
  let _name = Piqirun.gen_optional_field 2 gen__string x.Tensor_shape_proto_dim.name in
  Piqirun.gen_record code (_size :: _name :: [])

and gen__tensor_proto code x =
  let _dtype = Piqirun.gen_optional_field 1 gen__data_type x.Tensor_proto.dtype in
  let _tensor_shape = Piqirun.gen_optional_field 2 gen__tensor_shape_proto x.Tensor_proto.tensor_shape in
  let _version_number = Piqirun.gen_optional_field 3 gen__protobuf_int32 x.Tensor_proto.version_number in
  let _tensor_content = Piqirun.gen_optional_field 4 gen__binary x.Tensor_proto.tensor_content in
  let _float_val = Piqirun.gen_packed_repeated_field 5 packed_gen__float32 x.Tensor_proto.float_val in
  let _double_val = Piqirun.gen_packed_repeated_field 6 packed_gen__float64 x.Tensor_proto.double_val in
  let _int_val = Piqirun.gen_packed_repeated_field 7 packed_gen__protobuf_int32 x.Tensor_proto.int_val in
  let _string_val = Piqirun.gen_repeated_field 8 gen__binary x.Tensor_proto.string_val in
  let _scomplex_val = Piqirun.gen_packed_repeated_field 9 packed_gen__float32 x.Tensor_proto.scomplex_val in
  let _int64_val = Piqirun.gen_packed_repeated_field 10 packed_gen__protobuf_int64 x.Tensor_proto.int64_val in
  let _bool_val = Piqirun.gen_packed_repeated_field 11 packed_gen__bool x.Tensor_proto.bool_val in
  let _dcomplex_val = Piqirun.gen_packed_repeated_field 12 packed_gen__float64 x.Tensor_proto.dcomplex_val in
  let _half_val = Piqirun.gen_packed_repeated_field 13 packed_gen__protobuf_int32 x.Tensor_proto.half_val in
  Piqirun.gen_record code (_dtype :: _tensor_shape :: _version_number :: _tensor_content :: _float_val :: _double_val :: _int_val :: _string_val :: _scomplex_val :: _int64_val :: _bool_val :: _dcomplex_val :: _half_val :: [])

and gen__attr_value code x =
  let _list = Piqirun.gen_optional_field 1 gen__attr_value_list_value x.Attr_value.list in
  let _s = Piqirun.gen_optional_field 2 gen__binary x.Attr_value.s in
  let _i = Piqirun.gen_optional_field 3 gen__protobuf_int64 x.Attr_value.i in
  let _f = Piqirun.gen_optional_field 4 gen__float32 x.Attr_value.f in
  let _b = Piqirun.gen_optional_field 5 gen__bool x.Attr_value.b in
  let _type_ = Piqirun.gen_optional_field 6 gen__data_type x.Attr_value.type_ in
  let _shape = Piqirun.gen_optional_field 7 gen__tensor_shape_proto x.Attr_value.shape in
  let _tensor = Piqirun.gen_optional_field 8 gen__tensor_proto x.Attr_value.tensor in
  let _placeholder = Piqirun.gen_optional_field 9 gen__string x.Attr_value.placeholder in
  let _func = Piqirun.gen_optional_field 10 gen__name_attr_list x.Attr_value.func in
  Piqirun.gen_record code (_list :: _s :: _i :: _f :: _b :: _type_ :: _shape :: _tensor :: _placeholder :: _func :: [])

and gen__attr_value_list_value code x =
  let _s = Piqirun.gen_repeated_field 2 gen__binary x.Attr_value_list_value.s in
  let _i = Piqirun.gen_packed_repeated_field 3 packed_gen__protobuf_int64 x.Attr_value_list_value.i in
  let _f = Piqirun.gen_packed_repeated_field 4 packed_gen__float32 x.Attr_value_list_value.f in
  let _b = Piqirun.gen_packed_repeated_field 5 packed_gen__bool x.Attr_value_list_value.b in
  let _type_ = Piqirun.gen_packed_repeated_field 6 packed_gen__data_type x.Attr_value_list_value.type_ in
  let _shape = Piqirun.gen_repeated_field 7 gen__tensor_shape_proto x.Attr_value_list_value.shape in
  let _tensor = Piqirun.gen_repeated_field 8 gen__tensor_proto x.Attr_value_list_value.tensor in
  Piqirun.gen_record code (_s :: _i :: _f :: _b :: _type_ :: _shape :: _tensor :: [])

and gen__name_attr_list code x =
  let _name = Piqirun.gen_optional_field 1 gen__string x.Name_attr_list.name in
  let _attr = Piqirun.gen_repeated_field 2 gen__name_attr_list_attr_entry x.Name_attr_list.attr in
  Piqirun.gen_record code (_name :: _attr :: [])

and gen__name_attr_list_attr_entry code x =
  let _key = Piqirun.gen_optional_field 1 gen__string x.Name_attr_list_attr_entry.key in
  let _value = Piqirun.gen_optional_field 2 gen__attr_value x.Name_attr_list_attr_entry.value in
  Piqirun.gen_record code (_key :: _value :: [])

and gen__op_def code x =
  let _name = Piqirun.gen_optional_field 1 gen__string x.Op_def.name in
  let _input_arg = Piqirun.gen_repeated_field 2 gen__op_def_arg_def x.Op_def.input_arg in
  let _output_arg = Piqirun.gen_repeated_field 3 gen__op_def_arg_def x.Op_def.output_arg in
  let _attr = Piqirun.gen_repeated_field 4 gen__op_def_attr_def x.Op_def.attr in
  let _summary = Piqirun.gen_optional_field 5 gen__string x.Op_def.summary in
  let _description = Piqirun.gen_optional_field 6 gen__string x.Op_def.description in
  let _deprecation = Piqirun.gen_optional_field 8 gen__op_deprecation x.Op_def.deprecation in
  let _is_aggregate = Piqirun.gen_optional_field 16 gen__bool x.Op_def.is_aggregate in
  let _is_stateful = Piqirun.gen_optional_field 17 gen__bool x.Op_def.is_stateful in
  let _is_commutative = Piqirun.gen_optional_field 18 gen__bool x.Op_def.is_commutative in
  let _allows_uninitialized_input = Piqirun.gen_optional_field 19 gen__bool x.Op_def.allows_uninitialized_input in
  Piqirun.gen_record code (_name :: _input_arg :: _output_arg :: _attr :: _summary :: _description :: _deprecation :: _is_aggregate :: _is_stateful :: _is_commutative :: _allows_uninitialized_input :: [])

and gen__op_def_arg_def code x =
  let _name = Piqirun.gen_optional_field 1 gen__string x.Op_def_arg_def.name in
  let _description = Piqirun.gen_optional_field 2 gen__string x.Op_def_arg_def.description in
  let _type_ = Piqirun.gen_optional_field 3 gen__data_type x.Op_def_arg_def.type_ in
  let _type_attr = Piqirun.gen_optional_field 4 gen__string x.Op_def_arg_def.type_attr in
  let _number_attr = Piqirun.gen_optional_field 5 gen__string x.Op_def_arg_def.number_attr in
  let _type_list_attr = Piqirun.gen_optional_field 6 gen__string x.Op_def_arg_def.type_list_attr in
  let _is_ref = Piqirun.gen_optional_field 16 gen__bool x.Op_def_arg_def.is_ref in
  Piqirun.gen_record code (_name :: _description :: _type_ :: _type_attr :: _number_attr :: _type_list_attr :: _is_ref :: [])

and gen__op_def_attr_def code x =
  let _name = Piqirun.gen_optional_field 1 gen__string x.Op_def_attr_def.name in
  let _type_ = Piqirun.gen_optional_field 2 gen__string x.Op_def_attr_def.type_ in
  let _default_value = Piqirun.gen_optional_field 3 gen__attr_value x.Op_def_attr_def.default_value in
  let _description = Piqirun.gen_optional_field 4 gen__string x.Op_def_attr_def.description in
  let _has_minimum = Piqirun.gen_optional_field 5 gen__bool x.Op_def_attr_def.has_minimum in
  let _minimum = Piqirun.gen_optional_field 6 gen__protobuf_int64 x.Op_def_attr_def.minimum in
  let _allowed_values = Piqirun.gen_optional_field 7 gen__attr_value x.Op_def_attr_def.allowed_values in
  Piqirun.gen_record code (_name :: _type_ :: _default_value :: _description :: _has_minimum :: _minimum :: _allowed_values :: [])

and gen__op_deprecation code x =
  let _version = Piqirun.gen_optional_field 1 gen__protobuf_int32 x.Op_deprecation.version in
  let _explanation = Piqirun.gen_optional_field 2 gen__string x.Op_deprecation.explanation in
  Piqirun.gen_record code (_version :: _explanation :: [])

and gen__op_list code x =
  let _op = Piqirun.gen_repeated_field 1 gen__op_def x.Op_list.op in
  Piqirun.gen_record code (_op :: [])

and gen__function_def_library code x =
  let _function_ = Piqirun.gen_repeated_field 1 gen__function_def x.Function_def_library.function_ in
  let _gradient = Piqirun.gen_repeated_field 2 gen__gradient_def x.Function_def_library.gradient in
  Piqirun.gen_record code (_function_ :: _gradient :: [])

and gen__function_def code x =
  let _signature = Piqirun.gen_optional_field 1 gen__op_def x.Function_def.signature in
  let _node = Piqirun.gen_repeated_field 2 gen__function_def_node x.Function_def.node in
  Piqirun.gen_record code (_signature :: _node :: [])

and gen__function_def_node code x =
  let _ret = Piqirun.gen_repeated_field 1 gen__string x.Function_def_node.ret in
  let _op = Piqirun.gen_optional_field 2 gen__string x.Function_def_node.op in
  let _arg = Piqirun.gen_repeated_field 3 gen__string x.Function_def_node.arg in
  let _dep = Piqirun.gen_repeated_field 4 gen__string x.Function_def_node.dep in
  let _attr = Piqirun.gen_repeated_field 5 gen__function_def_node_attr_entry x.Function_def_node.attr in
  Piqirun.gen_record code (_ret :: _op :: _arg :: _dep :: _attr :: [])

and gen__function_def_node_attr_entry code x =
  let _key = Piqirun.gen_optional_field 1 gen__string x.Function_def_node_attr_entry.key in
  let _value = Piqirun.gen_optional_field 2 gen__attr_value x.Function_def_node_attr_entry.value in
  Piqirun.gen_record code (_key :: _value :: [])

and gen__gradient_def code x =
  let _function_name = Piqirun.gen_optional_field 1 gen__string x.Gradient_def.function_name in
  let _gradient_func = Piqirun.gen_optional_field 2 gen__string x.Gradient_def.gradient_func in
  Piqirun.gen_record code (_function_name :: _gradient_func :: [])

and gen__version_def code x =
  let _producer = Piqirun.gen_optional_field 1 gen__protobuf_int32 x.Version_def.producer in
  let _min_consumer = Piqirun.gen_optional_field 2 gen__protobuf_int32 x.Version_def.min_consumer in
  let _bad_consumers = Piqirun.gen_repeated_field 3 gen__protobuf_int32 x.Version_def.bad_consumers in
  Piqirun.gen_record code (_producer :: _min_consumer :: _bad_consumers :: [])

and gen__graph_def code x =
  let _node = Piqirun.gen_repeated_field 1 gen__node_def x.Graph_def.node in
  let _library = Piqirun.gen_optional_field 2 gen__function_def_library x.Graph_def.library in
  let _version = Piqirun.gen_optional_field 3 gen__protobuf_int32 x.Graph_def.version in
  let _versions = Piqirun.gen_optional_field 4 gen__version_def x.Graph_def.versions in
  Piqirun.gen_record code (_node :: _library :: _version :: _versions :: [])

and gen__node_def code x =
  let _name = Piqirun.gen_optional_field 1 gen__string x.Node_def.name in
  let _op = Piqirun.gen_optional_field 2 gen__string x.Node_def.op in
  let _input = Piqirun.gen_repeated_field 3 gen__string x.Node_def.input in
  let _device = Piqirun.gen_optional_field 4 gen__string x.Node_def.device in
  let _attr = Piqirun.gen_repeated_field 5 gen__node_def_attr_entry x.Node_def.attr in
  Piqirun.gen_record code (_name :: _op :: _input :: _device :: _attr :: [])

and gen__node_def_attr_entry code x =
  let _key = Piqirun.gen_optional_field 1 gen__string x.Node_def_attr_entry.key in
  let _value = Piqirun.gen_optional_field 2 gen__attr_value x.Node_def_attr_entry.value in
  Piqirun.gen_record code (_key :: _value :: [])

and gen__data_type code x =
  Piqirun.int32_to_signed_varint code (match x with
    | `dt_invalid -> 0l
    | `dt_float -> 1l
    | `dt_double -> 2l
    | `dt_int32 -> 3l
    | `dt_uint8 -> 4l
    | `dt_int16 -> 5l
    | `dt_int8 -> 6l
    | `dt_string -> 7l
    | `dt_complex64 -> 8l
    | `dt_int64 -> 9l
    | `dt_bool -> 10l
    | `dt_qint8 -> 11l
    | `dt_quint8 -> 12l
    | `dt_qint32 -> 13l
    | `dt_bfloat16 -> 14l
    | `dt_qint16 -> 15l
    | `dt_quint16 -> 16l
    | `dt_uint16 -> 17l
    | `dt_complex128 -> 18l
    | `dt_half -> 19l
    | `dt_float_ref -> 101l
    | `dt_double_ref -> 102l
    | `dt_int32_ref -> 103l
    | `dt_uint8_ref -> 104l
    | `dt_int16_ref -> 105l
    | `dt_int8_ref -> 106l
    | `dt_string_ref -> 107l
    | `dt_complex64_ref -> 108l
    | `dt_int64_ref -> 109l
    | `dt_bool_ref -> 110l
    | `dt_qint8_ref -> 111l
    | `dt_quint8_ref -> 112l
    | `dt_qint32_ref -> 113l
    | `dt_bfloat16_ref -> 114l
    | `dt_qint16_ref -> 115l
    | `dt_quint16_ref -> 116l
    | `dt_uint16_ref -> 117l
    | `dt_complex128_ref -> 118l
    | `dt_half_ref -> 119l
  )
and packed_gen__data_type x =
  Piqirun.int32_to_packed_signed_varint (match x with
    | `dt_invalid -> 0l
    | `dt_float -> 1l
    | `dt_double -> 2l
    | `dt_int32 -> 3l
    | `dt_uint8 -> 4l
    | `dt_int16 -> 5l
    | `dt_int8 -> 6l
    | `dt_string -> 7l
    | `dt_complex64 -> 8l
    | `dt_int64 -> 9l
    | `dt_bool -> 10l
    | `dt_qint8 -> 11l
    | `dt_quint8 -> 12l
    | `dt_qint32 -> 13l
    | `dt_bfloat16 -> 14l
    | `dt_qint16 -> 15l
    | `dt_quint16 -> 16l
    | `dt_uint16 -> 17l
    | `dt_complex128 -> 18l
    | `dt_half -> 19l
    | `dt_float_ref -> 101l
    | `dt_double_ref -> 102l
    | `dt_int32_ref -> 103l
    | `dt_uint8_ref -> 104l
    | `dt_int16_ref -> 105l
    | `dt_int8_ref -> 106l
    | `dt_string_ref -> 107l
    | `dt_complex64_ref -> 108l
    | `dt_int64_ref -> 109l
    | `dt_bool_ref -> 110l
    | `dt_qint8_ref -> 111l
    | `dt_quint8_ref -> 112l
    | `dt_qint32_ref -> 113l
    | `dt_bfloat16_ref -> 114l
    | `dt_qint16_ref -> 115l
    | `dt_quint16_ref -> 116l
    | `dt_uint16_ref -> 117l
    | `dt_complex128_ref -> 118l
    | `dt_half_ref -> 119l
  )


let gen_int64 x = gen__int64 (-1) x
let gen_int32 x = gen__int32 (-1) x
let gen_bool x = gen__bool (-1) x
let gen_protobuf_int64 x = gen__protobuf_int64 (-1) x
let gen_string x = gen__string (-1) x
let gen_protobuf_int32 x = gen__protobuf_int32 (-1) x
let gen_binary x = gen__binary (-1) x
let gen_float32 x = gen__float32 (-1) x
let gen_float64 x = gen__float64 (-1) x
let gen_tensor_shape_proto x = gen__tensor_shape_proto (-1) x
let gen_tensor_shape_proto_dim x = gen__tensor_shape_proto_dim (-1) x
let gen_tensor_proto x = gen__tensor_proto (-1) x
let gen_attr_value x = gen__attr_value (-1) x
let gen_attr_value_list_value x = gen__attr_value_list_value (-1) x
let gen_name_attr_list x = gen__name_attr_list (-1) x
let gen_name_attr_list_attr_entry x = gen__name_attr_list_attr_entry (-1) x
let gen_op_def x = gen__op_def (-1) x
let gen_op_def_arg_def x = gen__op_def_arg_def (-1) x
let gen_op_def_attr_def x = gen__op_def_attr_def (-1) x
let gen_op_deprecation x = gen__op_deprecation (-1) x
let gen_op_list x = gen__op_list (-1) x
let gen_function_def_library x = gen__function_def_library (-1) x
let gen_function_def x = gen__function_def (-1) x
let gen_function_def_node x = gen__function_def_node (-1) x
let gen_function_def_node_attr_entry x = gen__function_def_node_attr_entry (-1) x
let gen_gradient_def x = gen__gradient_def (-1) x
let gen_version_def x = gen__version_def (-1) x
let gen_graph_def x = gen__graph_def (-1) x
let gen_node_def x = gen__node_def (-1) x
let gen_node_def_attr_entry x = gen__node_def_attr_entry (-1) x
let gen_data_type x = gen__data_type (-1) x


let rec default_int64 () = 0L
and default_int32 () = 0l
and default_bool () = false
and default_protobuf_int64 () = default_int64 ()
and default_string () = ""
and default_protobuf_int32 () = default_int32 ()
and default_binary () = ""
and default_float32 () = 0.0
and default_float64 () = 0.0
and default_tensor_shape_proto () =
  {
    Tensor_shape_proto.dim = [];
    Tensor_shape_proto.unknown_rank = None;
  }
and default_tensor_shape_proto_dim () =
  {
    Tensor_shape_proto_dim.size = None;
    Tensor_shape_proto_dim.name = None;
  }
and default_tensor_proto () =
  {
    Tensor_proto.dtype = None;
    Tensor_proto.tensor_shape = None;
    Tensor_proto.version_number = None;
    Tensor_proto.tensor_content = None;
    Tensor_proto.float_val = [];
    Tensor_proto.double_val = [];
    Tensor_proto.int_val = [];
    Tensor_proto.string_val = [];
    Tensor_proto.scomplex_val = [];
    Tensor_proto.int64_val = [];
    Tensor_proto.bool_val = [];
    Tensor_proto.dcomplex_val = [];
    Tensor_proto.half_val = [];
  }
and default_attr_value () =
  {
    Attr_value.list = None;
    Attr_value.s = None;
    Attr_value.i = None;
    Attr_value.f = None;
    Attr_value.b = None;
    Attr_value.type_ = None;
    Attr_value.shape = None;
    Attr_value.tensor = None;
    Attr_value.placeholder = None;
    Attr_value.func = None;
  }
and default_attr_value_list_value () =
  {
    Attr_value_list_value.s = [];
    Attr_value_list_value.i = [];
    Attr_value_list_value.f = [];
    Attr_value_list_value.b = [];
    Attr_value_list_value.type_ = [];
    Attr_value_list_value.shape = [];
    Attr_value_list_value.tensor = [];
  }
and default_name_attr_list () =
  {
    Name_attr_list.name = None;
    Name_attr_list.attr = [];
  }
and default_name_attr_list_attr_entry () =
  {
    Name_attr_list_attr_entry.key = None;
    Name_attr_list_attr_entry.value = None;
  }
and default_op_def () =
  {
    Op_def.name = None;
    Op_def.input_arg = [];
    Op_def.output_arg = [];
    Op_def.attr = [];
    Op_def.summary = None;
    Op_def.description = None;
    Op_def.deprecation = None;
    Op_def.is_aggregate = None;
    Op_def.is_stateful = None;
    Op_def.is_commutative = None;
    Op_def.allows_uninitialized_input = None;
  }
and default_op_def_arg_def () =
  {
    Op_def_arg_def.name = None;
    Op_def_arg_def.description = None;
    Op_def_arg_def.type_ = None;
    Op_def_arg_def.type_attr = None;
    Op_def_arg_def.number_attr = None;
    Op_def_arg_def.type_list_attr = None;
    Op_def_arg_def.is_ref = None;
  }
and default_op_def_attr_def () =
  {
    Op_def_attr_def.name = None;
    Op_def_attr_def.type_ = None;
    Op_def_attr_def.default_value = None;
    Op_def_attr_def.description = None;
    Op_def_attr_def.has_minimum = None;
    Op_def_attr_def.minimum = None;
    Op_def_attr_def.allowed_values = None;
  }
and default_op_deprecation () =
  {
    Op_deprecation.version = None;
    Op_deprecation.explanation = None;
  }
and default_op_list () =
  {
    Op_list.op = [];
  }
and default_function_def_library () =
  {
    Function_def_library.function_ = [];
    Function_def_library.gradient = [];
  }
and default_function_def () =
  {
    Function_def.signature = None;
    Function_def.node = [];
  }
and default_function_def_node () =
  {
    Function_def_node.ret = [];
    Function_def_node.op = None;
    Function_def_node.arg = [];
    Function_def_node.dep = [];
    Function_def_node.attr = [];
  }
and default_function_def_node_attr_entry () =
  {
    Function_def_node_attr_entry.key = None;
    Function_def_node_attr_entry.value = None;
  }
and default_gradient_def () =
  {
    Gradient_def.function_name = None;
    Gradient_def.gradient_func = None;
  }
and default_version_def () =
  {
    Version_def.producer = None;
    Version_def.min_consumer = None;
    Version_def.bad_consumers = [];
  }
and default_graph_def () =
  {
    Graph_def.node = [];
    Graph_def.library = None;
    Graph_def.version = None;
    Graph_def.versions = None;
  }
and default_node_def () =
  {
    Node_def.name = None;
    Node_def.op = None;
    Node_def.input = [];
    Node_def.device = None;
    Node_def.attr = [];
  }
and default_node_def_attr_entry () =
  {
    Node_def_attr_entry.key = None;
    Node_def_attr_entry.value = None;
  }
and default_data_type () = `dt_invalid


include Graph_piqi
