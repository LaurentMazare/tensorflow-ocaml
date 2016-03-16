module rec All_piqi:
  sig
    type uint64 = int64
    type float32 = float
    type uint32 = int32
    type float64 = float
    type protobuf_int64 = int64
    type binary = string
    type uint64_fixed = All_piqi.uint64
    type protobuf_int32 = int32
    type bus_adjacency =
      [
        | `bus_0
        | `bus_1
        | `bus_any
        | `bus_num_adjacencies
      ]
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
      ]
    type allocation_description = Allocation_description.t
    type attr_value = Attr_value.t
    type attr_value_list_value = Attr_value_list_value.t
    type name_attr_list = Name_attr_list.t
    type name_attr_list_attr_entry = Name_attr_list_attr_entry.t
    type device_attributes = Device_attributes.t
    type function_def_library = Function_def_library.t
    type function_def = Function_def.t
    type function_def_node = Function_def_node.t
    type function_def_node_attr_entry = Function_def_node_attr_entry.t
    type graph_def = Graph_def.t
    type node_def = Node_def.t
    type node_def_attr_entry = Node_def_attr_entry.t
    type kernel_def = Kernel_def.t
    type kernel_def_attr_constraint = Kernel_def_attr_constraint.t
    type memory_log_step = Memory_log_step.t
    type memory_log_tensor_allocation = Memory_log_tensor_allocation.t
    type memory_log_tensor_deallocation = Memory_log_tensor_deallocation.t
    type memory_log_tensor_output = Memory_log_tensor_output.t
    type memory_log_raw_allocation = Memory_log_raw_allocation.t
    type memory_log_raw_deallocation = Memory_log_raw_deallocation.t
    type op_def = Op_def.t
    type op_def_arg_def = Op_def_arg_def.t
    type op_def_attr_def = Op_def_attr_def.t
    type op_list = Op_list.t
    type allocator_memory_used = Allocator_memory_used.t
    type node_output = Node_output.t
    type node_exec_stats = Node_exec_stats.t
    type device_step_stats = Device_step_stats.t
    type step_stats = Step_stats.t
    type histogram_proto = Histogram_proto.t
    type summary = Summary.t
    type summary_image = Summary_image.t
    type summary_value = Summary_value.t
    type tensor_description = Tensor_description.t
    type tensor_proto = Tensor_proto.t
    type tensor_shape_proto = Tensor_shape_proto.t
    type tensor_shape_proto_dim = Tensor_shape_proto_dim.t
    type tensor_slice_proto = Tensor_slice_proto.t
    type tensor_slice_proto_extent = Tensor_slice_proto_extent.t
    type variable_def = Variable_def.t
    type save_slice_info_def = Save_slice_info_def.t
    type version_def = Version_def.t
  end = All_piqi
and Allocation_description:
  sig
    type t = {
      mutable requested_bytes: All_piqi.protobuf_int64 option;
      mutable allocated_bytes: All_piqi.protobuf_int64 option;
      mutable allocator_name: string option;
      mutable allocation_id: All_piqi.protobuf_int64 option;
      mutable has_single_reference: bool option;
      mutable ptr: All_piqi.uint64 option;
    }
  end = Allocation_description
and Attr_value:
  sig
    type t = {
      mutable s: All_piqi.binary option;
      mutable i: All_piqi.protobuf_int64 option;
      mutable f: All_piqi.float32 option;
      mutable b: bool option;
      mutable type_: All_piqi.data_type option;
      mutable shape: All_piqi.tensor_shape_proto option;
      mutable tensor: All_piqi.tensor_proto option;
      mutable list: All_piqi.attr_value_list_value option;
      mutable func: All_piqi.name_attr_list option;
      mutable placeholder: string option;
    }
  end = Attr_value
and Attr_value_list_value:
  sig
    type t = {
      mutable s: All_piqi.binary list;
      mutable i: All_piqi.protobuf_int64 list;
      mutable f: All_piqi.float32 list;
      mutable b: bool list;
      mutable type_: All_piqi.data_type list;
      mutable shape: All_piqi.tensor_shape_proto list;
      mutable tensor: All_piqi.tensor_proto list;
    }
  end = Attr_value_list_value
and Name_attr_list:
  sig
    type t = {
      mutable name: string option;
      mutable attr: All_piqi.name_attr_list_attr_entry list;
    }
  end = Name_attr_list
and Name_attr_list_attr_entry:
  sig
    type t = {
      mutable key: string option;
      mutable value: All_piqi.attr_value option;
    }
  end = Name_attr_list_attr_entry
and Device_attributes:
  sig
    type t = {
      mutable name: string option;
      mutable device_type: string option;
      mutable memory_limit: All_piqi.protobuf_int64 option;
      mutable bus_adjacency: All_piqi.bus_adjacency option;
      mutable incarnation: All_piqi.uint64_fixed option;
      mutable physical_device_desc: string option;
    }
  end = Device_attributes
and Function_def_library:
  sig
    type t = {
      mutable function_: All_piqi.function_def list;
    }
  end = Function_def_library
and Function_def:
  sig
    type t = {
      mutable signature: All_piqi.op_def option;
      mutable node: All_piqi.function_def_node list;
    }
  end = Function_def
and Function_def_node:
  sig
    type t = {
      mutable ret: string list;
      mutable op: string option;
      mutable arg: string list;
      mutable dep: string list;
      mutable attr: All_piqi.function_def_node_attr_entry list;
    }
  end = Function_def_node
and Function_def_node_attr_entry:
  sig
    type t = {
      mutable key: string option;
      mutable value: All_piqi.attr_value option;
    }
  end = Function_def_node_attr_entry
and Graph_def:
  sig
    type t = {
      mutable node: All_piqi.node_def list;
      mutable versions: All_piqi.version_def option;
      mutable version: All_piqi.protobuf_int32 option;
      mutable library: All_piqi.function_def_library option;
    }
  end = Graph_def
and Node_def:
  sig
    type t = {
      mutable name: string option;
      mutable op: string option;
      mutable input: string list;
      mutable device: string option;
      mutable attr: All_piqi.node_def_attr_entry list;
    }
  end = Node_def
and Node_def_attr_entry:
  sig
    type t = {
      mutable key: string option;
      mutable value: All_piqi.attr_value option;
    }
  end = Node_def_attr_entry
and Kernel_def:
  sig
    type t = {
      mutable op: string option;
      mutable device_type: string option;
      mutable constraint_: All_piqi.kernel_def_attr_constraint list;
      mutable host_memory_arg: string list;
      mutable label: string option;
    }
  end = Kernel_def
and Kernel_def_attr_constraint:
  sig
    type t = {
      mutable name: string option;
      mutable allowed_values: All_piqi.attr_value option;
    }
  end = Kernel_def_attr_constraint
and Memory_log_step:
  sig
    type t = {
      mutable step_id: All_piqi.protobuf_int64 option;
      mutable handle: string option;
    }
  end = Memory_log_step
and Memory_log_tensor_allocation:
  sig
    type t = {
      mutable step_id: All_piqi.protobuf_int64 option;
      mutable kernel_name: string option;
      mutable tensor: All_piqi.tensor_description option;
    }
  end = Memory_log_tensor_allocation
and Memory_log_tensor_deallocation:
  sig
    type t = {
      mutable allocation_id: All_piqi.protobuf_int64 option;
      mutable allocator_name: string option;
    }
  end = Memory_log_tensor_deallocation
and Memory_log_tensor_output:
  sig
    type t = {
      mutable step_id: All_piqi.protobuf_int64 option;
      mutable kernel_name: string option;
      mutable index: All_piqi.protobuf_int32 option;
      mutable tensor: All_piqi.tensor_description option;
    }
  end = Memory_log_tensor_output
and Memory_log_raw_allocation:
  sig
    type t = {
      mutable step_id: All_piqi.protobuf_int64 option;
      mutable operation: string option;
      mutable num_bytes: All_piqi.protobuf_int64 option;
      mutable ptr: All_piqi.uint64 option;
      mutable allocation_id: All_piqi.protobuf_int64 option;
      mutable allocator_name: string option;
    }
  end = Memory_log_raw_allocation
and Memory_log_raw_deallocation:
  sig
    type t = {
      mutable step_id: All_piqi.protobuf_int64 option;
      mutable operation: string option;
      mutable allocation_id: All_piqi.protobuf_int64 option;
      mutable allocator_name: string option;
      mutable deferred: bool option;
    }
  end = Memory_log_raw_deallocation
and Op_def:
  sig
    type t = {
      mutable name: string option;
      mutable input_arg: All_piqi.op_def_arg_def list;
      mutable output_arg: All_piqi.op_def_arg_def list;
      mutable attr: All_piqi.op_def_attr_def list;
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
      mutable type_: All_piqi.data_type option;
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
      mutable default_value: All_piqi.attr_value option;
      mutable description: string option;
      mutable has_minimum: bool option;
      mutable minimum: All_piqi.protobuf_int64 option;
      mutable allowed_values: All_piqi.attr_value option;
    }
  end = Op_def_attr_def
and Op_list:
  sig
    type t = {
      mutable op: All_piqi.op_def list;
    }
  end = Op_list
and Allocator_memory_used:
  sig
    type t = {
      mutable allocator_name: string option;
      mutable total_bytes: All_piqi.protobuf_int64 option;
      mutable peak_bytes: All_piqi.protobuf_int64 option;
    }
  end = Allocator_memory_used
and Node_output:
  sig
    type t = {
      mutable slot: All_piqi.protobuf_int32 option;
      mutable tensor_description: All_piqi.tensor_description option;
    }
  end = Node_output
and Node_exec_stats:
  sig
    type t = {
      mutable node_name: string option;
      mutable all_start_micros: All_piqi.protobuf_int64 option;
      mutable op_start_rel_micros: All_piqi.protobuf_int64 option;
      mutable op_end_rel_micros: All_piqi.protobuf_int64 option;
      mutable all_end_rel_micros: All_piqi.protobuf_int64 option;
      mutable memory: All_piqi.allocator_memory_used list;
      mutable output: All_piqi.node_output list;
      mutable timeline_label: string option;
      mutable scheduled_micros: All_piqi.protobuf_int64 option;
      mutable thread_id: All_piqi.uint32 option;
      mutable referenced_tensor: All_piqi.allocation_description list;
    }
  end = Node_exec_stats
and Device_step_stats:
  sig
    type t = {
      mutable device: string option;
      mutable node_stats: All_piqi.node_exec_stats list;
    }
  end = Device_step_stats
and Step_stats:
  sig
    type t = {
      mutable dev_stats: All_piqi.device_step_stats list;
    }
  end = Step_stats
and Histogram_proto:
  sig
    type t = {
      mutable min: All_piqi.float64 option;
      mutable max: All_piqi.float64 option;
      mutable num: All_piqi.float64 option;
      mutable sum: All_piqi.float64 option;
      mutable sum_squares: All_piqi.float64 option;
      mutable bucket_limit: All_piqi.float64 list;
      mutable bucket: All_piqi.float64 list;
    }
  end = Histogram_proto
and Summary:
  sig
    type t = {
      mutable value: All_piqi.summary_value list;
    }
  end = Summary
and Summary_image:
  sig
    type t = {
      mutable height: All_piqi.protobuf_int32 option;
      mutable width: All_piqi.protobuf_int32 option;
      mutable colorspace: All_piqi.protobuf_int32 option;
      mutable encoded_image_string: All_piqi.binary option;
    }
  end = Summary_image
and Summary_value:
  sig
    type t = {
      mutable tag: string option;
      mutable simple_value: All_piqi.float32 option;
      mutable obsolete_old_style_histogram: All_piqi.binary option;
      mutable image: All_piqi.summary_image option;
      mutable histo: All_piqi.histogram_proto option;
    }
  end = Summary_value
and Tensor_description:
  sig
    type t = {
      mutable dtype: All_piqi.data_type option;
      mutable shape: All_piqi.tensor_shape_proto option;
      mutable allocation_description: All_piqi.allocation_description option;
    }
  end = Tensor_description
and Tensor_proto:
  sig
    type t = {
      mutable dtype: All_piqi.data_type option;
      mutable tensor_shape: All_piqi.tensor_shape_proto option;
      mutable version_number: All_piqi.protobuf_int32 option;
      mutable tensor_content: All_piqi.binary option;
      mutable float_val: All_piqi.float32 list;
      mutable double_val: All_piqi.float64 list;
      mutable int_val: All_piqi.protobuf_int32 list;
      mutable string_val: All_piqi.binary list;
      mutable scomplex_val: All_piqi.float32 list;
      mutable int64_val: All_piqi.protobuf_int64 list;
      mutable bool_val: bool list;
      mutable dcomplex_val: All_piqi.float64 list;
    }
  end = Tensor_proto
and Tensor_shape_proto:
  sig
    type t = {
      mutable dim: All_piqi.tensor_shape_proto_dim list;
      mutable unknown_rank: bool option;
    }
  end = Tensor_shape_proto
and Tensor_shape_proto_dim:
  sig
    type t = {
      mutable size: All_piqi.protobuf_int64 option;
      mutable name: string option;
    }
  end = Tensor_shape_proto_dim
and Tensor_slice_proto:
  sig
    type t = {
      mutable extent: All_piqi.tensor_slice_proto_extent list;
    }
  end = Tensor_slice_proto
and Tensor_slice_proto_extent:
  sig
    type t = {
      mutable start: All_piqi.protobuf_int64 option;
      mutable length: All_piqi.protobuf_int64 option;
    }
  end = Tensor_slice_proto_extent
and Variable_def:
  sig
    type t = {
      mutable variable_name: string option;
      mutable initializer_name: string option;
      mutable snapshot_name: string option;
      mutable save_slice_info_def: All_piqi.save_slice_info_def option;
    }
  end = Variable_def
and Save_slice_info_def:
  sig
    type t = {
      mutable full_name: string option;
      mutable full_shape: All_piqi.protobuf_int32 list;
      mutable var_offset: All_piqi.protobuf_int32 list;
      mutable var_shape: All_piqi.protobuf_int32 list;
    }
  end = Save_slice_info_def
and Version_def:
  sig
    type t = {
      mutable producer: All_piqi.protobuf_int32 option;
      mutable min_consumer: All_piqi.protobuf_int32 option;
      mutable bad_consumers: All_piqi.protobuf_int32 list;
    }
  end = Version_def


let rec parse_int64 x = Piqirun.int64_of_zigzag_varint x
and packed_parse_int64 x = Piqirun.int64_of_packed_zigzag_varint x

and parse_uint64 x = Piqirun.int64_of_varint x
and packed_parse_uint64 x = Piqirun.int64_of_packed_varint x

and parse_int32 x = Piqirun.int32_of_zigzag_varint x
and packed_parse_int32 x = Piqirun.int32_of_packed_zigzag_varint x

and parse_protobuf_int64 x = Piqirun.int64_of_signed_varint x
and packed_parse_protobuf_int64 x = Piqirun.int64_of_packed_signed_varint x

and parse_string x = Piqirun.string_of_block x

and parse_bool x = Piqirun.bool_of_varint x
and packed_parse_bool x = Piqirun.bool_of_packed_varint x

and parse_binary x = Piqirun.string_of_block x

and parse_float32 x = Piqirun.float_of_fixed32 x
and packed_parse_float32 x = Piqirun.float_of_packed_fixed32 x

and parse_uint64_fixed x = Piqirun.int64_of_fixed64 x
and packed_parse_uint64_fixed x = Piqirun.int64_of_packed_fixed64 x

and parse_protobuf_int32 x = Piqirun.int32_of_signed_varint x
and packed_parse_protobuf_int32 x = Piqirun.int32_of_packed_signed_varint x

and parse_uint32 x = Piqirun.int32_of_varint x
and packed_parse_uint32 x = Piqirun.int32_of_packed_varint x

and parse_float64 x = Piqirun.float_of_fixed64 x
and packed_parse_float64 x = Piqirun.float_of_packed_fixed64 x

and parse_allocation_description x =
  let x = Piqirun.parse_record x in
  let _requested_bytes, x = Piqirun.parse_optional_field 1 parse_protobuf_int64 x in
  let _allocated_bytes, x = Piqirun.parse_optional_field 2 parse_protobuf_int64 x in
  let _allocator_name, x = Piqirun.parse_optional_field 3 parse_string x in
  let _allocation_id, x = Piqirun.parse_optional_field 4 parse_protobuf_int64 x in
  let _has_single_reference, x = Piqirun.parse_optional_field 5 parse_bool x in
  let _ptr, x = Piqirun.parse_optional_field 6 parse_uint64 x in
  Piqirun.check_unparsed_fields x;
  {
    Allocation_description.requested_bytes = _requested_bytes;
    Allocation_description.allocated_bytes = _allocated_bytes;
    Allocation_description.allocator_name = _allocator_name;
    Allocation_description.allocation_id = _allocation_id;
    Allocation_description.has_single_reference = _has_single_reference;
    Allocation_description.ptr = _ptr;
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

and parse_device_attributes x =
  let x = Piqirun.parse_record x in
  let _name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _device_type, x = Piqirun.parse_optional_field 2 parse_string x in
  let _memory_limit, x = Piqirun.parse_optional_field 4 parse_protobuf_int64 x in
  let _bus_adjacency, x = Piqirun.parse_optional_field 5 parse_bus_adjacency x in
  let _incarnation, x = Piqirun.parse_optional_field 6 parse_uint64_fixed x in
  let _physical_device_desc, x = Piqirun.parse_optional_field 7 parse_string x in
  Piqirun.check_unparsed_fields x;
  {
    Device_attributes.name = _name;
    Device_attributes.device_type = _device_type;
    Device_attributes.memory_limit = _memory_limit;
    Device_attributes.bus_adjacency = _bus_adjacency;
    Device_attributes.incarnation = _incarnation;
    Device_attributes.physical_device_desc = _physical_device_desc;
  }

and parse_function_def_library x =
  let x = Piqirun.parse_record x in
  let _function_, x = Piqirun.parse_repeated_field 1 parse_function_def x in
  Piqirun.check_unparsed_fields x;
  {
    Function_def_library.function_ = _function_;
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

and parse_kernel_def x =
  let x = Piqirun.parse_record x in
  let _op, x = Piqirun.parse_optional_field 1 parse_string x in
  let _device_type, x = Piqirun.parse_optional_field 2 parse_string x in
  let _constraint_, x = Piqirun.parse_repeated_field 3 parse_kernel_def_attr_constraint x in
  let _host_memory_arg, x = Piqirun.parse_repeated_field 4 parse_string x in
  let _label, x = Piqirun.parse_optional_field 5 parse_string x in
  Piqirun.check_unparsed_fields x;
  {
    Kernel_def.op = _op;
    Kernel_def.device_type = _device_type;
    Kernel_def.constraint_ = _constraint_;
    Kernel_def.host_memory_arg = _host_memory_arg;
    Kernel_def.label = _label;
  }

and parse_kernel_def_attr_constraint x =
  let x = Piqirun.parse_record x in
  let _name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _allowed_values, x = Piqirun.parse_optional_field 2 parse_attr_value x in
  Piqirun.check_unparsed_fields x;
  {
    Kernel_def_attr_constraint.name = _name;
    Kernel_def_attr_constraint.allowed_values = _allowed_values;
  }

and parse_memory_log_step x =
  let x = Piqirun.parse_record x in
  let _step_id, x = Piqirun.parse_optional_field 1 parse_protobuf_int64 x in
  let _handle, x = Piqirun.parse_optional_field 2 parse_string x in
  Piqirun.check_unparsed_fields x;
  {
    Memory_log_step.step_id = _step_id;
    Memory_log_step.handle = _handle;
  }

and parse_memory_log_tensor_allocation x =
  let x = Piqirun.parse_record x in
  let _step_id, x = Piqirun.parse_optional_field 1 parse_protobuf_int64 x in
  let _kernel_name, x = Piqirun.parse_optional_field 2 parse_string x in
  let _tensor, x = Piqirun.parse_optional_field 3 parse_tensor_description x in
  Piqirun.check_unparsed_fields x;
  {
    Memory_log_tensor_allocation.step_id = _step_id;
    Memory_log_tensor_allocation.kernel_name = _kernel_name;
    Memory_log_tensor_allocation.tensor = _tensor;
  }

and parse_memory_log_tensor_deallocation x =
  let x = Piqirun.parse_record x in
  let _allocation_id, x = Piqirun.parse_optional_field 1 parse_protobuf_int64 x in
  let _allocator_name, x = Piqirun.parse_optional_field 2 parse_string x in
  Piqirun.check_unparsed_fields x;
  {
    Memory_log_tensor_deallocation.allocation_id = _allocation_id;
    Memory_log_tensor_deallocation.allocator_name = _allocator_name;
  }

and parse_memory_log_tensor_output x =
  let x = Piqirun.parse_record x in
  let _step_id, x = Piqirun.parse_optional_field 1 parse_protobuf_int64 x in
  let _kernel_name, x = Piqirun.parse_optional_field 2 parse_string x in
  let _index, x = Piqirun.parse_optional_field 3 parse_protobuf_int32 x in
  let _tensor, x = Piqirun.parse_optional_field 4 parse_tensor_description x in
  Piqirun.check_unparsed_fields x;
  {
    Memory_log_tensor_output.step_id = _step_id;
    Memory_log_tensor_output.kernel_name = _kernel_name;
    Memory_log_tensor_output.index = _index;
    Memory_log_tensor_output.tensor = _tensor;
  }

and parse_memory_log_raw_allocation x =
  let x = Piqirun.parse_record x in
  let _step_id, x = Piqirun.parse_optional_field 1 parse_protobuf_int64 x in
  let _operation, x = Piqirun.parse_optional_field 2 parse_string x in
  let _num_bytes, x = Piqirun.parse_optional_field 3 parse_protobuf_int64 x in
  let _ptr, x = Piqirun.parse_optional_field 4 parse_uint64 x in
  let _allocation_id, x = Piqirun.parse_optional_field 5 parse_protobuf_int64 x in
  let _allocator_name, x = Piqirun.parse_optional_field 6 parse_string x in
  Piqirun.check_unparsed_fields x;
  {
    Memory_log_raw_allocation.step_id = _step_id;
    Memory_log_raw_allocation.operation = _operation;
    Memory_log_raw_allocation.num_bytes = _num_bytes;
    Memory_log_raw_allocation.ptr = _ptr;
    Memory_log_raw_allocation.allocation_id = _allocation_id;
    Memory_log_raw_allocation.allocator_name = _allocator_name;
  }

and parse_memory_log_raw_deallocation x =
  let x = Piqirun.parse_record x in
  let _step_id, x = Piqirun.parse_optional_field 1 parse_protobuf_int64 x in
  let _operation, x = Piqirun.parse_optional_field 2 parse_string x in
  let _allocation_id, x = Piqirun.parse_optional_field 3 parse_protobuf_int64 x in
  let _allocator_name, x = Piqirun.parse_optional_field 4 parse_string x in
  let _deferred, x = Piqirun.parse_optional_field 5 parse_bool x in
  Piqirun.check_unparsed_fields x;
  {
    Memory_log_raw_deallocation.step_id = _step_id;
    Memory_log_raw_deallocation.operation = _operation;
    Memory_log_raw_deallocation.allocation_id = _allocation_id;
    Memory_log_raw_deallocation.allocator_name = _allocator_name;
    Memory_log_raw_deallocation.deferred = _deferred;
  }

and parse_op_def x =
  let x = Piqirun.parse_record x in
  let _name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _input_arg, x = Piqirun.parse_repeated_field 2 parse_op_def_arg_def x in
  let _output_arg, x = Piqirun.parse_repeated_field 3 parse_op_def_arg_def x in
  let _attr, x = Piqirun.parse_repeated_field 4 parse_op_def_attr_def x in
  let _summary, x = Piqirun.parse_optional_field 5 parse_string x in
  let _description, x = Piqirun.parse_optional_field 6 parse_string x in
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

and parse_op_list x =
  let x = Piqirun.parse_record x in
  let _op, x = Piqirun.parse_repeated_field 1 parse_op_def x in
  Piqirun.check_unparsed_fields x;
  {
    Op_list.op = _op;
  }

and parse_allocator_memory_used x =
  let x = Piqirun.parse_record x in
  let _allocator_name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _total_bytes, x = Piqirun.parse_optional_field 2 parse_protobuf_int64 x in
  let _peak_bytes, x = Piqirun.parse_optional_field 3 parse_protobuf_int64 x in
  Piqirun.check_unparsed_fields x;
  {
    Allocator_memory_used.allocator_name = _allocator_name;
    Allocator_memory_used.total_bytes = _total_bytes;
    Allocator_memory_used.peak_bytes = _peak_bytes;
  }

and parse_node_output x =
  let x = Piqirun.parse_record x in
  let _slot, x = Piqirun.parse_optional_field 1 parse_protobuf_int32 x in
  let _tensor_description, x = Piqirun.parse_optional_field 3 parse_tensor_description x in
  Piqirun.check_unparsed_fields x;
  {
    Node_output.slot = _slot;
    Node_output.tensor_description = _tensor_description;
  }

and parse_node_exec_stats x =
  let x = Piqirun.parse_record x in
  let _node_name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _all_start_micros, x = Piqirun.parse_optional_field 2 parse_protobuf_int64 x in
  let _op_start_rel_micros, x = Piqirun.parse_optional_field 3 parse_protobuf_int64 x in
  let _op_end_rel_micros, x = Piqirun.parse_optional_field 4 parse_protobuf_int64 x in
  let _all_end_rel_micros, x = Piqirun.parse_optional_field 5 parse_protobuf_int64 x in
  let _memory, x = Piqirun.parse_repeated_field 6 parse_allocator_memory_used x in
  let _output, x = Piqirun.parse_repeated_field 7 parse_node_output x in
  let _timeline_label, x = Piqirun.parse_optional_field 8 parse_string x in
  let _scheduled_micros, x = Piqirun.parse_optional_field 9 parse_protobuf_int64 x in
  let _thread_id, x = Piqirun.parse_optional_field 10 parse_uint32 x in
  let _referenced_tensor, x = Piqirun.parse_repeated_field 11 parse_allocation_description x in
  Piqirun.check_unparsed_fields x;
  {
    Node_exec_stats.node_name = _node_name;
    Node_exec_stats.all_start_micros = _all_start_micros;
    Node_exec_stats.op_start_rel_micros = _op_start_rel_micros;
    Node_exec_stats.op_end_rel_micros = _op_end_rel_micros;
    Node_exec_stats.all_end_rel_micros = _all_end_rel_micros;
    Node_exec_stats.memory = _memory;
    Node_exec_stats.output = _output;
    Node_exec_stats.timeline_label = _timeline_label;
    Node_exec_stats.scheduled_micros = _scheduled_micros;
    Node_exec_stats.thread_id = _thread_id;
    Node_exec_stats.referenced_tensor = _referenced_tensor;
  }

and parse_device_step_stats x =
  let x = Piqirun.parse_record x in
  let _device, x = Piqirun.parse_optional_field 1 parse_string x in
  let _node_stats, x = Piqirun.parse_repeated_field 2 parse_node_exec_stats x in
  Piqirun.check_unparsed_fields x;
  {
    Device_step_stats.device = _device;
    Device_step_stats.node_stats = _node_stats;
  }

and parse_step_stats x =
  let x = Piqirun.parse_record x in
  let _dev_stats, x = Piqirun.parse_repeated_field 1 parse_device_step_stats x in
  Piqirun.check_unparsed_fields x;
  {
    Step_stats.dev_stats = _dev_stats;
  }

and parse_histogram_proto x =
  let x = Piqirun.parse_record x in
  let _min, x = Piqirun.parse_optional_field 1 parse_float64 x in
  let _max, x = Piqirun.parse_optional_field 2 parse_float64 x in
  let _num, x = Piqirun.parse_optional_field 3 parse_float64 x in
  let _sum, x = Piqirun.parse_optional_field 4 parse_float64 x in
  let _sum_squares, x = Piqirun.parse_optional_field 5 parse_float64 x in
  let _bucket_limit, x = Piqirun.parse_packed_repeated_field 6 packed_parse_float64 parse_float64 x in
  let _bucket, x = Piqirun.parse_packed_repeated_field 7 packed_parse_float64 parse_float64 x in
  Piqirun.check_unparsed_fields x;
  {
    Histogram_proto.min = _min;
    Histogram_proto.max = _max;
    Histogram_proto.num = _num;
    Histogram_proto.sum = _sum;
    Histogram_proto.sum_squares = _sum_squares;
    Histogram_proto.bucket_limit = _bucket_limit;
    Histogram_proto.bucket = _bucket;
  }

and parse_summary x =
  let x = Piqirun.parse_record x in
  let _value, x = Piqirun.parse_repeated_field 1 parse_summary_value x in
  Piqirun.check_unparsed_fields x;
  {
    Summary.value = _value;
  }

and parse_summary_image x =
  let x = Piqirun.parse_record x in
  let _height, x = Piqirun.parse_optional_field 1 parse_protobuf_int32 x in
  let _width, x = Piqirun.parse_optional_field 2 parse_protobuf_int32 x in
  let _colorspace, x = Piqirun.parse_optional_field 3 parse_protobuf_int32 x in
  let _encoded_image_string, x = Piqirun.parse_optional_field 4 parse_binary x in
  Piqirun.check_unparsed_fields x;
  {
    Summary_image.height = _height;
    Summary_image.width = _width;
    Summary_image.colorspace = _colorspace;
    Summary_image.encoded_image_string = _encoded_image_string;
  }

and parse_summary_value x =
  let x = Piqirun.parse_record x in
  let _tag, x = Piqirun.parse_optional_field 1 parse_string x in
  let _simple_value, x = Piqirun.parse_optional_field 2 parse_float32 x in
  let _obsolete_old_style_histogram, x = Piqirun.parse_optional_field 3 parse_binary x in
  let _image, x = Piqirun.parse_optional_field 4 parse_summary_image x in
  let _histo, x = Piqirun.parse_optional_field 5 parse_histogram_proto x in
  Piqirun.check_unparsed_fields x;
  {
    Summary_value.tag = _tag;
    Summary_value.simple_value = _simple_value;
    Summary_value.obsolete_old_style_histogram = _obsolete_old_style_histogram;
    Summary_value.image = _image;
    Summary_value.histo = _histo;
  }

and parse_tensor_description x =
  let x = Piqirun.parse_record x in
  let _dtype, x = Piqirun.parse_optional_field 1 parse_data_type x in
  let _shape, x = Piqirun.parse_optional_field 2 parse_tensor_shape_proto x in
  let _allocation_description, x = Piqirun.parse_optional_field 4 parse_allocation_description x in
  Piqirun.check_unparsed_fields x;
  {
    Tensor_description.dtype = _dtype;
    Tensor_description.shape = _shape;
    Tensor_description.allocation_description = _allocation_description;
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
  }

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

and parse_tensor_slice_proto x =
  let x = Piqirun.parse_record x in
  let _extent, x = Piqirun.parse_repeated_field 1 parse_tensor_slice_proto_extent x in
  Piqirun.check_unparsed_fields x;
  {
    Tensor_slice_proto.extent = _extent;
  }

and parse_tensor_slice_proto_extent x =
  let x = Piqirun.parse_record x in
  let _start, x = Piqirun.parse_optional_field 1 parse_protobuf_int64 x in
  let _length, x = Piqirun.parse_optional_field 2 parse_protobuf_int64 x in
  Piqirun.check_unparsed_fields x;
  {
    Tensor_slice_proto_extent.start = _start;
    Tensor_slice_proto_extent.length = _length;
  }

and parse_variable_def x =
  let x = Piqirun.parse_record x in
  let _variable_name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _initializer_name, x = Piqirun.parse_optional_field 2 parse_string x in
  let _snapshot_name, x = Piqirun.parse_optional_field 3 parse_string x in
  let _save_slice_info_def, x = Piqirun.parse_optional_field 4 parse_save_slice_info_def x in
  Piqirun.check_unparsed_fields x;
  {
    Variable_def.variable_name = _variable_name;
    Variable_def.initializer_name = _initializer_name;
    Variable_def.snapshot_name = _snapshot_name;
    Variable_def.save_slice_info_def = _save_slice_info_def;
  }

and parse_save_slice_info_def x =
  let x = Piqirun.parse_record x in
  let _full_name, x = Piqirun.parse_optional_field 1 parse_string x in
  let _full_shape, x = Piqirun.parse_repeated_field 2 parse_protobuf_int32 x in
  let _var_offset, x = Piqirun.parse_repeated_field 3 parse_protobuf_int32 x in
  let _var_shape, x = Piqirun.parse_repeated_field 4 parse_protobuf_int32 x in
  Piqirun.check_unparsed_fields x;
  {
    Save_slice_info_def.full_name = _full_name;
    Save_slice_info_def.full_shape = _full_shape;
    Save_slice_info_def.var_offset = _var_offset;
    Save_slice_info_def.var_shape = _var_shape;
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

and parse_bus_adjacency x =
  match Piqirun.int32_of_signed_varint x with
    | 0l -> `bus_0
    | 1l -> `bus_1
    | 2l -> `bus_any
    | 3l -> `bus_num_adjacencies
    | x -> Piqirun.error_enum_const x
and packed_parse_bus_adjacency x =
  match Piqirun.int32_of_packed_signed_varint x with
    | 0l -> `bus_0
    | 1l -> `bus_1
    | 2l -> `bus_any
    | 3l -> `bus_num_adjacencies
    | x -> Piqirun.error_enum_const x

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
    | x -> Piqirun.error_enum_const x


let rec gen__int64 code x = Piqirun.int64_to_zigzag_varint code x
and packed_gen__int64 x = Piqirun.int64_to_packed_zigzag_varint x

and gen__uint64 code x = Piqirun.int64_to_varint code x
and packed_gen__uint64 x = Piqirun.int64_to_packed_varint x

and gen__int32 code x = Piqirun.int32_to_zigzag_varint code x
and packed_gen__int32 x = Piqirun.int32_to_packed_zigzag_varint x

and gen__protobuf_int64 code x = Piqirun.int64_to_signed_varint code x
and packed_gen__protobuf_int64 x = Piqirun.int64_to_packed_signed_varint x

and gen__string code x = Piqirun.string_to_block code x

and gen__bool code x = Piqirun.bool_to_varint code x
and packed_gen__bool x = Piqirun.bool_to_packed_varint x

and gen__binary code x = Piqirun.string_to_block code x

and gen__float32 code x = Piqirun.float_to_fixed32 code x
and packed_gen__float32 x = Piqirun.float_to_packed_fixed32 x

and gen__uint64_fixed code x = Piqirun.int64_to_fixed64 code x
and packed_gen__uint64_fixed x = Piqirun.int64_to_packed_fixed64 x

and gen__protobuf_int32 code x = Piqirun.int32_to_signed_varint code x
and packed_gen__protobuf_int32 x = Piqirun.int32_to_packed_signed_varint x

and gen__uint32 code x = Piqirun.int32_to_varint code x
and packed_gen__uint32 x = Piqirun.int32_to_packed_varint x

and gen__float64 code x = Piqirun.float_to_fixed64 code x
and packed_gen__float64 x = Piqirun.float_to_packed_fixed64 x

and gen__allocation_description code x =
  let _requested_bytes = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Allocation_description.requested_bytes in
  let _allocated_bytes = Piqirun.gen_optional_field 2 gen__protobuf_int64 x.Allocation_description.allocated_bytes in
  let _allocator_name = Piqirun.gen_optional_field 3 gen__string x.Allocation_description.allocator_name in
  let _allocation_id = Piqirun.gen_optional_field 4 gen__protobuf_int64 x.Allocation_description.allocation_id in
  let _has_single_reference = Piqirun.gen_optional_field 5 gen__bool x.Allocation_description.has_single_reference in
  let _ptr = Piqirun.gen_optional_field 6 gen__uint64 x.Allocation_description.ptr in
  Piqirun.gen_record code (_requested_bytes :: _allocated_bytes :: _allocator_name :: _allocation_id :: _has_single_reference :: _ptr :: [])

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

and gen__device_attributes code x =
  let _name = Piqirun.gen_optional_field 1 gen__string x.Device_attributes.name in
  let _device_type = Piqirun.gen_optional_field 2 gen__string x.Device_attributes.device_type in
  let _memory_limit = Piqirun.gen_optional_field 4 gen__protobuf_int64 x.Device_attributes.memory_limit in
  let _bus_adjacency = Piqirun.gen_optional_field 5 gen__bus_adjacency x.Device_attributes.bus_adjacency in
  let _incarnation = Piqirun.gen_optional_field 6 gen__uint64_fixed x.Device_attributes.incarnation in
  let _physical_device_desc = Piqirun.gen_optional_field 7 gen__string x.Device_attributes.physical_device_desc in
  Piqirun.gen_record code (_name :: _device_type :: _memory_limit :: _bus_adjacency :: _incarnation :: _physical_device_desc :: [])

and gen__function_def_library code x =
  let _function_ = Piqirun.gen_repeated_field 1 gen__function_def x.Function_def_library.function_ in
  Piqirun.gen_record code (_function_ :: [])

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

and gen__kernel_def code x =
  let _op = Piqirun.gen_optional_field 1 gen__string x.Kernel_def.op in
  let _device_type = Piqirun.gen_optional_field 2 gen__string x.Kernel_def.device_type in
  let _constraint_ = Piqirun.gen_repeated_field 3 gen__kernel_def_attr_constraint x.Kernel_def.constraint_ in
  let _host_memory_arg = Piqirun.gen_repeated_field 4 gen__string x.Kernel_def.host_memory_arg in
  let _label = Piqirun.gen_optional_field 5 gen__string x.Kernel_def.label in
  Piqirun.gen_record code (_op :: _device_type :: _constraint_ :: _host_memory_arg :: _label :: [])

and gen__kernel_def_attr_constraint code x =
  let _name = Piqirun.gen_optional_field 1 gen__string x.Kernel_def_attr_constraint.name in
  let _allowed_values = Piqirun.gen_optional_field 2 gen__attr_value x.Kernel_def_attr_constraint.allowed_values in
  Piqirun.gen_record code (_name :: _allowed_values :: [])

and gen__memory_log_step code x =
  let _step_id = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Memory_log_step.step_id in
  let _handle = Piqirun.gen_optional_field 2 gen__string x.Memory_log_step.handle in
  Piqirun.gen_record code (_step_id :: _handle :: [])

and gen__memory_log_tensor_allocation code x =
  let _step_id = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Memory_log_tensor_allocation.step_id in
  let _kernel_name = Piqirun.gen_optional_field 2 gen__string x.Memory_log_tensor_allocation.kernel_name in
  let _tensor = Piqirun.gen_optional_field 3 gen__tensor_description x.Memory_log_tensor_allocation.tensor in
  Piqirun.gen_record code (_step_id :: _kernel_name :: _tensor :: [])

and gen__memory_log_tensor_deallocation code x =
  let _allocation_id = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Memory_log_tensor_deallocation.allocation_id in
  let _allocator_name = Piqirun.gen_optional_field 2 gen__string x.Memory_log_tensor_deallocation.allocator_name in
  Piqirun.gen_record code (_allocation_id :: _allocator_name :: [])

and gen__memory_log_tensor_output code x =
  let _step_id = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Memory_log_tensor_output.step_id in
  let _kernel_name = Piqirun.gen_optional_field 2 gen__string x.Memory_log_tensor_output.kernel_name in
  let _index = Piqirun.gen_optional_field 3 gen__protobuf_int32 x.Memory_log_tensor_output.index in
  let _tensor = Piqirun.gen_optional_field 4 gen__tensor_description x.Memory_log_tensor_output.tensor in
  Piqirun.gen_record code (_step_id :: _kernel_name :: _index :: _tensor :: [])

and gen__memory_log_raw_allocation code x =
  let _step_id = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Memory_log_raw_allocation.step_id in
  let _operation = Piqirun.gen_optional_field 2 gen__string x.Memory_log_raw_allocation.operation in
  let _num_bytes = Piqirun.gen_optional_field 3 gen__protobuf_int64 x.Memory_log_raw_allocation.num_bytes in
  let _ptr = Piqirun.gen_optional_field 4 gen__uint64 x.Memory_log_raw_allocation.ptr in
  let _allocation_id = Piqirun.gen_optional_field 5 gen__protobuf_int64 x.Memory_log_raw_allocation.allocation_id in
  let _allocator_name = Piqirun.gen_optional_field 6 gen__string x.Memory_log_raw_allocation.allocator_name in
  Piqirun.gen_record code (_step_id :: _operation :: _num_bytes :: _ptr :: _allocation_id :: _allocator_name :: [])

and gen__memory_log_raw_deallocation code x =
  let _step_id = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Memory_log_raw_deallocation.step_id in
  let _operation = Piqirun.gen_optional_field 2 gen__string x.Memory_log_raw_deallocation.operation in
  let _allocation_id = Piqirun.gen_optional_field 3 gen__protobuf_int64 x.Memory_log_raw_deallocation.allocation_id in
  let _allocator_name = Piqirun.gen_optional_field 4 gen__string x.Memory_log_raw_deallocation.allocator_name in
  let _deferred = Piqirun.gen_optional_field 5 gen__bool x.Memory_log_raw_deallocation.deferred in
  Piqirun.gen_record code (_step_id :: _operation :: _allocation_id :: _allocator_name :: _deferred :: [])

and gen__op_def code x =
  let _name = Piqirun.gen_optional_field 1 gen__string x.Op_def.name in
  let _input_arg = Piqirun.gen_repeated_field 2 gen__op_def_arg_def x.Op_def.input_arg in
  let _output_arg = Piqirun.gen_repeated_field 3 gen__op_def_arg_def x.Op_def.output_arg in
  let _attr = Piqirun.gen_repeated_field 4 gen__op_def_attr_def x.Op_def.attr in
  let _summary = Piqirun.gen_optional_field 5 gen__string x.Op_def.summary in
  let _description = Piqirun.gen_optional_field 6 gen__string x.Op_def.description in
  let _is_aggregate = Piqirun.gen_optional_field 16 gen__bool x.Op_def.is_aggregate in
  let _is_stateful = Piqirun.gen_optional_field 17 gen__bool x.Op_def.is_stateful in
  let _is_commutative = Piqirun.gen_optional_field 18 gen__bool x.Op_def.is_commutative in
  let _allows_uninitialized_input = Piqirun.gen_optional_field 19 gen__bool x.Op_def.allows_uninitialized_input in
  Piqirun.gen_record code (_name :: _input_arg :: _output_arg :: _attr :: _summary :: _description :: _is_aggregate :: _is_stateful :: _is_commutative :: _allows_uninitialized_input :: [])

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

and gen__op_list code x =
  let _op = Piqirun.gen_repeated_field 1 gen__op_def x.Op_list.op in
  Piqirun.gen_record code (_op :: [])

and gen__allocator_memory_used code x =
  let _allocator_name = Piqirun.gen_optional_field 1 gen__string x.Allocator_memory_used.allocator_name in
  let _total_bytes = Piqirun.gen_optional_field 2 gen__protobuf_int64 x.Allocator_memory_used.total_bytes in
  let _peak_bytes = Piqirun.gen_optional_field 3 gen__protobuf_int64 x.Allocator_memory_used.peak_bytes in
  Piqirun.gen_record code (_allocator_name :: _total_bytes :: _peak_bytes :: [])

and gen__node_output code x =
  let _slot = Piqirun.gen_optional_field 1 gen__protobuf_int32 x.Node_output.slot in
  let _tensor_description = Piqirun.gen_optional_field 3 gen__tensor_description x.Node_output.tensor_description in
  Piqirun.gen_record code (_slot :: _tensor_description :: [])

and gen__node_exec_stats code x =
  let _node_name = Piqirun.gen_optional_field 1 gen__string x.Node_exec_stats.node_name in
  let _all_start_micros = Piqirun.gen_optional_field 2 gen__protobuf_int64 x.Node_exec_stats.all_start_micros in
  let _op_start_rel_micros = Piqirun.gen_optional_field 3 gen__protobuf_int64 x.Node_exec_stats.op_start_rel_micros in
  let _op_end_rel_micros = Piqirun.gen_optional_field 4 gen__protobuf_int64 x.Node_exec_stats.op_end_rel_micros in
  let _all_end_rel_micros = Piqirun.gen_optional_field 5 gen__protobuf_int64 x.Node_exec_stats.all_end_rel_micros in
  let _memory = Piqirun.gen_repeated_field 6 gen__allocator_memory_used x.Node_exec_stats.memory in
  let _output = Piqirun.gen_repeated_field 7 gen__node_output x.Node_exec_stats.output in
  let _timeline_label = Piqirun.gen_optional_field 8 gen__string x.Node_exec_stats.timeline_label in
  let _scheduled_micros = Piqirun.gen_optional_field 9 gen__protobuf_int64 x.Node_exec_stats.scheduled_micros in
  let _thread_id = Piqirun.gen_optional_field 10 gen__uint32 x.Node_exec_stats.thread_id in
  let _referenced_tensor = Piqirun.gen_repeated_field 11 gen__allocation_description x.Node_exec_stats.referenced_tensor in
  Piqirun.gen_record code (_node_name :: _all_start_micros :: _op_start_rel_micros :: _op_end_rel_micros :: _all_end_rel_micros :: _memory :: _output :: _timeline_label :: _scheduled_micros :: _thread_id :: _referenced_tensor :: [])

and gen__device_step_stats code x =
  let _device = Piqirun.gen_optional_field 1 gen__string x.Device_step_stats.device in
  let _node_stats = Piqirun.gen_repeated_field 2 gen__node_exec_stats x.Device_step_stats.node_stats in
  Piqirun.gen_record code (_device :: _node_stats :: [])

and gen__step_stats code x =
  let _dev_stats = Piqirun.gen_repeated_field 1 gen__device_step_stats x.Step_stats.dev_stats in
  Piqirun.gen_record code (_dev_stats :: [])

and gen__histogram_proto code x =
  let _min = Piqirun.gen_optional_field 1 gen__float64 x.Histogram_proto.min in
  let _max = Piqirun.gen_optional_field 2 gen__float64 x.Histogram_proto.max in
  let _num = Piqirun.gen_optional_field 3 gen__float64 x.Histogram_proto.num in
  let _sum = Piqirun.gen_optional_field 4 gen__float64 x.Histogram_proto.sum in
  let _sum_squares = Piqirun.gen_optional_field 5 gen__float64 x.Histogram_proto.sum_squares in
  let _bucket_limit = Piqirun.gen_packed_repeated_field 6 packed_gen__float64 x.Histogram_proto.bucket_limit in
  let _bucket = Piqirun.gen_packed_repeated_field 7 packed_gen__float64 x.Histogram_proto.bucket in
  Piqirun.gen_record code (_min :: _max :: _num :: _sum :: _sum_squares :: _bucket_limit :: _bucket :: [])

and gen__summary code x =
  let _value = Piqirun.gen_repeated_field 1 gen__summary_value x.Summary.value in
  Piqirun.gen_record code (_value :: [])

and gen__summary_image code x =
  let _height = Piqirun.gen_optional_field 1 gen__protobuf_int32 x.Summary_image.height in
  let _width = Piqirun.gen_optional_field 2 gen__protobuf_int32 x.Summary_image.width in
  let _colorspace = Piqirun.gen_optional_field 3 gen__protobuf_int32 x.Summary_image.colorspace in
  let _encoded_image_string = Piqirun.gen_optional_field 4 gen__binary x.Summary_image.encoded_image_string in
  Piqirun.gen_record code (_height :: _width :: _colorspace :: _encoded_image_string :: [])

and gen__summary_value code x =
  let _tag = Piqirun.gen_optional_field 1 gen__string x.Summary_value.tag in
  let _simple_value = Piqirun.gen_optional_field 2 gen__float32 x.Summary_value.simple_value in
  let _obsolete_old_style_histogram = Piqirun.gen_optional_field 3 gen__binary x.Summary_value.obsolete_old_style_histogram in
  let _image = Piqirun.gen_optional_field 4 gen__summary_image x.Summary_value.image in
  let _histo = Piqirun.gen_optional_field 5 gen__histogram_proto x.Summary_value.histo in
  Piqirun.gen_record code (_tag :: _simple_value :: _obsolete_old_style_histogram :: _image :: _histo :: [])

and gen__tensor_description code x =
  let _dtype = Piqirun.gen_optional_field 1 gen__data_type x.Tensor_description.dtype in
  let _shape = Piqirun.gen_optional_field 2 gen__tensor_shape_proto x.Tensor_description.shape in
  let _allocation_description = Piqirun.gen_optional_field 4 gen__allocation_description x.Tensor_description.allocation_description in
  Piqirun.gen_record code (_dtype :: _shape :: _allocation_description :: [])

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
  Piqirun.gen_record code (_dtype :: _tensor_shape :: _version_number :: _tensor_content :: _float_val :: _double_val :: _int_val :: _string_val :: _scomplex_val :: _int64_val :: _bool_val :: _dcomplex_val :: [])

and gen__tensor_shape_proto code x =
  let _dim = Piqirun.gen_repeated_field 2 gen__tensor_shape_proto_dim x.Tensor_shape_proto.dim in
  let _unknown_rank = Piqirun.gen_optional_field 3 gen__bool x.Tensor_shape_proto.unknown_rank in
  Piqirun.gen_record code (_dim :: _unknown_rank :: [])

and gen__tensor_shape_proto_dim code x =
  let _size = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Tensor_shape_proto_dim.size in
  let _name = Piqirun.gen_optional_field 2 gen__string x.Tensor_shape_proto_dim.name in
  Piqirun.gen_record code (_size :: _name :: [])

and gen__tensor_slice_proto code x =
  let _extent = Piqirun.gen_repeated_field 1 gen__tensor_slice_proto_extent x.Tensor_slice_proto.extent in
  Piqirun.gen_record code (_extent :: [])

and gen__tensor_slice_proto_extent code x =
  let _start = Piqirun.gen_optional_field 1 gen__protobuf_int64 x.Tensor_slice_proto_extent.start in
  let _length = Piqirun.gen_optional_field 2 gen__protobuf_int64 x.Tensor_slice_proto_extent.length in
  Piqirun.gen_record code (_start :: _length :: [])

and gen__variable_def code x =
  let _variable_name = Piqirun.gen_optional_field 1 gen__string x.Variable_def.variable_name in
  let _initializer_name = Piqirun.gen_optional_field 2 gen__string x.Variable_def.initializer_name in
  let _snapshot_name = Piqirun.gen_optional_field 3 gen__string x.Variable_def.snapshot_name in
  let _save_slice_info_def = Piqirun.gen_optional_field 4 gen__save_slice_info_def x.Variable_def.save_slice_info_def in
  Piqirun.gen_record code (_variable_name :: _initializer_name :: _snapshot_name :: _save_slice_info_def :: [])

and gen__save_slice_info_def code x =
  let _full_name = Piqirun.gen_optional_field 1 gen__string x.Save_slice_info_def.full_name in
  let _full_shape = Piqirun.gen_repeated_field 2 gen__protobuf_int32 x.Save_slice_info_def.full_shape in
  let _var_offset = Piqirun.gen_repeated_field 3 gen__protobuf_int32 x.Save_slice_info_def.var_offset in
  let _var_shape = Piqirun.gen_repeated_field 4 gen__protobuf_int32 x.Save_slice_info_def.var_shape in
  Piqirun.gen_record code (_full_name :: _full_shape :: _var_offset :: _var_shape :: [])

and gen__version_def code x =
  let _producer = Piqirun.gen_optional_field 1 gen__protobuf_int32 x.Version_def.producer in
  let _min_consumer = Piqirun.gen_optional_field 2 gen__protobuf_int32 x.Version_def.min_consumer in
  let _bad_consumers = Piqirun.gen_repeated_field 3 gen__protobuf_int32 x.Version_def.bad_consumers in
  Piqirun.gen_record code (_producer :: _min_consumer :: _bad_consumers :: [])

and gen__bus_adjacency code x =
  Piqirun.int32_to_signed_varint code (match x with
    | `bus_0 -> 0l
    | `bus_1 -> 1l
    | `bus_any -> 2l
    | `bus_num_adjacencies -> 3l
  )
and packed_gen__bus_adjacency x =
  Piqirun.int32_to_packed_signed_varint (match x with
    | `bus_0 -> 0l
    | `bus_1 -> 1l
    | `bus_any -> 2l
    | `bus_num_adjacencies -> 3l
  )

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
  )


let gen_int64 x = gen__int64 (-1) x
let gen_uint64 x = gen__uint64 (-1) x
let gen_int32 x = gen__int32 (-1) x
let gen_protobuf_int64 x = gen__protobuf_int64 (-1) x
let gen_string x = gen__string (-1) x
let gen_bool x = gen__bool (-1) x
let gen_binary x = gen__binary (-1) x
let gen_float32 x = gen__float32 (-1) x
let gen_uint64_fixed x = gen__uint64_fixed (-1) x
let gen_protobuf_int32 x = gen__protobuf_int32 (-1) x
let gen_uint32 x = gen__uint32 (-1) x
let gen_float64 x = gen__float64 (-1) x
let gen_allocation_description x = gen__allocation_description (-1) x
let gen_attr_value x = gen__attr_value (-1) x
let gen_attr_value_list_value x = gen__attr_value_list_value (-1) x
let gen_name_attr_list x = gen__name_attr_list (-1) x
let gen_name_attr_list_attr_entry x = gen__name_attr_list_attr_entry (-1) x
let gen_device_attributes x = gen__device_attributes (-1) x
let gen_function_def_library x = gen__function_def_library (-1) x
let gen_function_def x = gen__function_def (-1) x
let gen_function_def_node x = gen__function_def_node (-1) x
let gen_function_def_node_attr_entry x = gen__function_def_node_attr_entry (-1) x
let gen_graph_def x = gen__graph_def (-1) x
let gen_node_def x = gen__node_def (-1) x
let gen_node_def_attr_entry x = gen__node_def_attr_entry (-1) x
let gen_kernel_def x = gen__kernel_def (-1) x
let gen_kernel_def_attr_constraint x = gen__kernel_def_attr_constraint (-1) x
let gen_memory_log_step x = gen__memory_log_step (-1) x
let gen_memory_log_tensor_allocation x = gen__memory_log_tensor_allocation (-1) x
let gen_memory_log_tensor_deallocation x = gen__memory_log_tensor_deallocation (-1) x
let gen_memory_log_tensor_output x = gen__memory_log_tensor_output (-1) x
let gen_memory_log_raw_allocation x = gen__memory_log_raw_allocation (-1) x
let gen_memory_log_raw_deallocation x = gen__memory_log_raw_deallocation (-1) x
let gen_op_def x = gen__op_def (-1) x
let gen_op_def_arg_def x = gen__op_def_arg_def (-1) x
let gen_op_def_attr_def x = gen__op_def_attr_def (-1) x
let gen_op_list x = gen__op_list (-1) x
let gen_allocator_memory_used x = gen__allocator_memory_used (-1) x
let gen_node_output x = gen__node_output (-1) x
let gen_node_exec_stats x = gen__node_exec_stats (-1) x
let gen_device_step_stats x = gen__device_step_stats (-1) x
let gen_step_stats x = gen__step_stats (-1) x
let gen_histogram_proto x = gen__histogram_proto (-1) x
let gen_summary x = gen__summary (-1) x
let gen_summary_image x = gen__summary_image (-1) x
let gen_summary_value x = gen__summary_value (-1) x
let gen_tensor_description x = gen__tensor_description (-1) x
let gen_tensor_proto x = gen__tensor_proto (-1) x
let gen_tensor_shape_proto x = gen__tensor_shape_proto (-1) x
let gen_tensor_shape_proto_dim x = gen__tensor_shape_proto_dim (-1) x
let gen_tensor_slice_proto x = gen__tensor_slice_proto (-1) x
let gen_tensor_slice_proto_extent x = gen__tensor_slice_proto_extent (-1) x
let gen_variable_def x = gen__variable_def (-1) x
let gen_save_slice_info_def x = gen__save_slice_info_def (-1) x
let gen_version_def x = gen__version_def (-1) x
let gen_bus_adjacency x = gen__bus_adjacency (-1) x
let gen_data_type x = gen__data_type (-1) x


let rec default_int64 () = 0L
and default_uint64 () = 0L
and default_int32 () = 0l
and default_protobuf_int64 () = default_int64 ()
and default_string () = ""
and default_bool () = false
and default_binary () = ""
and default_float32 () = 0.0
and default_uint64_fixed () = default_uint64 ()
and default_protobuf_int32 () = default_int32 ()
and default_uint32 () = 0l
and default_float64 () = 0.0
and default_allocation_description () =
  {
    Allocation_description.requested_bytes = None;
    Allocation_description.allocated_bytes = None;
    Allocation_description.allocator_name = None;
    Allocation_description.allocation_id = None;
    Allocation_description.has_single_reference = None;
    Allocation_description.ptr = None;
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
and default_device_attributes () =
  {
    Device_attributes.name = None;
    Device_attributes.device_type = None;
    Device_attributes.memory_limit = None;
    Device_attributes.bus_adjacency = None;
    Device_attributes.incarnation = None;
    Device_attributes.physical_device_desc = None;
  }
and default_function_def_library () =
  {
    Function_def_library.function_ = [];
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
and default_kernel_def () =
  {
    Kernel_def.op = None;
    Kernel_def.device_type = None;
    Kernel_def.constraint_ = [];
    Kernel_def.host_memory_arg = [];
    Kernel_def.label = None;
  }
and default_kernel_def_attr_constraint () =
  {
    Kernel_def_attr_constraint.name = None;
    Kernel_def_attr_constraint.allowed_values = None;
  }
and default_memory_log_step () =
  {
    Memory_log_step.step_id = None;
    Memory_log_step.handle = None;
  }
and default_memory_log_tensor_allocation () =
  {
    Memory_log_tensor_allocation.step_id = None;
    Memory_log_tensor_allocation.kernel_name = None;
    Memory_log_tensor_allocation.tensor = None;
  }
and default_memory_log_tensor_deallocation () =
  {
    Memory_log_tensor_deallocation.allocation_id = None;
    Memory_log_tensor_deallocation.allocator_name = None;
  }
and default_memory_log_tensor_output () =
  {
    Memory_log_tensor_output.step_id = None;
    Memory_log_tensor_output.kernel_name = None;
    Memory_log_tensor_output.index = None;
    Memory_log_tensor_output.tensor = None;
  }
and default_memory_log_raw_allocation () =
  {
    Memory_log_raw_allocation.step_id = None;
    Memory_log_raw_allocation.operation = None;
    Memory_log_raw_allocation.num_bytes = None;
    Memory_log_raw_allocation.ptr = None;
    Memory_log_raw_allocation.allocation_id = None;
    Memory_log_raw_allocation.allocator_name = None;
  }
and default_memory_log_raw_deallocation () =
  {
    Memory_log_raw_deallocation.step_id = None;
    Memory_log_raw_deallocation.operation = None;
    Memory_log_raw_deallocation.allocation_id = None;
    Memory_log_raw_deallocation.allocator_name = None;
    Memory_log_raw_deallocation.deferred = None;
  }
and default_op_def () =
  {
    Op_def.name = None;
    Op_def.input_arg = [];
    Op_def.output_arg = [];
    Op_def.attr = [];
    Op_def.summary = None;
    Op_def.description = None;
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
and default_op_list () =
  {
    Op_list.op = [];
  }
and default_allocator_memory_used () =
  {
    Allocator_memory_used.allocator_name = None;
    Allocator_memory_used.total_bytes = None;
    Allocator_memory_used.peak_bytes = None;
  }
and default_node_output () =
  {
    Node_output.slot = None;
    Node_output.tensor_description = None;
  }
and default_node_exec_stats () =
  {
    Node_exec_stats.node_name = None;
    Node_exec_stats.all_start_micros = None;
    Node_exec_stats.op_start_rel_micros = None;
    Node_exec_stats.op_end_rel_micros = None;
    Node_exec_stats.all_end_rel_micros = None;
    Node_exec_stats.memory = [];
    Node_exec_stats.output = [];
    Node_exec_stats.timeline_label = None;
    Node_exec_stats.scheduled_micros = None;
    Node_exec_stats.thread_id = None;
    Node_exec_stats.referenced_tensor = [];
  }
and default_device_step_stats () =
  {
    Device_step_stats.device = None;
    Device_step_stats.node_stats = [];
  }
and default_step_stats () =
  {
    Step_stats.dev_stats = [];
  }
and default_histogram_proto () =
  {
    Histogram_proto.min = None;
    Histogram_proto.max = None;
    Histogram_proto.num = None;
    Histogram_proto.sum = None;
    Histogram_proto.sum_squares = None;
    Histogram_proto.bucket_limit = [];
    Histogram_proto.bucket = [];
  }
and default_summary () =
  {
    Summary.value = [];
  }
and default_summary_image () =
  {
    Summary_image.height = None;
    Summary_image.width = None;
    Summary_image.colorspace = None;
    Summary_image.encoded_image_string = None;
  }
and default_summary_value () =
  {
    Summary_value.tag = None;
    Summary_value.simple_value = None;
    Summary_value.obsolete_old_style_histogram = None;
    Summary_value.image = None;
    Summary_value.histo = None;
  }
and default_tensor_description () =
  {
    Tensor_description.dtype = None;
    Tensor_description.shape = None;
    Tensor_description.allocation_description = None;
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
  }
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
and default_tensor_slice_proto () =
  {
    Tensor_slice_proto.extent = [];
  }
and default_tensor_slice_proto_extent () =
  {
    Tensor_slice_proto_extent.start = None;
    Tensor_slice_proto_extent.length = None;
  }
and default_variable_def () =
  {
    Variable_def.variable_name = None;
    Variable_def.initializer_name = None;
    Variable_def.snapshot_name = None;
    Variable_def.save_slice_info_def = None;
  }
and default_save_slice_info_def () =
  {
    Save_slice_info_def.full_name = None;
    Save_slice_info_def.full_shape = [];
    Save_slice_info_def.var_offset = [];
    Save_slice_info_def.var_shape = [];
  }
and default_version_def () =
  {
    Version_def.producer = None;
    Version_def.min_consumer = None;
    Version_def.bad_consumers = [];
  }
and default_bus_adjacency () = `bus_0
and default_data_type () = `dt_invalid


include All_piqi
