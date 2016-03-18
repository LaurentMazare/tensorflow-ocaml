open Node

let const_float_1d
    ?(name = "Const")
    ~type_
    values
  =
  (* TODO: check shape, check type_ *)
  { name = Name.make_fresh ~name
  ; op_name = "Const"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes = [
      "dtype", Type (P type_);
      "value", Tensor_float { type_ = P type_; shape = [ List.length values ]; values };
    ]
  ; output_name = None
  }


