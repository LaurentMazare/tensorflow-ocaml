open Node

let const_float
    ?(name = "Const")
    ?shape
    ~type_
    values
  =
  (* TODO: check shape, check type_ *)
  let shape =
    match shape with
    | Some shape -> shape
    | None -> [ List.length values ]
  in
  { name = Name.make_fresh ~name
  ; op_name = "Const"
  ; output_type = type_
  ; inputs = [  ]
  ; attributes = [
      "dtype", Type (P type_);
      "value", Tensor_float { type_ = P type_; shape; values };
    ]
  ; output_name = None
  }
