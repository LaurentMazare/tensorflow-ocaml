open Node

let abs ?(name="Abs") x =
  { name = Name.make_fresh ~name
  ; op_name = "Abs"
  ; output_type = x.output_type
  ; inputs = [ P x ]
  ; attributes = [ "T", Type (P x.output_type) ]
  }

