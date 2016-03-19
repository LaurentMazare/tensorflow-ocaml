val const_float
  :  ?name:string
  -> ?shape:int list
  -> type_:([< `float | `double ] as 'dtype) Node.Type.t
  -> float list
  -> 'dtype Node.t
