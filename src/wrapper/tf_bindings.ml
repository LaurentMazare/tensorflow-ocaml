open Ctypes

module C(F: Cstubs.FOREIGN) = struct
  type t = unit ptr
  let t : t typ = ptr void

  let tf_newtensor =
    F.(foreign "TF_NewTensor"
      (int            (* data type *)
      @-> ptr int64_t (* dims *)
      @-> int         (* num dims *)
      @-> ptr void    (* data *)
      @-> size_t      (* len *)
      @-> Foreign.funptr Ctypes.(ptr void @-> int @-> ptr void @-> returning void) (* deallocator *)
      @-> ptr void    (* deallocator arg *)
      @-> returning t))
end

