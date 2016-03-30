type t

val create : unit -> t

module Input :
sig
  type t
  val float  : [`float] Node.t -> (float, Bigarray.float32_elt) Tensor.t -> t
  val double : [`double] Node.t -> (float, Bigarray.float64_elt) Tensor.t -> t
end

module Target :
sig
  type t = Node.p
end

val target : 'a Node.t -> Target.t

module Output :
sig
  type 'a t
  val return : 'a -> 'a t
  val map : 'a t -> f:('a -> 'b) -> 'b t
  val both : 'a t -> 'b t -> ('a * 'b) t
  val three : 'a t -> 'b t -> 'c t -> ('a * 'b * 'c) t
  val four : 'a t -> 'b t -> 'c t -> 'd t -> ('a * 'b * 'c *'d) t
  val five : 'a t -> 'b t -> 'c t -> 'd t -> 'e t -> ('a * 'b * 'c * 'd * 'e) t
  val empty : unit t

  val float  : [`float] Node.t -> (float, Bigarray.float32_elt) Tensor.t t
  val double : [`double] Node.t -> (float, Bigarray.float64_elt) Tensor.t t

  (* Useful for loss *)
  val scalar_float  : [`float] Node.t -> float t
  val scalar_double : [`double] Node.t -> float t
end

val run :
  ?inputs:Input.t list
  -> ?targets:Target.t list
  -> ?session:t
  -> 'a Output.t
  -> 'a
