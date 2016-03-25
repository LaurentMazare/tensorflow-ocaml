open Core_kernel.Std
(* An higher level view of a session *)

(* noury: change that to a real Id in the node *)
module Id = Node.Name

(* the structure of variable to initialize might be slightly tricky:
   some variables migh depend on other to be already initialised in order to be
   initialised.
   I think it can even be a nice way to make sure the size of multiple layers matches.
   (use the parameter of one layer to initialise the parameter of the next) *)
module Variable_initialisation =
struct
  (* Initialisation will run each list in order *)
  type t = Node.p list list
end
(* There is no uninitialised variables because I think we can initialize
them last minute when someone call run, as there is no extend graph *)
type t =
{ session : Wrapper.Session.t;
  (* The nodes already in the graph of the session,
     with their name there *)
  exported_nodes : string Id.Table.t;
  (* The names already present on the server, with the number of times
     it has been used *)
  names : int String.Table.t;
}
