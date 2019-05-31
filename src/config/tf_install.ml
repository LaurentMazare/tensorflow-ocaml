open Base
module Filename = Caml.Filename
module Sys = Caml.Sys

let tf_lib_url =
  "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.10.0.tar.gz"

let lib_basename = "libtensorflow.so"

let system cmd =
  match Unix.system cmd with
  | WEXITED 0 -> true
  | _ -> false

let system_exn cmd =
  Stdio.printf "Running %s\n%!" cmd;
  if not (system cmd) then Printf.failwithf "error while running %s" cmd ()

let env_exists name =
  match Sys.getenv_opt name with
  | None | Some "" -> false
  | Some _ -> true

let tf_is_available ~lib_filename =
  Sys.file_exists lib_filename
  || env_exists "LIBTENSORFLOW"
  || system "ldconfig -p | grep libtensorflow"

let maybe_ask_user () =
  if Unix.isatty Unix.stdout
  then (
    Stdio.printf
      "Cannot find %s, should I download it from\n  %s\nand install it? [y/n] %!"
      lib_basename
      tf_lib_url;
    match Stdlib.read_line () with
    | "y" -> true
    | _ ->
      Stdio.printf "Cancelling install.\n%!";
      false)
  else (
    Stdio.printf "Installing %s\n%!" lib_basename;
    true)

let create_missing_directory dirname =
  if not (Sys.file_exists dirname)
  then (
    Stdio.printf "Creating missing directory %s\n%!" dirname;
    Unix.mkdir dirname 0o777)

let tf_install ~lib_filename =
  Stdio.printf "Downloading and installing TensorFlow to %s.\n%!" lib_filename;
  let dirname = Filename.dirname lib_filename in
  create_missing_directory dirname;
  let tmp_file = Filename.temp_file lib_basename ".tgz" in
  let tmp_dir = Filename.concat (Filename.get_temp_dir_name ()) "tf-install" in
  create_missing_directory tmp_dir;
  let () = Printf.sprintf "wget %s -O %s" tf_lib_url tmp_file |> system_exn in
  let () = Printf.sprintf "tar xzf %s -C %s" tmp_file tmp_dir |> system_exn in
  let () = Printf.sprintf "cp %s/lib/lib*.so %s" tmp_dir dirname |> system_exn in
  Unix.unlink tmp_file;
  Printf.sprintf "rm -Rf %s" tmp_dir |> system_exn

let () =
  if Array.length Caml.Sys.argv <> 2
  then Printf.failwithf "usage: %s lib_dir" Caml.Sys.argv.(0) ();
  let lib_filename =
    Filename.concat (Filename.concat Caml.Sys.argv.(1) "tensorflow") lib_basename
  in
  if (not (tf_is_available ~lib_filename)) && maybe_ask_user ()
  then tf_install ~lib_filename
