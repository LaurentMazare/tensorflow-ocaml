type t = string

let of_string x = x
let to_string x = x

let read_file filename =
  let input_channel = open_in filename in
  let size = in_channel_length input_channel in
  let content = Bytes.create size in
  really_input input_channel content 0 size;
  close_in input_channel;
  content
