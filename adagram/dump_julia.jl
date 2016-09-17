using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
  "model"
    help = "Path to AdaGram model file"
    arg_type = String
    required = true
  "out"
    help = "Path to output directory"
    arg_type = String
    required = true
end

args = parse_args(ARGS, s)

if !isdir(args["out"])
  mkdir(args["out"])
end

import JSON
import AdaGram

println("loading model...")
vm, dict = AdaGram.load_model(args["model"])
println("done")

println("saving dict.id2word...")
JSON.print(open(joinpath(args["out"], "id2word.json"), "w"), dict.id2word)
println("done")

println("saving vm...")
JSON.print(open(joinpath(args["out"], "vm.json"), "w"), vm)
println("done")
