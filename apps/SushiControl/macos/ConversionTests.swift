import Foundation
@main struct ConversionTests {
 @MainActor static func main() async throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at:root) }
  let source = root.appendingPathComponent("source with spaces"), output = root.appendingPathComponent("output")
  try FileManager.default.createDirectory(at:source,withIntermediateDirectories:true)
  let sourceConfig = Data(#"{"model_type":"unsupported_fixture","torch_dtype":"bfloat16"}"#.utf8)
  try sourceConfig.write(to:source.appendingPathComponent("config.json"))
  var plan = ConversionPlan(source:source.path,output:output.path)
  try Data(#"{"model_type":"qwen4_exp","expert_quant":{"format":"exl3"}}"#.utf8).write(to:source.appendingPathComponent("config.json"))
  do { _ = try plan.arguments(); fatalError("Accepted already-converted Sushi pack") }
  catch let error as ServerFailure { precondition(error.message.contains("already")) }
  try sourceConfig.write(to:source.appendingPathComponent("config.json"))
  let metadata = try ConversionSourceInfo.read(source.path)
  precondition(metadata.architecture == "unsupported_fixture" && metadata.precision == "bfloat16" && !metadata.isSushiPack)
  try Data("[1,2]".utf8).write(to:source.appendingPathComponent("config.json"))
  do { _ = try plan.arguments(); fatalError("Accepted malformed source metadata") } catch {}
  try sourceConfig.write(to:source.appendingPathComponent("config.json"))
  try Data(#"{"model_type":"qwen4_exp","expert_quant":{"format":"stq"}}"#.utf8).write(to:source.appendingPathComponent("config.json"))
  do { _ = try plan.arguments(); fatalError("Accepted STQ pack as original checkpoint") }
  catch let error as ServerFailure { precondition(error.message.contains("already")) }
  try sourceConfig.write(to:source.appendingPathComponent("config.json"))
  precondition(ConverterSyntax(help:"convert exl3 --expert-codebook") == .expert)
  precondition(ConverterSyntax(help:"convert exl3 --codebook") == .legacy)
  precondition(ConverterSyntax(help:"convert exl3") == nil)
  let modern = try plan.arguments(syntax:.expert)
  precondition(modern.contains("--expert-codebook") && modern.contains("exl3-mcg") && !modern.contains("--codebook"))
  plan.codebook = "mul1"
  let mul1 = try plan.arguments(syntax:.expert); precondition(mul1.contains("exl3-mul1"))
  plan.codebook = "mcg"
  let args = try plan.arguments()
  precondition(args.first == "convert" && args[3].contains("source with spaces") && !args.contains("--resume"))
  plan.output = source.appendingPathComponent("nested").path
  do { _ = try plan.arguments(); fatalError("Nested output accepted") } catch {}
  let alias = root.appendingPathComponent("source alias")
  try FileManager.default.createSymbolicLink(at:alias,withDestinationURL:source)
  plan.output = alias.appendingPathComponent("nested").path
  do { _ = try plan.arguments(); fatalError("Symlink overlap accepted") } catch {}
  plan.output = output.path; plan.expertBits = 2.6
  do { _ = try plan.arguments(); fatalError("Unsupported expert rate accepted") } catch {}
  plan.expertBits = 2.625
  try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
  try Data("partial".utf8).write(to:output.appendingPathComponent("partial"))
  do { _ = try plan.arguments(); fatalError("Existing files overwritten") } catch {}
  plan.resume = true
  let resumed = try plan.arguments(); precondition(resumed.last == "--resume")
  if CommandLine.arguments.count > 1 {
   let task = ToolTask()
   try task.run(binary:CommandLine.arguments[1],arguments:["convert","--help"],title:"Converter help",timeout:10)
   precondition(!task.exclusiveServerUse)
   try await finish(task)
   precondition(task.status == "Completed" && task.converterAvailable == true)
   plan.output = root.appendingPathComponent("unsupported-output").path; plan.resume = false
   try task.run(binary:CommandLine.arguments[1],arguments:plan.arguments(syntax:task.converterSyntax ?? .legacy),title:"Unsupported source fixture",timeout:20)
   precondition(task.converting && task.exclusiveServerUse)
   try await finish(task)
   precondition(task.status.hasPrefix("Failed") && !task.exclusiveServerUse)
  }
  print("Conversion validation passed: exact argv, rate steps, path isolation, overwrite refusal, explicit resume")
 }
 @MainActor static func finish(_ task:ToolTask) async throws {
  let end = Date().addingTimeInterval(25)
  while task.running {
   if Date() > end { task.cancel(); throw ServerFailure(message:"Converter fixture exceeded deadline") }
   try await Task.sleep(nanoseconds:50_000_000)
  }
 }
}
