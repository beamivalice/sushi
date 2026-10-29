import Foundation
@main struct IntegrationTests {
 static func main() async throws {
  let endpoint = CommandLine.arguments[1], key = CommandLine.arguments[2]
  let client = Connection(endpoint:endpoint,apiKey:key)
  _ = try await client.fetch("health")
  let models = try await client.fetch("v1/models")
  precondition(ModelEntry.parse(models).isEmpty)
  _ = try await client.fetch("props")
  _ = try await client.fetch("v1/models/rescan",body:[:])
  do { _ = try await client.fetch("v1/load-model",body:["model":"missing-test-model"]); fatalError("Unknown model accepted") }
  catch { precondition(error.localizedDescription.contains("404")) }
  do { _ = try await Connection(endpoint:endpoint,apiKey:"wrong-key").fetch("v1/models"); fatalError("Wrong key accepted") }
  catch { precondition(error.localizedDescription.contains("401")) }
  if CommandLine.arguments[3] == "metrics" {
   let metrics = try await client.fetch("metrics.json"); precondition(metrics["gauges"] is [String:Any])
   let stats = ServingStats(metrics)
   precondition(stats.available && stats.successfulRequests == 0 && stats.prefillRate == nil)
   precondition(stats.csv.contains("Successful requests,0"))
  } else {
   do { _ = try await client.fetch("metrics.json"); fatalError("Disabled metrics accepted") }
   catch { precondition(error.localizedDescription.contains("503")) }
  }
  print("Stock HTTP compatibility passed (\(CommandLine.arguments[3]))")
 }
}
