import Foundation

@main struct BenchmarkHTTPTests {
    @MainActor static func main() async throws {
        let connection = Connection(endpoint:CommandLine.arguments[1],apiKey:"fixture-key")
        let plan = BenchmarkPlan(model:"fixture",words:256,outputTokens:32,rounds:2,reusePrompt:true,timeout:30)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:dir) }
        let file = dir.appendingPathComponent("benchmarks.json"), runner = BenchmarkRunner(storageURL:dir.appendingPathComponent("benchmarks.json"))
        try runner.start(plan:plan,connection:connection)
        try await finish(runner)
        precondition(runner.status == "Completed" && runner.results.count == 2)
        precondition(runner.results.allSatisfy { $0.prefillRate == 2000 && $0.decodeRate == 100 && $0.outputTokens == 20 })
        let restored = BenchmarkRunner(storageURL:file)
        precondition(restored.results.count == 2 && !restored.running)
        let exported = try JSONSerialization.jsonObject(with:restored.exportData()) as! [[String:Any]]
        precondition(exported.count == 2 && exported[0]["model"] as? String == "fixture" && exported[0]["apiKey"] == nil)
        var slow = plan; slow.model = "slow"
        try runner.start(plan:slow,connection:connection)
        try await Task.sleep(nanoseconds:100_000_000)
        runner.cancel(); try await finish(runner)
        precondition(runner.status == "Cancelled" && runner.results.count == 2)
        do { _ = try await connection.fetch("v1/chat/completions",body:slow.body(nonce:"timeout"),timeout:0.1); fatalError("Deadline ignored") }
        catch let error as URLError { precondition(error.code == .timedOut) }
        do { _ = try await connection.fetch("redirect",timeout:2); fatalError("Redirect followed") }
        catch let error as ServerFailure { precondition(error.status == 302) }
        for path in ["oversize-declared","oversize-streamed"] {
            do { _ = try await connection.fetch(path,timeout:5); fatalError("Accepted oversized response") }
            catch let error as ServerFailure { precondition(error.message.contains("8 MiB")) }
        }
        do { _ = try await connection.fetch("echo-secret"); fatalError("Accepted error response") }
        catch let error as ServerFailure {
            precondition(!error.message.contains("fixture-key") && error.message.contains("[redacted]") && error.message.count < 2200)
        }
        restored.clear()
        precondition(BenchmarkRunner(storageURL:file).results.isEmpty)
        print("Benchmark HTTP passed: authenticated requests, timings, persistence, cancellation, deadline, redirect refusal, bounded declared/streamed responses and redacted errors")
    }
    @MainActor static func finish(_ runner:BenchmarkRunner) async throws {
        let deadline = Date().addingTimeInterval(5)
        while runner.running {
            if Date() > deadline { runner.cancel(); throw ServerFailure(message:"HTTP benchmark fixture timed out") }
            try await Task.sleep(nanoseconds:20_000_000)
        }
    }
}
