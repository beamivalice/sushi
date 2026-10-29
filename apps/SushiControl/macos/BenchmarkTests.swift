import Foundation

@main struct BenchmarkTests {
    @MainActor static func main() async throws {
        let plan = BenchmarkPlan(model:"fixture",words:256,outputTokens:32,rounds:2,reusePrompt:true,timeout:30)
        let first = try plan.body(nonce:"same"), second = try plan.body(nonce:"same")
        let firstData = try JSONSerialization.data(withJSONObject:first,options:.sortedKeys), secondData = try JSONSerialization.data(withJSONObject:second,options:.sortedKeys)
        precondition(firstData == secondData)
        let response:[String:Any] = ["choices":[["finish_reason":"length"]],"usage":["prompt_tokens":100,"completion_tokens":20,"prompt_tokens_details":["cached_tokens":80]],"timings":["prompt_n":100,"cached_n":80,"prompt_ms":10,"predicted_n":20,"predicted_ms":200]]
        let result = try BenchmarkResult.parse(response,plan:plan,round:1,wallSeconds:2)
        precondition(result.prefillRate == 2000 && result.decodeRate == 100 && result.wallRate == 10 && result.cachedTokens == 80)
        let old = try BenchmarkResult.parse(["choices":[["finish_reason":"stop"]]],plan:plan,round:1,wallSeconds:1)
        precondition(old.prefillRate == nil && old.decodeRate == nil && old.promptTokens == nil)
        var sample = result
        sample.endpoint = "http://127.0.0.1:12345"
        var secondSample = sample; secondSample.id = UUID()
        let same = BenchmarkSummary.group([sample,secondSample])
        precondition(same.count == 1 && same[0].samples.count == 2)
        precondition(same[0].decode.count == 2 && same[0].decode.median == result.decodeRate)
        var otherServer = sample; otherServer.id = UUID(); otherServer.endpoint = "http://127.0.0.1:8000"
        precondition(BenchmarkSummary.group([sample,otherServer]).count == 2)
        let uncached = BenchmarkResult(model:sample.model,words:sample.words,outputLimit:sample.outputLimit,repeatedPrompt:sample.repeatedPrompt,round:1,wallSeconds:sample.wallSeconds,promptTokens:sample.promptTokens,outputTokens:sample.outputTokens,cachedTokens:0,prefillRate:sample.prefillRate,decodeRate:sample.decodeRate,configuration:sample.configuration,finishReason:sample.finishReason,endpoint:sample.endpoint)
        precondition(BenchmarkSummary.group([sample,uncached]).count == 2)
        var legacyA = sample; legacyA.id = UUID(); legacyA.endpoint = nil
        var legacyB = legacyA; legacyB.id = UUID()
        precondition(BenchmarkSummary.group([legacyA,legacyB]).count == 2)
        let metric = BenchmarkMedian([1,3,2,4,Double.nan,-1])
        precondition(metric.count == 4 && metric.median == 2.5)
        let encodedSample = try JSONEncoder().encode(sample)
        var legacyObject = try JSONSerialization.jsonObject(with:encodedSample) as! [String:Any]
        legacyObject.removeValue(forKey:"endpoint")
        let legacyDecoded = try JSONDecoder().decode(BenchmarkResult.self,from:JSONSerialization.data(withJSONObject:legacyObject))
        precondition(legacyDecoded.endpoint == nil && legacyDecoded.model == sample.model)
        var differentResponse = response
        var differentUsage = response["usage"] as! [String:Any]
        differentUsage["prompt_tokens_details"] = ["cached_tokens":79]
        differentResponse["usage"] = differentUsage
        var differentTokens = try BenchmarkResult.parse(differentResponse,plan:plan,round:1,wallSeconds:2)
        differentTokens.endpoint = sample.endpoint
        precondition(BenchmarkSummary.group([sample,differentTokens]).count == 2)
        precondition(BenchmarkMedian([]).median == nil)
        let runner = BenchmarkRunner()
        var calls = 0
        try runner.start(plan:plan,connection:Connection(endpoint:"http://127.0.0.1:1"),fetch:{ _,_,_ in calls += 1; return response })
        try await finish(runner)
        precondition(calls == 2 && runner.results.count == 2 && runner.status == "Completed")
        precondition(runner.results.allSatisfy{$0.endpoint == "http://127.0.0.1:1"})
        try runner.start(plan:plan,connection:Connection(endpoint:"http://127.0.0.1:1"),fetch:{ _,_,_ in try await Task.sleep(nanoseconds:5_000_000_000); return response })
        runner.cancel(); try await finish(runner)
        precondition(runner.status == "Cancelled" && runner.results.count == 2)
        let historyDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:historyDir) }
        try FileManager.default.createDirectory(at:historyDir,withIntermediateDirectories:true)
        let historyURL = historyDir.appendingPathComponent("benchmarks.json")
        let corrupt = Data("unreadable history".utf8)
        try corrupt.write(to:historyURL)
        let broken = BenchmarkRunner(storageURL:historyURL)
        precondition(broken.historyBlocked)
        do { try broken.start(plan:plan,connection:Connection(endpoint:"http://127.0.0.1:1"),fetch:{ _,_,_ in fatalError("Ran inference with corrupt history") }); fatalError("Accepted corrupt history") } catch {}
        let preserved = try Data(contentsOf:historyURL)
        precondition(preserved == corrupt)
        broken.clear()
        precondition(!broken.historyBlocked && broken.backupURL != nil)
        let backedUp = try Data(contentsOf:broken.backupURL!)
        precondition(backedUp == corrupt && BenchmarkRunner(storageURL:historyURL).results.isEmpty)
        let changing = BenchmarkRunner(storageURL:historyURL)
        try changing.start(plan:plan,connection:Connection(endpoint:"http://127.0.0.1:1"),fetch:{ _,_,_ in try corrupt.write(to:historyURL); return response })
        try await finish(changing)
        precondition(changing.historyBlocked && changing.results.count == 1 && changing.persistenceError != nil)
        let externallyChanged = try Data(contentsOf:historyURL)
        precondition(externallyChanged == corrupt)
        changing.reloadHistory()
        precondition(changing.historyBlocked && changing.results.count == 1)
        try JSONEncoder().encode([sample]).write(to:historyURL)
        changing.reloadHistory()
        precondition(!changing.historyBlocked && changing.results.first?.id == sample.id)
        print("Benchmark tests passed: request, cache-aware timings, missing fields, serial runs, cancellation")
    }
    @MainActor static func finish(_ runner:BenchmarkRunner) async throws {
        let deadline = Date().addingTimeInterval(5)
        while runner.running {
            if Date() > deadline { runner.cancel(); throw ServerFailure(message:"Benchmark test timed out") }
            try await Task.sleep(nanoseconds:20_000_000)
        }
    }
}
