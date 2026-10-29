import Foundation

@main struct UsageHistoryTests {
    @MainActor static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let url = root.appendingPathComponent("usage.json")
        precondition(UsageCounts(metrics:[:]) == nil)
        precondition(UsageCounts(metrics:["counters":["prompt_tokens_total":true]]) == nil)
        let history = UsageHistory(storageURL:url)
        let start = Date(timeIntervalSince1970:1_800_000_000)
        func metrics(_ value:Double) -> [String:Any] { ["counters":["prompt_tokens_total":value*10,"prefill_tokens_total":value*2,"generation_tokens_total":value,"prefix_cache_tokens_total":value*8,"requests_success_total":value]] }
        history.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(100),at:start)
        precondition(history.buckets.isEmpty)
        history.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(110),at:start.addingTimeInterval(3))
        precondition(history.buckets.count == 1 && history.buckets[0].counts.generated == 10)
        history.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(110),at:start.addingTimeInterval(6))
        precondition(history.buckets[0].counts.generated == 10)
        history.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(1),at:start.addingTimeInterval(9))
        history.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(2),at:start.addingTimeInterval(12))
        precondition(history.buckets[0].counts.generated == 11)
        history.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(1000),at:start.addingTimeInterval(60))
        precondition(history.buckets[0].counts.generated == 11)
        history.resetSampling()
        history.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(2000),at:start.addingTimeInterval(63))
        precondition(history.buckets[0].counts.generated == 11)
        history.record(endpoint:"http://127.0.0.1:9000",metrics:metrics(3000),at:start.addingTimeInterval(66))
        precondition(history.buckets.count == 1)
        history.record(endpoint:"http://127.0.0.1:9000",metrics:metrics(3002),at:start.addingTimeInterval(69))
        precondition(history.buckets.count == 2)
        let restored = UsageHistory(storageURL:url)
        precondition(restored.buckets.count == 2 && restored.error.isEmpty)
        let permissions = try FileManager.default.attributesOfItem(atPath:url.path)[.posixPermissions] as? Int
        precondition(permissions == 0o600)
        let exported = try restored.export(endpoint:"http://127.0.0.1:8000")
        precondition(String(decoding:exported,as:UTF8.self).contains("generated"))
        history.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(10000),at:start.addingTimeInterval(31*86400))
        precondition(history.buckets.isEmpty)
        try Data("invalid".utf8).write(to:url)
        let malformed = UsageHistory(storageURL:url)
        precondition(!malformed.error.isEmpty)
        malformed.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(1),at:start)
        malformed.record(endpoint:"http://127.0.0.1:8000",metrics:metrics(2),at:start.addingTimeInterval(3))
        let preserved = try Data(contentsOf:url)
        precondition(preserved == Data("invalid".utf8))
        print("Usage history passed: sample deltas, reset/gap isolation, endpoint separation, retention, persistence, export, corruption refusal")
    }
}
