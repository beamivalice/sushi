import Foundation
import Combine
import CoreFoundation

struct UsageCounts:Codable {
    var prompt:Double = 0
    var prefill:Double = 0
    var generated:Double = 0
    var cached:Double = 0
    var requests:Double = 0
    var values:[Double] { [prompt,prefill,generated,cached,requests] }
    var valid:Bool { values.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1e15 && $0.rounded() == $0 } }
    init() {}
    init?(metrics:[String:Any]) {
        guard let counters = metrics["counters"] as? [String:Any] else { return nil }
        func number(_ key:String)->Double? {
            guard let value = counters[key] as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID() else { return nil }
            return value.doubleValue
        }
        guard let prompt = number("prompt_tokens_total"), let generated = number("generation_tokens_total"),
              let cached = number("prefix_cache_tokens_total"), let requests = number("requests_success_total") else { return nil }
        self.prompt = prompt; self.generated = generated; self.cached = cached; self.requests = requests
        prefill = number("prefill_tokens_total") ?? max(0,prompt-cached)
        guard valid else { return nil }
    }
    func delta(from previous:UsageCounts)->UsageCounts? {
        guard zip(values,previous.values).allSatisfy({ $0 >= $1 }) else { return nil }
        var result = UsageCounts()
        result.prompt = prompt-previous.prompt; result.prefill = prefill-previous.prefill
        result.generated = generated-previous.generated; result.cached = cached-previous.cached; result.requests = requests-previous.requests
        return result
    }
    mutating func add(_ counts:UsageCounts) {
        prompt += counts.prompt; prefill += counts.prefill; generated += counts.generated; cached += counts.cached; requests += counts.requests
    }
}
struct UsageBucket:Codable,Identifiable {
    let endpoint:String
    let hour:Date
    var counts:UsageCounts
    var id:String { endpoint+":"+String(hour.timeIntervalSince1970) }
}
private struct UsageArchive:Codable { var version = 1; let buckets:[UsageBucket] }
@MainActor final class UsageHistory:ObservableObject {
    @Published private(set) var buckets:[UsageBucket] = []
    @Published private(set) var error = ""
    private let storageURL:URL?
    private var writable = true
    private var previous:(endpoint:String,time:Date,counts:UsageCounts)?
    init(storageURL:URL? = nil) {
        self.storageURL = storageURL
        guard let storageURL, FileManager.default.fileExists(atPath:storageURL.path) else { return }
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath:storageURL.path)
            guard let size = attributes[.size] as? NSNumber, size.intValue <= 4*1024*1024 else { throw ServerFailure(message:"Usage history exceeds 4 MiB.") }
            let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
            let archive = try decoder.decode(UsageArchive.self,from:Data(contentsOf:storageURL))
            guard archive.version == 1, archive.buckets.count <= 8192,
                  Set(archive.buckets.map(\.id)).count == archive.buckets.count,
                  archive.buckets.allSatisfy({ $0.counts.valid && $0.hour.timeIntervalSince1970.isFinite && (try? Connection(endpoint:$0.endpoint).request("health")) != nil }) else {
                throw ServerFailure(message:"Unsupported usage history format.")
            }
            buckets = archive.buckets
        } catch { writable = false; self.error = "Saved usage history could not be opened. Its file has been kept unchanged. \(error.localizedDescription)" }
    }
    func resetSampling() { previous = nil }
    func record(endpoint:String,metrics:[String:Any],at time:Date = Date()) {
        guard writable, time.timeIntervalSince1970.isFinite,
              (try? Connection(endpoint:endpoint).request("health")) != nil,
              let counts = UsageCounts(metrics:metrics) else { resetSampling(); return }
        let oldCount = buckets.count
        buckets.removeAll { $0.hour < time.addingTimeInterval(-30*86400) }
        defer { previous = (endpoint,time,counts) }
        guard let previous, previous.endpoint == endpoint, time > previous.time,
              time.timeIntervalSince(previous.time) <= 30, let delta = counts.delta(from:previous.counts), delta.values.contains(where:{$0 > 0}) else {
            if buckets.count != oldCount { save() }; return
        }
        let hour = Date(timeIntervalSince1970:floor(time.timeIntervalSince1970/3600)*3600)
        if let index = buckets.firstIndex(where:{$0.endpoint == endpoint && $0.hour == hour}) { buckets[index].counts.add(delta) }
        else { buckets.append(UsageBucket(endpoint:endpoint,hour:hour,counts:delta)) }
        buckets = Array(buckets.sorted { $0.hour < $1.hour }.suffix(8192))
        save()
    }
    func rows(endpoint:String,since:Date)->[UsageBucket] { buckets.filter { $0.endpoint == endpoint && $0.hour >= since }.sorted { $0.hour < $1.hour } }
    func export(endpoint:String) throws -> Data {
        try encoded(buckets.filter { $0.endpoint == endpoint })
    }
    private func encoded(_ rows:[UsageBucket]) throws -> Data {
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601; encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
        return try encoder.encode(UsageArchive(buckets:rows))
    }
    private func save() {
        guard let storageURL, writable else { return }
        do {
            try FileManager.default.createDirectory(at:storageURL.deletingLastPathComponent(),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
            let data = try encoded(buckets)
            try data.write(to:storageURL,options:.atomic)
            try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:storageURL.path)
            error = ""
        } catch { self.error = "Usage history could not be saved. \(error.localizedDescription)" }
    }
}
