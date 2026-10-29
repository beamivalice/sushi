import Foundation

struct BenchmarkPlan {
    var model:String
    var words:Int
    var outputTokens:Int
    var rounds:Int
    var reusePrompt:Bool
    var timeout:Double
    var configuration = ""
    func body(nonce:String) throws -> [String:Any] {
        guard !model.isEmpty, [256,2048,8192,32768].contains(words), [32,128,512].contains(outputTokens), (1...5).contains(rounds), timeout.isFinite, (30...600).contains(timeout) else { throw ServerFailure(message:"Choose a model and supported benchmark limits.") }
        let vocabulary = ["river","stone","forest","cloud","morning","bridge","garden","light","window","path","water","hill","field","tree","house","sky"]
        let reference = (0..<words).map { vocabulary[$0 % vocabulary.count] }.joined(separator:" ")
        let prompt = "Benchmark \(nonce). Read this reference:\n\(reference)\nNow write a long numbered list of short facts about nature. Continue for at least 100 items."
        return ["model":model,"messages":[["role":"user","content":prompt]],"max_tokens":outputTokens,"temperature":0,"stream":false,"chat_template_kwargs":["enable_thinking":false]]
    }
}

struct BenchmarkResult:Identifiable,Codable {
    var id = UUID()
    var date = Date()
    let model:String
    let words:Int
    let outputLimit:Int
    let repeatedPrompt:Bool
    let round:Int
    let wallSeconds:Double
    let promptTokens:Int?
    let outputTokens:Int?
    let cachedTokens:Int?
    let prefillRate:Double?
    let decodeRate:Double?
    let configuration:String
    let finishReason:String
    var endpoint:String? = nil
    var wallRate:Double? { outputTokens.flatMap { wallSeconds > 0 ? Double($0)/wallSeconds : nil } }
    static func parse(_ object:[String:Any],plan:BenchmarkPlan,round:Int,wallSeconds:Double) throws -> BenchmarkResult {
        guard let choices = object["choices"] as? [[String:Any]], let choice = choices.first, wallSeconds.isFinite, wallSeconds > 0 else { throw ServerFailure(message:"The benchmark response did not contain a completed choice.") }
        let usage = object["usage"] as? [String:Any] ?? [:], timings = object["timings"] as? [String:Any] ?? [:]
        func count(_ value:Any?) -> Int? { ModelEntry.samplingNumber(value).flatMap { $0 < Double(Int.max) && $0.rounded() == $0 ? Int($0) : nil } }
        let prompt = count(usage["prompt_tokens"]), output = count(usage["completion_tokens"])
        let cached = count((usage["prompt_tokens_details"] as? [String:Any])?["cached_tokens"])
        var prefill:Double?, decode:Double?
        if let total = count(timings["prompt_n"]), let hit = count(timings["cached_n"]), hit <= total, let ms = ModelEntry.samplingNumber(timings["prompt_ms"]), ms > 0 { prefill = Double(total-hit)*1000/ms }
        if let tokens = count(timings["predicted_n"]), let ms = ModelEntry.samplingNumber(timings["predicted_ms"]), ms > 0 { decode = Double(tokens)*1000/ms }
        return BenchmarkResult(model:plan.model,words:plan.words,outputLimit:plan.outputTokens,repeatedPrompt:plan.reusePrompt,round:round,wallSeconds:wallSeconds,promptTokens:prompt,outputTokens:output,cachedTokens:cached.flatMap { prompt == nil || $0 <= prompt! ? $0 : nil },prefillRate:prefill,decodeRate:decode,configuration:plan.configuration,finishReason:choice["finish_reason"] as? String ?? "unknown")
    }
}

@MainActor final class BenchmarkRunner:ObservableObject {
    @Published private(set) var running = false
    @Published private(set) var status = ""
    @Published private(set) var results:[BenchmarkResult] = []
    @Published private(set) var persistenceError:String?
    @Published private(set) var historyBlocked = false
    @Published private(set) var backupURL:URL?
    private let storageURL:URL?
    private var savedData:Data?
    private var operation:Task<Void,Never>?
    init(storageURL:URL? = nil) {
        self.storageURL = storageURL
        do {
            try loadStoredHistory()
        } catch { historyBlocked = true; persistenceError = "Could not restore benchmark history. Its file has been kept unchanged. \(error.localizedDescription)" }
    }

    func start(plan:BenchmarkPlan,connection:Connection,fetch:((Connection,[String:Any],Double) async throws -> [String:Any])? = nil,onFinished:(()->Void)? = nil) throws {
        guard !running else { throw ServerFailure(message:"A benchmark is already running.") }
        guard !historyBlocked else { throw ServerFailure(message:"Recover the saved benchmark history before starting another run.") }
        do { try verifyStoredData() }
        catch { historyBlocked = true; persistenceError = error.localizedDescription; throw error }
        _ = try plan.body(nonce:"validation")
        _ = try connection.request("v1/chat/completions")
        running = true; status = "Starting…"
        operation = Task {
            defer { running = false; operation = nil; onFinished?() }
            let sharedNonce = UUID().uuidString
            do {
                for round in 1...plan.rounds {
                    try Task.checkCancellation()
                    status = "Run \(round) of \(plan.rounds)"
                    let body = try plan.body(nonce:plan.reusePrompt ? sharedNonce : UUID().uuidString)
                    let start = ProcessInfo.processInfo.systemUptime
                    let response:[String:Any]
                    if let fetch { response = try await fetch(connection,body,plan.timeout) }
                    else { response = try await connection.fetch("v1/chat/completions",body:body,timeout:plan.timeout) }
                    try Task.checkCancellation()
                    var result = try BenchmarkResult.parse(response,plan:plan,round:round,wallSeconds:ProcessInfo.processInfo.systemUptime-start)
                    result.endpoint = connection.endpoint
                    results.insert(result,at:0); results = Array(results.prefix(100))
                    guard save() else { status = "Stopped: results remain in memory; history could not be saved."; return }
                }
                status = "Completed"
            } catch {
                status = Task.isCancelled || error is CancellationError ? "Cancelled" : "Failed: \(error.localizedDescription)"
            }
        }
    }
    func cancel() { guard running else { return }; status = "Cancelling…"; operation?.cancel() }
    func clear() {
        guard !running else { return }
        do {
            if let storageURL {
                try FileManager.default.createDirectory(at:storageURL.deletingLastPathComponent(),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
                var backup:URL?
                if FileManager.default.fileExists(atPath:storageURL.path) {
                    let destination = storageURL.deletingLastPathComponent().appendingPathComponent("benchmarks-backup-\(UUID().uuidString).json")
                    try FileManager.default.moveItem(at:storageURL,to:destination); backup = destination; backupURL = destination
                }
                do {
                    let empty = try JSONEncoder().encode([BenchmarkResult]())
                    try empty.write(to:storageURL,options:.atomic)
                    try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:storageURL.path)
                    if let backup { try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:backup.path) }
                    savedData = empty
                } catch {
                    if let backup, !FileManager.default.fileExists(atPath:storageURL.path) {
                        try? FileManager.default.moveItem(at:backup,to:storageURL)
                        if !FileManager.default.fileExists(atPath:backup.path) { backupURL = nil }
                    }
                    throw error
                }
            }
            results.removeAll(); historyBlocked = false; persistenceError = nil; status = backupURL == nil ? "History cleared." : "History reset; previous file preserved in a backup."
        } catch { historyBlocked = true; persistenceError = "History could not be reset. \(error.localizedDescription)" }
    }
    func exportData() throws -> Data { let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]; encoder.dateEncodingStrategy = .iso8601; return try encoder.encode(results) }
    func reloadHistory() {
        guard !running else { return }
        do { try loadStoredHistory(); status = "Saved history reloaded." }
        catch { historyBlocked = true; persistenceError = "Could not reload benchmark history. Current results and the file have been kept. \(error.localizedDescription)" }
    }
    private func loadStoredHistory() throws {
        let data = try readStoredData()
        let loaded = try data.map { Array(try JSONDecoder().decode([BenchmarkResult].self,from:$0).prefix(100)) } ?? []
        results = loaded; savedData = data; historyBlocked = false; persistenceError = nil
    }
    private func readStoredData() throws -> Data? {
        guard let storageURL, FileManager.default.fileExists(atPath:storageURL.path) else { return nil }
        let size = try FileManager.default.attributesOfItem(atPath:storageURL.path)[.size] as? NSNumber
        guard let size, size.intValue <= 1024*1024 else { throw ServerFailure(message:"Benchmark history exceeds its size limit.") }
        return try Data(contentsOf:storageURL)
    }
    private func verifyStoredData() throws {
        guard storageURL != nil else { return }
        guard try readStoredData() == savedData else { throw ServerFailure(message:"Benchmark history changed outside this window. The file has been kept unchanged; export any in-memory results before recovering history.") }
    }
    private func save() -> Bool {
        guard let storageURL else { return true }
        do {
            try verifyStoredData()
            let data = try JSONEncoder().encode(results)
            guard data.count <= 1024*1024 else { throw ServerFailure(message:"Benchmark history exceeds its size limit. Export the in-memory results before resetting history.") }
            try FileManager.default.createDirectory(at:storageURL.deletingLastPathComponent(),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
            try data.write(to:storageURL,options:.atomic)
            savedData = data
            try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:storageURL.path)
            persistenceError = nil; return true
        } catch { historyBlocked = true; persistenceError = "Could not save benchmark history: \(error.localizedDescription)"; return false }
    }

}

struct BenchmarkMedian {
    let count:Int
    let median:Double?
    init(_ values:[Double?]) {
        let sorted = values.compactMap{$0}.filter{$0.isFinite && $0 >= 0}.sorted()
        count = sorted.count
        if sorted.isEmpty { median = nil }
        else if sorted.count % 2 == 1 { median = sorted[sorted.count/2] }
        else { median = sorted[sorted.count/2-1]/2+sorted[sorted.count/2]/2 }
    }
}
struct BenchmarkSummary:Identifiable {
    enum Cache:String,Hashable {
        case none, partial, full, unknown
        var title:String {
            switch self { case .none:"No prefix reuse"; case .partial:"Partial prefix reuse"; case .full:"Full prefix reuse"; case .unknown:"Cache usage unreported" }
        }
        init(_ sample:BenchmarkResult) {
            guard let cached = sample.cachedTokens, let prompt = sample.promptTokens, prompt > 0, cached >= 0, cached <= prompt else { self = .unknown; return }
            self = cached == 0 ? .none : cached == prompt ? .full : .partial
        }
    }
    struct Key:Hashable {
        let endpoint:String?
        let model:String
        let words:Int
        let outputLimit:Int
        let repeatedPrompt:Bool
        let configuration:String
        let promptTokens:Int?
        let cachedTokens:Int?
        let outputTokens:Int?
        let finishReason:String
        let cache:Cache
        let legacyID:UUID?
    }
    let key:Key
    let samples:[BenchmarkResult]
    var id:Key { key }
    var prefill:BenchmarkMedian { BenchmarkMedian(samples.map(\.prefillRate)) }
    var decode:BenchmarkMedian { BenchmarkMedian(samples.map(\.decodeRate)) }
    var wall:BenchmarkMedian { BenchmarkMedian(samples.map(\.wallRate)) }
    var duration:BenchmarkMedian { BenchmarkMedian(samples.map { $0.wallSeconds }) }
    static func group(_ samples:[BenchmarkResult])->[BenchmarkSummary] {
        let groups = Dictionary(grouping:samples) { sample in
            Key(endpoint:sample.endpoint,model:sample.model,words:sample.words,outputLimit:sample.outputLimit,repeatedPrompt:sample.repeatedPrompt,configuration:sample.configuration,promptTokens:sample.promptTokens,cachedTokens:sample.cachedTokens,outputTokens:sample.outputTokens,finishReason:sample.finishReason,cache:Cache(sample),legacyID:sample.endpoint == nil ? sample.id : nil)
        }
        return groups.map { BenchmarkSummary(key:$0.key,samples:$0.value) }.sorted {
            let first = $0.samples.map(\.date).max() ?? .distantPast, second = $1.samples.map(\.date).max() ?? .distantPast
            if first != second { return first > second }
            return ($0.samples.map { $0.id.uuidString }.min() ?? "") < ($1.samples.map { $0.id.uuidString }.min() ?? "")
        }
    }
    func range(_ values:[Int?])->String {
        let valid = values.compactMap{$0}.filter{$0 >= 0}
        guard let first = valid.min(), let last = valid.max() else { return "—" }
        return first == last ? first.formatted() : first.formatted()+"–"+last.formatted()
    }
}
