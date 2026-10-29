import Foundation
import CoreFoundation

final class LocalRedirectPolicy: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

struct Connection {
    var endpoint: String
    var apiKey = ""
    func request(_ path: String, body: [String:Any]? = nil) throws -> URLRequest {
        guard let url = URL(string:endpoint), url.scheme == "http",
              ["localhost", "127.0.0.1", "[::1]"].contains(url.host ?? ""),
              url.user == nil, url.password == nil, url.query == nil, url.fragment == nil,
              url.path.isEmpty || url.path == "/" else {
            throw ServerFailure(message:"Enter a local HTTP server root, for example http://127.0.0.1:12345.")
        }
        var request = URLRequest(url:url.appendingPathComponent(path))
        request.timeoutInterval = body == nil ? 5 : 180
        if !apiKey.isEmpty { request.setValue("Bearer \(apiKey)", forHTTPHeaderField:"Authorization") }
        if let body {
            request.httpMethod = "POST"
            request.httpBody = try JSONSerialization.data(withJSONObject:body)
            request.setValue("application/json", forHTTPHeaderField:"Content-Type")
        }
        return request
    }
    func fetch(_ path:String, body:[String:Any]? = nil,timeout:Double? = nil) async throws -> [String:Any] {
        if let timeout, !timeout.isFinite || timeout <= 0 || timeout > 600 { throw ServerFailure(message:"Request timeout must be between 0 and 600 seconds.") }
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForResource = timeout ?? (body == nil ? 15 : 180)
        config.timeoutIntervalForRequest = timeout ?? (body == nil ? 5 : 180)
        config.httpShouldSetCookies = false; config.httpCookieStorage = nil; config.urlCredentialStorage = nil; config.urlCache = nil
        config.connectionProxyDictionary = [:]
        let session = URLSession(configuration:config,delegate:LocalRedirectPolicy(),delegateQueue:nil)
        defer { session.invalidateAndCancel() }
        var request = try request(path,body:body)
        if let timeout { request.timeoutInterval = timeout }
        let (bytes,response) = try await session.bytes(for:request)
        defer { bytes.task.cancel() }
        let limit = 8*1024*1024
        guard response.expectedContentLength <= limit else { throw ServerFailure(message:"The server response exceeds 8 MiB.") }
        var data = Data()
        for try await byte in bytes {
            try Task.checkCancellation()
            guard data.count < limit else { throw ServerFailure(message:"The server response exceeds 8 MiB.") }
            data.append(byte)
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else { throw ServerFailure(status:status,data:data,redacting:apiKey) }
        guard let object = try JSONSerialization.jsonObject(with:data) as? [String:Any] else {
            throw ServerFailure(message:"The server returned an unexpected response.")
        }
        return object
    }
}
struct ServerFailure: LocalizedError {
    var status:Int? = nil
    var message:String
    var errorDescription:String? { message }
    init(message:String) { self.message = message }
    init(status:Int,data:Data,redacting secret:String = "") {
        self.status = status
        let object = (try? JSONSerialization.jsonObject(with:data)) as? [String:Any] ?? [:]
        let detail = (object["error"] as? [String:Any])?["message"] as? String ?? object["error"] as? String
        let text = detail ?? String(decoding:data,as:UTF8.self)
        let safeText = secret.isEmpty ? text : text.replacingOccurrences(of:secret,with:"[redacted]")
        message = "HTTP \(status): \(safeText.prefix(2000))"
    }
}
struct ModelEntry: Identifiable {
    var id:String
    var loaded:Bool
    var state:String
    var architecture:String
    var quantization:String
    var contextTokens:Int?
    var residentBytes:Double?
    var capabilities:[String] = []
    var inputModalities:[String] = []
    var reasoningEfforts:[String] = []
    var defaultReasoningEffort:String? = nil
    var maximumContext:Int? = nil
    var diskBytes:Double? = nil
    var kvFormat:String? = nil
    var drafterLoaded:Bool? = nil
    var mtpLoaded:Bool? = nil
    var recommendedTemperature:Double? = nil
    var recommendedTopP:Double? = nil
    var recommendedTopK:Int? = nil
    static func samplingNumber(_ value:Any?) -> Double? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        let value = number.doubleValue
        return value.isFinite && value >= 0 ? value : nil
    }
    static func library(_ models:[ModelEntry],query:String = "",scope:String = "All",favorites:Set<String> = [],sort:String = "Name") -> [ModelEntry] {
        let search = query.trimmingCharacters(in:.whitespacesAndNewlines)
        return models.filter { model in
            let included = scope == "Favorites" ? favorites.contains(model.id) : scope == "Loaded" ? model.loaded : scope == "Menu" ? model.loaded || favorites.contains(model.id) : true
            return included && (search.isEmpty || [model.id,model.architecture,model.quantization].contains { $0.localizedCaseInsensitiveContains(search) })
        }.sorted { a,b in
            if scope == "Menu" || sort == "Menu" {
                if a.loaded != b.loaded { return a.loaded }
                if favorites.contains(a.id) != favorites.contains(b.id) { return favorites.contains(a.id) }
            }
            if sort == "Memory", a.residentBytes != b.residentBytes { return (a.residentBytes ?? -1) > (b.residentBytes ?? -1) }
            if sort == "Context", a.contextTokens != b.contextTokens { return (a.contextTokens ?? -1) > (b.contextTokens ?? -1) }
            return a.id.localizedStandardCompare(b.id) == .orderedAscending
        }
    }
    static func selection(_ current:String,in models:[ModelEntry]) -> String {
        if models.contains(where:{$0.id == current}) { return current }
        return models.first(where:{$0.loaded})?.id ?? models.first?.id ?? ""
    }
    static func parse(_ object:[String:Any]) -> [ModelEntry] {
        (object["data"] as? [[String:Any]] ?? []).compactMap { row in
            guard let id = row["id"] as? String else { return nil }
            return ModelEntry(id:id, loaded:row["loaded"] as? Bool ?? false,
                state:row["state"] as? String ?? "unknown",
                architecture:(row["meta"] as? [String:Any])?["architecture"] as? String ?? "",
                quantization:(row["meta"] as? [String:Any])?["quantization"] as? String ?? "",
                contextTokens:row["context_length"] as? Int, residentBytes:row["bytes_resident"] as? Double,
                capabilities:row["capabilities"] as? [String] ?? [],inputModalities:row["input_modalities"] as? [String] ?? [],
                reasoningEfforts:row["reasoning_efforts"] as? [String] ?? [],defaultReasoningEffort:row["default_reasoning_effort"] as? String,
                maximumContext:row["max_model_len"] as? Int,diskBytes:row["bytes_on_disk"] as? Double,
                kvFormat:(row["meta"] as? [String:Any])?["kv_quant"] as? String,
                drafterLoaded:(row["meta"] as? [String:Any])?["drafter_loaded"] as? Bool,
                mtpLoaded:(row["meta"] as? [String:Any])?["mtp_loaded"] as? Bool,
                recommendedTemperature:samplingNumber((row["meta"] as? [String:Any])?["gen_temperature"]),
                recommendedTopP:samplingNumber((row["meta"] as? [String:Any])?["gen_top_p"]).flatMap { $0 <= 1 ? $0 : nil },
                recommendedTopK:samplingNumber((row["meta"] as? [String:Any])?["gen_top_k"]).flatMap { $0 < Double(Int.max) && $0.rounded() == $0 ? Int($0) : nil })
        }
    }
    static func decode(_ data:Data) throws -> [ModelEntry] {
        parse(try JSONSerialization.jsonObject(with:data) as? [String:Any] ?? [:])
    }
}

struct LaunchSettings {
    var port = 12345
    var modelDirectory = ""
    var context = ""
    var memoryGiB = ""
    var idleMinutes = 0
    var kvBits = 8
    var metrics = true
    var ramCacheGiB = ""
    var diskCacheGiB = ""
    var prefillChunk = ""
    var drafter = ""
    func arguments() throws -> [String] {
        guard (1...65535).contains(port), idleMinutes >= 0, idleMinutes <= 10080 else { throw ServerFailure(message:"Enter a valid port and idle timeout.") }
        var result = ["serve","--host","127.0.0.1","--port",String(port)]
        if metrics { result.append("--metrics") }
        if !modelDirectory.isEmpty {
            guard modelDirectory.hasPrefix("/") else { throw ServerFailure(message:"Choose a full path for the models folder.") }
            result += ["--model-dir",modelDirectory]
        }
        if !context.isEmpty {
            guard let tokens = Int(context), tokens > 0, tokens <= Int(UInt32.max) else { throw ServerFailure(message:"Context must be a positive token count, or empty for automatic.") }
            result += ["--ctx-size",String(tokens)]
        }
        if !memoryGiB.isEmpty {
            guard let gib = Double(memoryGiB), gib.isFinite, gib >= 1, gib <= 4096 else { throw ServerFailure(message:"Memory must be between 1 and 4096 GiB, or empty for automatic.") }
            result += ["--max-resident-mem",String(UInt64(gib * 1073741824))]
        }
        if idleMinutes > 0 { result += ["--idle-evict-secs",String(idleMinutes * 60)] }
        guard [4,6,8,16].contains(kvBits) else { throw ServerFailure(message:"Choose a supported KV cache width.") }
        for (flag,value) in [("--prefix-cache-mem",ramCacheGiB),("--prefix-cache-disk",diskCacheGiB)] where !value.isEmpty {
            guard let gib = Double(value), gib.isFinite, gib >= 0, gib <= 4096 else { throw ServerFailure(message:"Cache budgets must be between 0 and 4096 GiB.") }
            result += [flag,String(UInt64(gib * 1073741824))]
        }
        if !prefillChunk.isEmpty {
            guard let chunk = UInt32(prefillChunk), chunk > 0 else { throw ServerFailure(message:"Prefill chunk must be a positive token count.") }
            result += ["--prefill-chunk",String(chunk)]
        }
        if !drafter.isEmpty {
            guard drafter.hasPrefix("/") else { throw ServerFailure(message:"Enter a full path for the assistant model.") }
            result += ["--drafter",drafter]
        }
        result += ["--kv-quant",String(kvBits)]
        return result
    }
}
struct ThroughputSample: Identifiable {
    var id:Date { time }
    var time:Date
    var tokensPerSecond:Double
    var memoryGiB:Double?
    var startsNewSegment = false
}
struct TimelineValue:Identifiable {
    var id:Date { time }
    var time:Date
    var value:Double
    var segment:Int
    static func points(_ samples:[ThroughputSample],value:(ThroughputSample)->Double?,maximumGap:TimeInterval)->[TimelineValue] {
        var result:[TimelineValue] = []
        var last:Date?
        var segment = 0
        for sample in samples {
            if sample.startsNewSegment { last = nil; segment += 1 }
            guard let amount = value(sample), amount.isFinite else { last = nil; segment += 1; continue }
            if let last, sample.time.timeIntervalSince(last) > maximumGap { segment += 1 }
            result.append(TimelineValue(time:sample.time,value:amount,segment:segment))
            last = sample.time
        }
        return result
    }
}
struct ThroughputHistory {
    static let maximumInterval:TimeInterval = 30
    private var previous:(Date,Double)?
    private var startsNewSegment = false
    var samples:[ThroughputSample] = []
    var currentRate:Double? { guard let previous, let sample = samples.last, sample.time == previous.0 else { return nil }; return sample.tokensPerSecond }
    mutating func reset() { previous = nil; startsNewSegment = false; samples = [] }
    mutating func record(time:Date,tokens:Double?,memoryBytes:Double?) {
        guard let tokens, tokens.isFinite, tokens >= 0 else { previous = nil; startsNewSegment = true; return }
        guard let (lastTime,lastTokens) = previous else { previous = (time,tokens); return }
        guard time > lastTime else { return }
        previous = (time,tokens)
        guard time.timeIntervalSince(lastTime) <= Self.maximumInterval else { startsNewSegment = true; return }
        guard tokens >= lastTokens else { startsNewSegment = true; return }
        samples.append(ThroughputSample(time:time,tokensPerSecond:(tokens-lastTokens)/time.timeIntervalSince(lastTime),memoryGiB:memoryBytes.map{$0/1073741824},startsNewSegment:startsNewSegment))
        startsNewSegment = false
        samples.removeAll { $0.time < time.addingTimeInterval(-180) }
        if samples.count > 120 { samples.removeFirst(samples.count-120) }
    }
}

struct PrefillHistory {
    private var previous:(time:Date,tokens:Double,changed:Date,rate:Double)?
    private var startsNewSegment = false
    var samples:[ThroughputSample] = []
    var currentRate:Double? { guard let previous, let sample = samples.last, sample.time == previous.time else { return nil }; return sample.tokensPerSecond }
    mutating func reset() { previous = nil; startsNewSegment = false; samples = [] }
    mutating func record(time:Date,tokens:Double?) {
        guard let tokens, tokens.isFinite, tokens >= 0 else { previous = nil; startsNewSegment = true; return }
        guard let last = previous else { previous = (time,tokens,time,0); return }
        guard time > last.time else { return }
        let interval = time.timeIntervalSince(last.time)
        guard interval <= ThroughputHistory.maximumInterval else { previous = (time,tokens,time,0); startsNewSegment = true; return }
        if tokens < last.tokens {
            let rate = tokens / interval
            previous = (time,tokens,time,rate)
            samples.append(ThroughputSample(time:time,tokensPerSecond:rate,memoryGiB:nil,startsNewSegment:startsNewSegment))
            startsNewSegment = false
            samples.removeAll { $0.time < time.addingTimeInterval(-180) }
            if samples.count > 120 { samples.removeFirst(samples.count-120) }
            return
        }
        let changed = tokens != last.tokens
        let elapsed = min(time.timeIntervalSince(last.changed),max(3,time.timeIntervalSince(last.time)))
        let rate = changed ? (tokens-last.tokens)/elapsed : (time.timeIntervalSince(last.changed) > 3.2 ? 0 : last.rate)
        previous = (time,tokens,changed ? time : last.changed,rate)
        samples.append(ThroughputSample(time:time,tokensPerSecond:rate,memoryGiB:nil,startsNewSegment:startsNewSegment))
        startsNewSegment = false
        samples.removeAll { $0.time < time.addingTimeInterval(-180) }
        if samples.count > 120 { samples.removeFirst(samples.count-120) }
    }
}

struct PrefixCacheStatus {
    var ramEnabled:Bool?
    var ramBudget:Double?
    var diskBudget:Double?
    var diskUsed:Double?
    var diskEntries:Int?
    init(_ props:[String:Any]) {
        let cache = (props["settings"] as? [String:Any])?["prefix_cache"] as? [String:Any] ?? [:]
        ramEnabled = cache["ram_enabled"] as? Bool
        ramBudget = Self.bytes(cache["mem_bytes"])
        diskBudget = Self.bytes(cache["disk_bytes"])
        diskUsed = Self.bytes(cache["disk_used_bytes"])
        diskEntries = (cache["disk_entries"] as? Int).flatMap { $0 >= 0 ? $0 : nil }
    }
    var available:Bool { ramEnabled != nil || ramBudget != nil || diskBudget != nil || diskUsed != nil }
    var diskFraction:Double? {
        guard let diskBudget, diskBudget > 0, let diskUsed else { return nil }
        return min(1,diskUsed/diskBudget)
    }
    private static func bytes(_ value:Any?)->Double? {
        guard let number = value as? Double, number.isFinite, number >= 0 else { return nil }
        return number
    }
}

struct ServingStats {
    private let counters:[String:Any]
    private let histograms:[String:Any]
    init(_ metrics:[String:Any]) {
        counters = metrics["counters"] as? [String:Any] ?? [:]
        histograms = metrics["histograms"] as? [String:Any] ?? [:]
    }
    static func number(_ value:Any?) -> Double? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite, number.doubleValue >= 0 else { return nil }
        return number.doubleValue
    }
    private func counter(_ key:String) -> Double? { Self.number(counters[key]) }
    private func histogram(_ key:String,_ field:String) -> Double? { Self.number((histograms[key] as? [String:Any])?[field]) }
    private func ratio(_ numerator:Double?,_ denominator:Double?) -> Double? {
        guard let numerator, let denominator, denominator > 0 else { return nil }
        let value = numerator/denominator
        return value.isFinite ? value : nil
    }
    var promptTokens:Double? { counter("prompt_tokens_total") }
    var cachedTokens:Double? { counter("prefix_cache_tokens_total") }
    var outputTokens:Double? { counter("generation_tokens_total") }
    var forwardedTokens:Double? {
        if counters["prefill_tokens_total"] != nil { return counter("prefill_tokens_total") }
        guard let promptTokens, let cachedTokens, promptTokens >= cachedTokens else { return nil }
        return promptTokens-cachedTokens
    }
    var successfulRequests:Double? { counter("requests_success_total") }
    var cancelledRequests:Double? { counter("requests_cancelled_total") }
    var failedRequests:Double? { counter("requests_failed_total") }
    var rejectedRequests:Double? { counter("requests_rejected_total") }
    var totalRequests:Double? {
        guard let successfulRequests, let cancelledRequests, let failedRequests, let rejectedRequests else { return nil }
        let total = successfulRequests+cancelledRequests+failedRequests+rejectedRequests
        return total.isFinite ? total : nil
    }
    var available:Bool { promptTokens != nil || outputTokens != nil || successfulRequests != nil }
    var prefillRate:Double? { ratio(forwardedTokens,histogram("prefill_time_seconds","sum")) }
    var decodeRate:Double? { ratio(outputTokens,histogram("decode_time_seconds","sum")) }
    var meanTTFT:Double? { ratio(histogram("time_to_first_token_seconds","sum"),histogram("time_to_first_token_seconds","count")) }
    var cachedFraction:Double? { ratio(cachedTokens,promptTokens).map { min(1,$0) } }
    var cacheHitFraction:Double? { ratio(counter("prefix_cache_hits_total"),counter("prefix_cache_queries_total")).map { min(1,$0) } }
    var csv:String {
        let fields:[(String,Double?,String)] = [
            ("Requests",totalRequests,"requests"),("Successful requests",successfulRequests,"requests"),
            ("Cancelled requests",cancelledRequests,"requests"),("Failed requests",failedRequests,"requests"),("Rejected requests",rejectedRequests,"requests"),
            ("Prompt tokens",promptTokens,"tokens"),("Forwarded prompt tokens",forwardedTokens,"tokens"),("Cached tokens",cachedTokens,"tokens"),("Generated tokens",outputTokens,"tokens"),
            ("Average prefill speed",prefillRate,"tokens/second"),("Average generation speed",decodeRate,"tokens/second"),("Mean time to first token",meanTTFT,"seconds"),
            ("Cached token fraction",cachedFraction,"fraction"),("Cache lookup hit fraction",cacheHitFraction,"fraction")]
        return "Metric,Value,Unit\n" + fields.map { name,value,unit in name+","+(value.map { String($0) } ?? "")+","+unit }.joined(separator:"\n") + "\n"
    }
}

enum LaunchClient:String,CaseIterable,Identifiable {
    case pi, hermes, claude, opencode, codex, omp, aider, zcode, grok
    var id:String { rawValue }
    var title:String {
        switch self {
        case .pi: "Pi"
        case .hermes: "Hermes"
        case .claude: "Claude Code"
        case .opencode: "OpenCode"
        case .codex: "Codex"
        case .omp: "Oh My Pi"
        case .aider: "Aider"
        case .zcode: "ZCode"
        case .grok: "Grok CLI"
        }
    }
}
struct AgentLaunchCommand {
    static func make(binary:String,client:String,endpoint:String,model:String?) throws -> String {
        guard LaunchClient(rawValue:client) != nil else { throw ServerFailure(message:"Choose a supported client.") }
        _ = try Connection(endpoint:endpoint).request("health")
        guard !binary.contains("\0"), model?.contains("\0") != true else { throw ServerFailure(message:"The command contains an invalid character.") }
        guard binary.isEmpty || (binary.hasPrefix("/") && FileManager.default.isExecutableFile(atPath:binary)) else { throw ServerFailure(message:"Choose an installed Sushi CLI, or use sushi from your terminal’s PATH.") }
        var command = quote(binary.isEmpty ? "sushi" : binary) + " launch " + client + " --url " + quote(endpoint)
        if let model, !model.isEmpty { command += " --model " + quote(model) }
        return command
    }
    private static func quote(_ word:String) -> String { "'" + word.replacingOccurrences(of:"'",with:"'\\''") + "'" }
}

struct DesktopControllerState {
    let paused:Bool
    let state:String
    let activeRequests:Int
    let currentModel:String
    let models:[String:String]
    init(_ object:[String:Any]) throws {
        guard let desktop = object["desktop"] as? [String:Any],
              let paused = desktop["paused"] as? Bool,
              let controller = object["idle_controller"] as? [String:Any],
              let state = controller["state"] as? String,
              let active = controller["active_requests"] as? Int, active >= 0 else {
            throw ServerFailure(message:"This server does not advertise the supported desktop controller API.")
        }
        self.paused = paused; self.state = state; activeRequests = active
        currentModel = object["model"] as? String ?? ""
        var catalog:[String:String] = [:]
        for row in desktop["models"] as? [[String:Any]] ?? [] {
            if let id = row["id"] as? String, let path = row["path"] as? String, path.hasPrefix("/") { catalog[id] = path }
        }
        models = catalog
    }
}

extension DesktopControllerState {
    func modelActions(_ action:String,model:String?) throws -> [[String:Any]] {
        guard activeRequests == 0, !["loading","stopping"].contains(state) else { throw ServerFailure(message:"Wait for the controller’s active request or model operation to finish.") }
        switch action {
        case "load-model":
            guard let model, !model.isEmpty else { throw ServerFailure(message:"Choose a model to load.") }
            if model == currentModel { return [["action":"load"]] }
            guard let path = models[model] ?? (model.hasPrefix("/") ? model : nil) else { throw ServerFailure(message:"This model is not in the controller’s catalog. Refresh the library before loading it.") }
            return [["action":"select","path":path],["action":"load"]]
        case "unload-model":
            guard model == nil || model == currentModel else { throw ServerFailure(message:"Only the controller’s current model can be unloaded.") }
            return [["action":"unload"]]
        default: throw ServerFailure(message:"This controller does not expose model rescanning. Its catalog refreshes automatically.")
        }
    }
}
struct ModelActionPlan {
    struct Request { let path:String; let body:[String:Any] }
    let requests:[Request]
    let usesController:Bool
    static func make(_ action:String,model:String?,controller:DesktopControllerState?) throws -> ModelActionPlan {
        if let controller {
            return ModelActionPlan(requests:try controller.modelActions(action,model:model).map { Request(path:"control/action",body:$0) },usesController:true)
        }
        guard ["load-model","unload-model","models/rescan"].contains(action) else { throw ServerFailure(message:"Unknown model action.") }
        var body:[String:Any] = [:]
        if let model { body["model"] = model }
        if action == "load-model" { body["default"] = true }
        return ModelActionPlan(requests:[Request(path:"v1/"+action,body:body)],usesController:false)
    }
    func execute(_ client:Connection) async throws {
        var selected = false
        do {
            for request in requests {
                _ = try await client.fetch(request.path,body:request.body)
                if request.body["action"] as? String == "select" { selected = true }
            }
        } catch {
            if selected { throw ServerFailure(message:"The controller selected the model, but loading failed. It remains selected; retry Load when ready. \(error.localizedDescription)") }
            throw error
        }
    }
}
