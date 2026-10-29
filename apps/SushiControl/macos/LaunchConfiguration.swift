import Foundation

struct LaunchConfiguration {
    static func decode(_ text:String) throws -> [String] {
        guard let data = text.data(using:.utf8), let arguments = (try? JSONSerialization.jsonObject(with:data)) as? [String] else {
            throw ServerFailure(message:"Launch arguments must be a JSON array of strings.")
        }
        return arguments
    }
    static func value(_ flag:String,in arguments:[String]) throws -> String? {
        guard let index = arguments.lastIndex(of:flag) else { return nil }
        guard arguments.indices.contains(index+1), !arguments[index+1].hasPrefix("--") else { throw ServerFailure(message:"Enter a value after \(flag).") }
        return arguments[index+1]
    }
    static func validateServer(_ arguments:[String],endpoint:String) throws {
        _ = try Connection(endpoint:endpoint).request("health")
        guard arguments.first == "serve" || arguments.contains("--serve") else { throw ServerFailure(message:"Server arguments must start with serve or include --serve.") }
        let host = try value("--host",in:arguments) ?? "127.0.0.1"
        guard ["127.0.0.1","localhost","::1"].contains(host) else { throw ServerFailure(message:"Use a loopback host for an app-managed server.") }
        let portText = try value("--port",in:arguments) ?? "12345"
        guard let port = Int(portText), (1...65535).contains(port) else { throw ServerFailure(message:"Enter a server port from 1 to 65535.") }
        let url = URL(string:endpoint)!
        guard (url.port ?? 80) == port else { throw ServerFailure(message:"The server port and connection URL must match.") }
        let clientHost = url.host ?? ""
        guard clientHost == "localhost" || host == "localhost" || (host == "::1" ? clientHost == "[::1]" : clientHost == "127.0.0.1") else { throw ServerFailure(message:"The connection URL and server host must use the same loopback address.") }
    }
}
extension LaunchSettings {
    static let valueFlags = ["--host","--port","--model-dir","--ctx-size","--max-resident-mem","--idle-evict-secs","--kv-quant","--prefix-cache-mem","--prefix-cache-disk","--prefill-chunk","--drafter"]
    static func read(_ text:String) throws -> LaunchSettings {
        let args = try LaunchConfiguration.decode(text)
        var settings = LaunchSettings()
        settings.metrics = args.contains("--metrics")
        if let port = try LaunchConfiguration.value("--port",in:args) { guard let value = Int(port) else { throw ServerFailure(message:"Invalid server port.") }; settings.port = value }
        settings.modelDirectory = try LaunchConfiguration.value("--model-dir",in:args) ?? ""
        settings.context = try LaunchConfiguration.value("--ctx-size",in:args) ?? ""
        settings.prefillChunk = try LaunchConfiguration.value("--prefill-chunk",in:args) ?? ""
        settings.drafter = try LaunchConfiguration.value("--drafter",in:args) ?? ""
        if let kv = try LaunchConfiguration.value("--kv-quant",in:args) {
            guard let width = Int(kv), [4,6,8,16].contains(width) else { throw ServerFailure(message:"Use advanced arguments for this KV cache format.") }; settings.kvBits = width
        }
        if let seconds = try LaunchConfiguration.value("--idle-evict-secs",in:args) {
            guard let value = Int(seconds), value >= 0, value % 60 == 0 else { throw ServerFailure(message:"Use advanced arguments for idle timeouts shorter than a whole minute.") }; settings.idleMinutes = value/60
        }
        settings.memoryGiB = try gib(LaunchConfiguration.value("--max-resident-mem",in:args))
        settings.ramCacheGiB = try gib(LaunchConfiguration.value("--prefix-cache-mem",in:args))
        settings.diskCacheGiB = try gib(LaunchConfiguration.value("--prefix-cache-disk",in:args))
        return settings
    }
    func mergedArguments(_ text:String) throws -> [String] {
        let original = try LaunchConfiguration.decode(text)
        var remainder:[String] = [], index = original.first == "serve" ? 1 : 0
        while index < original.count {
            let argument = original[index]
            if Self.valueFlags.contains(argument) {
                guard original.indices.contains(index+1), !original[index+1].hasPrefix("--") else { throw ServerFailure(message:"Enter a value after \(argument).") }
                index += 2
            } else if ["--metrics","--serve"].contains(argument) { index += 1 }
            else { remainder.append(argument); index += 1 }
        }
        return try arguments() + remainder
    }
    private static func gib(_ input:String?) throws -> String {
        guard let input, !["auto","off"].contains(input.lowercased()) else { return input?.lowercased() == "off" ? "0" : "" }
        var value = input.uppercased(), multiplier = 1.0
        for (suffix,scale) in [("GIB",1073741824.0),("MIB",1048576.0),("KIB",1024.0),("GB",1073741824.0),("MB",1048576.0),("KB",1024.0),("G",1073741824.0),("M",1048576.0),("K",1024.0)] {
            if value.hasSuffix(suffix) { value.removeLast(suffix.count); multiplier = scale; break }
        }
        guard let bytes = Double(value), bytes.isFinite, bytes >= 0 else { throw ServerFailure(message:"Use advanced arguments for this memory budget.") }
        return String(bytes * multiplier / 1073741824)
    }
}
