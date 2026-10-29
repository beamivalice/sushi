import Foundation
import CoreFoundation

struct ModelOverrides:Codable {
    var context = ""
    var kv = "Default"
    var mtp = "Default"
    var vision = "Default"
    var preserveThinking = "Default"
    var thinkPenalty = ""
    var acceptance = "Default"
    var greedyTail = "Default"
    var ssdBudget = ""
    static let keys = ["ctx_size","kv_quant","mtp","vision","preserve_thinking","think_penalty","mtp_acceptance","mtp_greedy_tail","ssd_budget_gb"]
    func values() throws -> [String:Any] {
        var result:[String:Any] = [:]
        if !context.isEmpty {
            guard let value = UInt32(context), value > 0 else { throw ServerFailure(message:"Context must be a positive token count, or empty for automatic.") }
            result["ctx_size"] = value
        }
        guard ["Default","4","6","8","16"].contains(kv) else { throw ServerFailure(message:"Choose a supported KV width.") }
        if kv != "Default" { result["kv_quant"] = kv }
        for (key,value) in [("mtp",mtp),("vision",vision),("preserve_thinking",preserveThinking),("mtp_greedy_tail",greedyTail)] {
            guard ["Default","On","Off"].contains(value) else { throw ServerFailure(message:"Choose Default, On or Off.") }
            if value != "Default" { result[key] = value == "On" }
        }
        if !thinkPenalty.isEmpty {
            guard let value = Double(thinkPenalty), value.isFinite, (0...20).contains(value) else { throw ServerFailure(message:"Thinking penalty must be between 0 and 20, or empty for default.") }
            result["think_penalty"] = value
        }
        guard ["Default","exact","typical","tokenv3"].contains(acceptance) else { throw ServerFailure(message:"Choose a supported MTP acceptance mode.") }
        if acceptance != "Default" { result["mtp_acceptance"] = acceptance }
        if !ssdBudget.isEmpty {
            guard let value = UInt32(ssdBudget), value > 0 else { throw ServerFailure(message:"SSD budget must be a positive whole number of GB, or empty for default.") }
            result["ssd_budget_gb"] = value
        }
        return result
    }
    init() {}
    init(_ row:[String:Any]) throws {
        func number(_ key:String) throws -> NSNumber? {
            guard let value = row[key] else { return nil }
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { throw ServerFailure(message:"Existing \(key) has an unsupported value. Inspect the settings file before editing.") }
            return number
        }
        if let value = try number("ctx_size") { context = value.stringValue }
        if let value = row["kv_quant"] {
            if let string = value as? String { kv = string }
            else if let number = try number("kv_quant") { kv = number.stringValue }
            if ["off","0"].contains(kv) { kv = "16" }
        }
        for key in ["mtp","vision","preserve_thinking","mtp_greedy_tail"] {
            guard let value = row[key] else { continue }
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { throw ServerFailure(message:"Existing \(key) must be a boolean.") }
            let choice = number.boolValue ? "On" : "Off"
            if key == "mtp" { mtp = choice }
            if key == "mtp_greedy_tail" { greedyTail = choice }
            if key == "vision" { vision = choice }
            if key == "preserve_thinking" { preserveThinking = choice }
        }
        if let value = try number("think_penalty") { thinkPenalty = value.stringValue }
        if let value = row["mtp_acceptance"] {
            guard let name = value as? String else { throw ServerFailure(message:"Existing MTP acceptance must name a mode.") }
            acceptance = name
        }
        if let value = try number("ssd_budget_gb") { ssdBudget = value.stringValue }
        _ = try values()
    }
}

struct ModelSettingsDocument {
    let url:URL
    private let original:Data?
    private let root:[String:Any]
    init(url:URL) throws {
        self.url = url
        if FileManager.default.fileExists(atPath:url.path) {
            let size = try FileManager.default.attributesOfItem(atPath:url.path)[.size] as? NSNumber
            guard let size, size.intValue <= 1024*1024 else { throw ServerFailure(message:"Settings file exceeds 1 MiB.") }
            let data = try Data(contentsOf:url)
            guard let root = try JSONSerialization.jsonObject(with:data) as? [String:Any] else { throw ServerFailure(message:"Settings must contain a JSON object.") }
            self.original = data; self.root = root
        } else { original = nil; root = [:] }
    }
    private func key(for path:String) throws -> String {
        guard path.hasPrefix("/"), path != "/", !path.contains("\0") else { throw ServerFailure(message:"Choose an absolute model folder or GGUF file.") }
        func trim(_ value:String) -> String { var result = value; while result.count > 1 && result.hasSuffix("/") { result.removeLast() }; return result }
        let normalized = trim(path)
        let matches = root.keys.filter { trim($0) == normalized }
        guard matches.count <= 1 else { throw ServerFailure(message:"Multiple settings entries name this model path. Resolve them before editing.") }
        return matches.first ?? normalized
    }
    func overrides(for path:String) throws -> ModelOverrides {
        let key = try key(for:path)
        guard let value = root[key] else { return ModelOverrides() }
        guard let row = value as? [String:Any] else { throw ServerFailure(message:"This model’s settings must be a JSON object.") }
        return try ModelOverrides(row)
    }
    @discardableResult func save(_ overrides:ModelOverrides,for path:String) throws -> URL? {
        let key = try key(for:path), values = try overrides.values()
        let current = FileManager.default.fileExists(atPath:url.path) ? try Data(contentsOf:url) : nil
        guard current == original else { throw ServerFailure(message:"Settings changed outside this editor. Reopen the model settings before saving.") }
        guard root[key] == nil || root[key] is [String:Any] else { throw ServerFailure(message:"This model’s settings must be a JSON object.") }
        var updated = root
        var row = root[key] as? [String:Any] ?? [:]
        for key in ModelOverrides.keys { row.removeValue(forKey:key) }
        for (key,value) in values { row[key] = value }
        if row.isEmpty { updated.removeValue(forKey:key) } else { updated[key] = row }
        let data = try JSONSerialization.data(withJSONObject:updated,options:[.prettyPrinted,.sortedKeys,.withoutEscapingSlashes])
        try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        var backup:URL?
        if let original {
            let destination = url.deletingLastPathComponent().appendingPathComponent("model-settings-backup-\(UUID().uuidString).json")
            try original.write(to:destination,options:.withoutOverwriting)
            try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:destination.path)
            backup = destination
        }
        try data.write(to:url,options:.atomic)
        try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:url.path)
        return backup
    }
}

struct ModelPreset:Codable,Identifiable {
    let id:UUID
    var name:String
    var values:ModelOverrides
}
struct ModelPresetLibrary {
    private struct Archive:Codable { let version:Int; let presets:[ModelPreset] }
    private(set) var presets:[ModelPreset] = []
    init(data:Data?) throws {
        guard let data else { return }
        guard data.count <= 65536 else { throw ServerFailure(message:"Saved presets exceed 64 KiB.") }
        let archive = try JSONDecoder().decode(Archive.self,from:data)
        guard archive.version == 1, archive.presets.count <= 20,
              Set(archive.presets.map(\.id)).count == archive.presets.count else { throw ServerFailure(message:"Unsupported preset archive.") }
        var names = Set<String>()
        for preset in archive.presets {
            let name = try Self.validName(preset.name)
            guard names.insert(name.lowercased()).inserted else { throw ServerFailure(message:"Saved presets contain duplicate names.") }
            _ = try preset.values.values()
        }
        presets = archive.presets
    }
    private static func validName(_ input:String) throws -> String {
        let name = input.trimmingCharacters(in:.whitespacesAndNewlines)
        guard !name.isEmpty, name.utf8.count <= 80, !name.unicodeScalars.contains(where:{CharacterSet.controlCharacters.contains($0)}) else {
            throw ServerFailure(message:"Keep the preset name short and on one line.")
        }
        return name
    }
    mutating func save(name:String,values:ModelOverrides) throws {
        let name = try Self.validName(name); _ = try values.values()
        if let index = presets.firstIndex(where:{$0.name.caseInsensitiveCompare(name) == .orderedSame}) {
            presets[index].name = name; presets[index].values = values
        } else {
            guard presets.count < 20 else { throw ServerFailure(message:"Keep up to 20 presets. Remove one before adding another.") }
            presets.append(ModelPreset(id:UUID(),name:name,values:values))
        }
        presets.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }
    @discardableResult mutating func remove(_ id:UUID) -> ModelPreset? {
        guard let index = presets.firstIndex(where:{$0.id == id}) else { return nil }
        return presets.remove(at:index)
    }
    func encoded() throws -> Data { try JSONEncoder().encode(Archive(version:1,presets:presets)) }
}
