import Foundation

@main struct ModelSettingsTests {
    static func main() throws {
        var presets = try ModelPresetLibrary(data:nil)
        var tuning = ModelOverrides(); tuning.context = "65536"; tuning.kv = "8"
        try presets.save(name:"  Long context  ",values:tuning)
        precondition(presets.presets.count == 1 && presets.presets[0].name == "Long context")
        let presetID = presets.presets[0].id
        tuning.context = "131072"
        try presets.save(name:"LONG CONTEXT",values:tuning)
        precondition(presets.presets.count == 1 && presets.presets[0].id == presetID)
        let restoredPresets = try ModelPresetLibrary(data:presets.encoded())
        precondition(restoredPresets.presets[0].values.context == "131072")
        for invalidName in ["",String(repeating:"x",count:81),"line\nbreak"] {
            do { try presets.save(name:invalidName,values:tuning); fatalError("Accepted invalid preset name") } catch {}
        }
        for index in 1...19 { try presets.save(name:"Preset \(index)",values:tuning) }
        do { try presets.save(name:"Overflow",values:tuning); fatalError("Accepted unbounded presets") } catch {}
        let removed = presets.remove(presetID)
        precondition(removed != nil && presets.presets.count == 19)
        do { _ = try ModelPresetLibrary(data:Data("invalid".utf8)); fatalError("Accepted corrupt presets") } catch {}
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:dir) }
        try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true)
        let file = dir.appendingPathComponent("settings.json")
        let initial = Data(#"{"/models/a/":{"ctx_size":65536,"kv_quant":8,"mtp":false,"custom":17},"/other":{"vision":true}}"#.utf8)
        try initial.write(to:file)
        let document = try ModelSettingsDocument(url:file)
        var overrides = try document.overrides(for:"/models/a")
        precondition(overrides.context == "65536" && overrides.kv == "8" && overrides.mtp == "Off")
        overrides.context = "131072"; overrides.mtp = "On"
        overrides.acceptance = "exact"; overrides.greedyTail = "Off"; overrides.ssdBudget = "16"
        let backup = try document.save(overrides,for:"/models/a")!
        let backupData = try Data(contentsOf:backup)
        precondition(backupData == initial)
        let object = try JSONSerialization.jsonObject(with:Data(contentsOf:file)) as! [String:Any]
        precondition((object["/models/a/"] as! [String:Any])["custom"] as! Int == 17)
        precondition((object["/other"] as! [String:Any])["vision"] as! Bool)
        do { try document.save(overrides,for:"/models/a"); fatalError("Concurrent edit overwritten") } catch {}
        let reopened = try ModelSettingsDocument(url:file)
        let restored = try reopened.overrides(for:"/models/a")
        precondition(restored.mtp == "On" && restored.acceptance == "exact" && restored.greedyTail == "Off" && restored.ssdBudget == "16")
        overrides.acceptance = "invented"
        do { _ = try overrides.values(); fatalError("Unknown acceptance accepted") } catch {}
        overrides.acceptance = "Default"; overrides.ssdBudget = "0"
        do { _ = try overrides.values(); fatalError("Zero SSD override accepted") } catch {}
        overrides.ssdBudget = ""
        overrides.thinkPenalty = "nan"
        do { _ = try overrides.values(); fatalError("Invalid penalty accepted") } catch {}
        overrides.thinkPenalty = "21"
        do { _ = try overrides.values(); fatalError("Out of range accepted") } catch {}
        overrides.thinkPenalty = ""; overrides.context = "4294967296"
        do { _ = try overrides.values(); fatalError("Context overflow accepted") } catch {}
        try Data(#"{"/models/a":17}"#.utf8).write(to:file)
        let malformedRow = try ModelSettingsDocument(url:file)
        do { try malformedRow.save(ModelOverrides(),for:"/models/a"); fatalError("Non-object model settings overwritten") } catch {}
        try Data("[]".utf8).write(to:file)
        do { _ = try ModelSettingsDocument(url:file); fatalError("Malformed root accepted") } catch {}
        print("Model settings passed: path matching, preservation, backup, conflict, validation")
    }
}
