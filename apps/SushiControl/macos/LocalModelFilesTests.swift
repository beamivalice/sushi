import Foundation

@main struct LocalModelFilesTests {
    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at:root) }
        let model = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at:model,withIntermediateDirectories:true)
        try Data(#"{"model_type":"fixture"}"#.utf8).write(to:model.appendingPathComponent("config.json"))
        try Data(repeating:1,count:32).write(to:model.appendingPathComponent("model.safetensors"))
        let selected = try LocalModelFile.inspect(model)
        precondition(selected.architecture == "fixture" && selected.fileCount == 2 && selected.bytes >= 32)
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at:alias,withDestinationURL:model)
        do { _ = try LocalModelFile.inspect(alias); fatalError("Accepted a model alias") } catch {}
        let invalid = root.appendingPathComponent("bad.gguf")
        try Data("wrong".utf8).write(to:invalid)
        do { _ = try LocalModelFile.inspect(invalid); fatalError("Accepted invalid GGUF header") } catch {}
        let gguf = root.appendingPathComponent("model.gguf")
        try Data("GGUFfixture".utf8).write(to:gguf)
        let ggufSelection = try LocalModelFile.inspect(gguf)
        precondition(ggufSelection.architecture == "GGUF")
        do { _ = try LocalModelFile.inspect(FileManager.default.homeDirectoryForCurrentUser); fatalError("Accepted home folder") } catch {}
        let moved = root.appendingPathComponent("original")
        try FileManager.default.moveItem(at:model,to:moved)
        try FileManager.default.createDirectory(at:model,withIntermediateDirectories:true)
        do { try selected.verifyIdentity(); fatalError("Accepted replaced model path") } catch {}
        let item = try LocalModelFile.inspect(gguf)
        let receipt = try item.moveToTrash()
        precondition(!FileManager.default.fileExists(atPath:gguf.path) && FileManager.default.fileExists(atPath:receipt.trashedURL.path))
        try Data("occupied".utf8).write(to:gguf)
        do { try receipt.restore(); fatalError("Overwrote occupied restore path") } catch {}
        let occupied = try Data(contentsOf:gguf)
        precondition(occupied == Data("occupied".utf8))
        try FileManager.default.removeItem(at:gguf)
        try receipt.restore()
        precondition(FileManager.default.fileExists(atPath:gguf.path))
        print("Local model files passed: recognition, protected roots, aliases, replacement refusal, real Trash/restore round trip")
    }
}
