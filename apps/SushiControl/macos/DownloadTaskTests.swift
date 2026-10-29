import Foundation

@main struct DownloadTaskTests {
    @MainActor static func main() async throws {
        let tool = ToolTask(), binary = CommandLine.arguments[1]
        let home = URL(fileURLWithPath:ProcessInfo.processInfo.environment["HOME"]!)
        let model = home.appendingPathComponent(".sushi/models/beamster/app-download-fixture")
        let partial = model.appendingPathComponent("model.safetensors.partial")
        try tool.run(binary:binary,arguments:["pull","beamster/app-download-fixture"],title:"Fixture download",timeout:20)
        let deadline = Date().addingTimeInterval(10)
        while Date() < deadline && tool.running {
            if let size = (try? FileManager.default.attributesOfItem(atPath:partial.path)[.size]) as? NSNumber, size.intValue > 0 { break }
            try await Task.sleep(nanoseconds:50_000_000)
        }
        guard FileManager.default.fileExists(atPath:partial.path) else { throw ServerFailure(message:"Download did not produce a partial file") }
        tool.cancel(); try await finish(tool)
        precondition(tool.status == "Cancelled")
        precondition(tool.history.first?.downloadModel == "beamster/app-download-fixture" && tool.history.first?.status == "Cancelled")
        precondition(FileManager.default.fileExists(atPath:model.appendingPathComponent(".pull-incomplete").path))
        try tool.run(binary:binary,arguments:["pull","beamster/app-download-fixture"],title:"Resume fixture",timeout:20)
        try await finish(tool)
        guard tool.status == "Completed" else { throw ServerFailure(message:tool.output) }
        precondition(!FileManager.default.fileExists(atPath:model.appendingPathComponent(".pull-incomplete").path))
        precondition(!FileManager.default.fileExists(atPath:partial.path))
        let data = try Data(contentsOf:model.appendingPathComponent("model.safetensors"))
        precondition(data.count == 2*1024*1024 && data.allSatisfy{$0 == 42})
        precondition(tool.history.count == 2 && tool.history.first?.status == "Completed")
        precondition(tool.downloadProgress == nil)
        print("Download integration passed: partial transfer, cancellation, resume, complete bytes")
    }
    @MainActor static func finish(_ tool:ToolTask) async throws {
        let deadline = Date().addingTimeInterval(25)
        while tool.running {
            if Date() > deadline { tool.cancel(); throw ServerFailure(message:"Download check exceeded its deadline") }
            try await Task.sleep(nanoseconds:100_000_000)
        }
    }
}
