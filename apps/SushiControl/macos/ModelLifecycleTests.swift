import Foundation

@main struct ModelLifecycleTests {
    static func main() async throws {
        let binary = CommandLine.arguments[1], model = CommandLine.arguments[2], port = CommandLine.arguments[3]
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("sushi-model-check-" + UUID().uuidString)
        let root = directory.appendingPathComponent("models")
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let server = ManagedServer(label:"org.sushi.model.test.\(UUID().uuidString.lowercased())",directory:directory)
        if CommandLine.arguments.count > 4 {
            let record:[String:String] = ["service":server.service,"directory":directory.path]
            try JSONSerialization.data(withJSONObject:record).write(to:URL(fileURLWithPath:CommandLine.arguments[4]),options:.atomic)
        }
        let client = Connection(endpoint:"http://127.0.0.1:\(port)")
        try server.install(binary:binary,arguments:["serve","--host","127.0.0.1","--port",port,"--model-dir",root.path,"--metrics","--ctx-size","8192","--max-resident-mem","60129542144","--prefix-cache-mem","268435456","--prefill-chunk","512"])
        do {
            try await server.start()
            let deadline = Date().addingTimeInterval(30)
            while Date() < deadline {
                if (try? await client.fetch("health")) != nil { break }
                try await Task.sleep(nanoseconds:200_000_000)
            }
            let id = URL(fileURLWithPath:model).lastPathComponent
            for round in 1...2 {
                _ = try await client.fetch("v1/load-model",body:["model":model,"default":true])
                let rows = ModelEntry.parse(try await client.fetch("v1/models"))
                guard rows.contains(where:{$0.id == id && $0.loaded}) else { throw ServerFailure(message:"Loaded model absent from registry") }
                let body:[String:Any] = ["model":id,"messages":[["role":"user","content":"Reply with only the number 42."]],"enable_thinking":false,"temperature":0,"max_tokens":64]
                let answer = try await client.fetch("v1/chat/completions",body:body)
                let choices = answer["choices"] as? [[String:Any]] ?? []
                let content = (choices.first?["message"] as? [String:Any])?["content"] as? String ?? ""
                guard content.trimmingCharacters(in:.whitespacesAndNewlines).contains("42") else { throw ServerFailure(message:"Inference did not return the expected answer") }
                _ = try await client.fetch("v1/unload-model",body:["model":id])
                let unloaded = ModelEntry.parse(try await client.fetch("v1/models"))
                guard unloaded.contains(where:{$0.id == id && !$0.loaded}) else { throw ServerFailure(message:"Unloaded model absent from registry") }
                print("Model lifecycle round \(round) passed: load, registry, inference, unload")
                fflush(stdout)
            }
            try await server.stop()
            try FileManager.default.removeItem(at:directory)
        } catch {
            if await server.registered() { try? await server.stop() }
            print("Failure logs: \(server.logURL.path)")
            throw error
        }
    }
}
