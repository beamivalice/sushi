import Foundation

@main struct ControllerLifecycleTests {
    static func main() async throws {
        guard [4,5].contains(CommandLine.arguments.count) else { throw ServerFailure(message:"Usage: controller-lifecycle-tests endpoint control-token-file result-file [inference-token-file]. Run only against an idle server while holding the GPU lock.") }
        let tokenURL = URL(fileURLWithPath:CommandLine.arguments[2])
        let token = try String(contentsOf:tokenURL,encoding:.utf8).trimmingCharacters(in:.whitespacesAndNewlines)
        let control = Connection(endpoint:CommandLine.arguments[1],apiKey:token)
        let inferenceKey = CommandLine.arguments.count == 5 ? try String(contentsOfFile:CommandLine.arguments[4],encoding:.utf8).trimmingCharacters(in:.whitespacesAndNewlines) : ""
        let inference = Connection(endpoint:CommandLine.arguments[1],apiKey:inferenceKey)
        let original = try DesktopControllerState(await control.fetch("control/status"))
        guard !token.isEmpty, original.state == "ready", original.activeRequests == 0, !original.paused, !original.currentModel.isEmpty else {
            throw ServerFailure(message:"The live controller must be ready, unpaused and idle before testing.")
        }
        do {
            try await ModelActionPlan.make("unload-model",model:original.currentModel,controller:original).execute(control)
            let asleep = try DesktopControllerState(await control.fetch("control/status"))
            guard asleep.state == "sleeping", asleep.activeRequests == 0, asleep.currentModel == original.currentModel else { throw ServerFailure(message:"Controller did not unload the current model.") }
            try await ModelActionPlan.make("load-model",model:original.currentModel,controller:asleep).execute(control)
            let restored = try DesktopControllerState(await control.fetch("control/status"))
            guard restored.state == "ready", restored.currentModel == original.currentModel, !restored.paused else { throw ServerFailure(message:"Controller did not restore the original model.") }
            let models = ModelEntry.parse(try await inference.fetch("v1/models"))
            guard models.contains(where:{$0.id == original.currentModel && $0.loaded}) else { throw ServerFailure(message:"The inference endpoint does not report the restored model as loaded.") }
            let response = try await inference.fetch("v1/chat/completions",body:["model":original.currentModel,"messages":[["role":"user","content":"Write a list of simple colours."]],"max_tokens":16,"temperature":0,"stream":false,"chat_template_kwargs":["enable_thinking":false]],timeout:90)
            guard let usage = response["usage"] as? [String:Any], let generated = usage["completion_tokens"] as? Int, generated > 0 else { throw ServerFailure(message:"The restored model did not produce completion tokens.") }
            let result:[String:Any] = ["model":original.currentModel,"unloaded":true,"reloaded":true,"inference":true,"completion_tokens":generated,"paused":restored.paused]
            try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:CommandLine.arguments[3]),options:.atomic)
            print("Controller lifecycle passed: current-model unload, reload, discovery and inference; original selection preserved")
        } catch {
            if let state = try? await control.fetch("control/status"), let current = try? DesktopControllerState(state), current.state == "sleeping", current.activeRequests == 0, current.currentModel == original.currentModel {
                _ = try? await control.fetch("control/action",body:["action":"load"])
            }
            throw error
        }
    }
}
