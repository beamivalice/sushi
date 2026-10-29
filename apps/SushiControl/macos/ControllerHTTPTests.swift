import Foundation

@main struct ControllerHTTPTests {
    static func main() async throws {
        let endpoint = CommandLine.arguments[1]
        let client = Connection(endpoint:endpoint,apiKey:"control-fixture")
        let initial = try DesktopControllerState(await client.fetch("control/status"))
        precondition(!initial.paused && initial.activeRequests == 1)
        _ = try await client.fetch("control/action",body:["action":"pause"])
        let paused = try DesktopControllerState(await client.fetch("control/status"))
        precondition(paused.paused && paused.activeRequests == 1)
        do { _ = try await client.fetch("control/action",body:["action":"unload"]); fatalError("Accepted busy unload") }
        catch { precondition((error as? ServerFailure)?.status == 409) }
        _ = try await client.fetch("control/action",body:["action":"resume"])
        let resumed = try DesktopControllerState(await client.fetch("control/status"))
        precondition(!resumed.paused)
        let current = try DesktopControllerState(await client.fetch("control/status"))
        try await ModelActionPlan.make("load-model",model:"current",controller:current).execute(client)
        try await ModelActionPlan.make("load-model",model:"other",controller:current).execute(client)
        let changed = try DesktopControllerState(await client.fetch("control/status"))
        precondition(changed.currentModel == "other")
        do { try await ModelActionPlan.make("load-model",model:"broken",controller:changed).execute(client); fatalError("Ignored load failure") }
        catch { precondition(error.localizedDescription.contains("remains selected")) }
        let failed = try DesktopControllerState(await client.fetch("control/status"))
        precondition(failed.currentModel == "broken")
        do { _ = try await Connection(endpoint:endpoint,apiKey:"inference-key").fetch("control/status"); fatalError("Accepted inference key") }
        catch { precondition((error as? ServerFailure)?.status == 401) }
        do { _ = try await client.fetch("v1/control/status"); fatalError("Accepted unsupported endpoint") }
        catch { precondition((error as? ServerFailure)?.status == 404) }
        print("Controller HTTP checks passed: separate auth, pause/resume, admitted request preserved, busy refusal, unsupported route, model selection/load, partial failure")
    }
}
