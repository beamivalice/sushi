import Foundation

@main struct ClientTests {
    static func tryPlan(_ state:DesktopControllerState,_ action:String,_ model:String) -> [String] {
        try! state.modelActions(action,model:model).map { $0["action"] as! String }
    }
    static func main() throws {
        let desktop = try DesktopControllerState(["desktop":["paused":true,"models":[["id":"test","path":"/tmp/test"]]],"idle_controller":["state":"sleeping","active_requests":0]])
        precondition(desktop.paused && desktop.state == "sleeping" && desktop.models["test"] == "/tmp/test")
        for invalid:[String:Any] in [[:],["desktop":["paused":false]],["desktop":["paused":false],"idle_controller":["state":"ready","active_requests":-1]]] {
            do { _ = try DesktopControllerState(invalid); fatalError("Accepted invalid controller response") } catch {}
        }
        let controller = try DesktopControllerState(["model":"current","desktop":["paused":false,"models":[["id":"current","path":"/models/current"],["id":"other","path":"/models/other"]]],"idle_controller":["state":"sleeping","active_requests":0]])
        precondition(tryPlan(controller,"load-model","current") == ["load"])
        precondition(tryPlan(controller,"load-model","other") == ["select","load"])
        precondition(tryPlan(controller,"unload-model","current") == ["unload"])
        for pair in [("load-model","missing"),("unload-model","other")] {
            do { _ = try controller.modelActions(pair.0,model:pair.1); fatalError("Accepted unknown controller model") } catch {}
        }
        let stockPlan = try ModelActionPlan.make("load-model",model:"stock",controller:nil)
        precondition(stockPlan.requests.first?.path == "v1/load-model")
        precondition(stockPlan.requests.first?.body["default"] as? Bool == true)
        let config = Connection(endpoint: "http://127.0.0.1:12345", apiKey: "test")
        let request = try config.request("v1/load-model", body: ["model": "/tmp/model", "default": true])
        precondition(request.url!.path == "/v1/load-model")
        precondition(request.value(forHTTPHeaderField: "Authorization") == "Bearer test")
        let slashURL = try Connection(endpoint:"http://127.0.0.1:12345/").request("v1").url!
        precondition(slashURL.path == "/v1")
        let body = try JSONSerialization.jsonObject(with: request.httpBody!) as! [String:Any]
        precondition(body["model"] as? String == "/tmp/model")
        for endpoint in ["http://example.com:12345", "http://127.0.0.1:12345/v1", "http://user:secret@localhost:12345", "file:///tmp/server"] {
            do { _ = try Connection(endpoint:endpoint).request("props"); fatalError("Accepted unsafe endpoint") } catch {}
        }
        let rows = try ModelEntry.decode(Data(#"{"data":[{"id":"qwen","loaded":false,"state":"unloaded"},{"id":"mimo","loaded":true,"state":"ready","meta":{"architecture":"mimo_v2"}}]}"#.utf8))
        precondition(rows.count == 2 && !rows[0].loaded && rows[1].loaded)
        precondition(rows[1].architecture == "mimo_v2")
        precondition(ModelEntry.library(rows,scope:"Favorites",favorites:["qwen"]).map(\.id) == ["qwen"])
        precondition(ModelEntry.library(rows,scope:"Menu",favorites:["qwen"]).map(\.id) == ["mimo","qwen"])
        precondition(ModelEntry.library(rows,query:"  MIMO  ").map(\.id) == ["mimo"])
        precondition(ModelEntry.library(rows,scope:"Favorites").isEmpty)
        var ranked = rows
        ranked[0].contextTokens = 131072; ranked[0].residentBytes = 8*1073741824
        ranked[1].contextTokens = 262144; ranked[1].residentBytes = 2*1073741824
        precondition(ModelEntry.library(ranked,sort:"Context").map(\.id) == ["mimo","qwen"])
        precondition(ModelEntry.library(ranked,sort:"Memory").map(\.id) == ["qwen","mimo"])
        precondition(ModelEntry.selection("hidden",in:ModelEntry.library(rows,scope:"Menu")).isEmpty == false)
        let recommendations = ModelEntry.parse(["data":[["id":"recommended","meta":["gen_temperature":0.6,"gen_top_p":0.95,"gen_top_k":20]]]])[0]
        precondition(recommendations.recommendedTemperature == 0.6 && recommendations.recommendedTopP == 0.95 && recommendations.recommendedTopK == 20)
        let invalidRecommendations = ModelEntry.parse(["data":[["id":"bad","meta":["gen_temperature":true,"gen_top_p":1.5,"gen_top_k":2.5]]]])[0]
        precondition(invalidRecommendations.recommendedTemperature == nil && invalidRecommendations.recommendedTopP == nil && invalidRecommendations.recommendedTopK == nil)
        let detail = ModelEntry.parse(["data":[["id":"detailed","loaded":true,"context_length":131072,"max_model_len":262144,"bytes_on_disk":1073741824.0,"capabilities":["chat","vision","tool_use"],"input_modalities":["text","image"],"reasoning_efforts":["low","high"],"default_reasoning_effort":"high","meta":["kv_quant":"affine8","drafter_loaded":true,"mtp_loaded":false]]]])[0]
        precondition(detail.capabilities.contains("vision") && detail.inputModalities == ["text","image"])
        precondition(detail.maximumContext == 262144 && detail.contextTokens == 131072 && detail.diskBytes == 1073741824)
        precondition(detail.defaultReasoningEffort == "high" && detail.reasoningEfforts == ["low","high"])
        precondition(detail.drafterLoaded == true && detail.mtpLoaded == false && detail.kvFormat == "affine8")
        precondition(rows[0].capabilities.isEmpty && rows[0].drafterLoaded == nil)
        let error = ServerFailure(status:401, data:Data(#"{"error":{"message":"Invalid API key"}}"#.utf8))
        precondition(error.status == 401 && error.localizedDescription.contains("Invalid API key"))
        precondition(ModelEntry.selection("qwen",in:rows) == "qwen")
        precondition(ModelEntry.selection("removed",in:rows) == "mimo")
        precondition(ModelEntry.selection("",in:[]) == "")
        var launch = LaunchSettings()
        launch.modelDirectory = "/tmp/models with spaces"
        launch.context = "131072"; launch.memoryGiB = "48"; launch.idleMinutes = 15
        let args = try launch.arguments()
        precondition(args.contains("/tmp/models with spaces") && args.contains("51539607552") && args.contains("900"))
        launch.context = "bad"
        do { _ = try launch.arguments(); fatalError("Bad context accepted") } catch {}
        var prefill = PrefillHistory()
        let epoch = Date(timeIntervalSince1970:100)
        prefill.record(time:epoch,tokens:100)
        prefill.record(time:epoch.addingTimeInterval(3),tokens:160)
        precondition(prefill.samples.last!.tokensPerSecond == 20)
        prefill.record(time:epoch.addingTimeInterval(6),tokens:30)
        precondition(prefill.samples.last!.tokensPerSecond == 10 && !prefill.samples.last!.startsNewSegment)
        var resetPrefill = PrefillHistory()
        resetPrefill.record(time:epoch,tokens:100)
        resetPrefill.record(time:epoch.addingTimeInterval(3),tokens:200)
        resetPrefill.record(time:epoch.addingTimeInterval(6),tokens:0)
        resetPrefill.record(time:epoch.addingTimeInterval(9),tokens:90)
        let resetPoints = TimelineValue.points(resetPrefill.samples,value:{$0.tokensPerSecond},maximumGap:8)
        precondition(resetPoints.map(\.value) == [100.0/3.0,0,30] && Set(resetPoints.map(\.segment)).count == 1)
        prefill.record(time:epoch.addingTimeInterval(12),tokens:30)
        precondition(prefill.samples.last!.tokensPerSecond == 0)
        prefill.record(time:epoch.addingTimeInterval(15),tokens:90)
        precondition(prefill.samples.last!.tokensPerSecond == 20)
        let prefillBeforeMissing = prefill.samples.count
        prefill.record(time:epoch.addingTimeInterval(18),tokens:nil)
        precondition(prefill.samples.count == prefillBeforeMissing && prefill.currentRate == nil)
        prefill.record(time:epoch.addingTimeInterval(21),tokens:1000)
        precondition(prefill.samples.count == prefillBeforeMissing && prefill.currentRate == nil)
        prefill.record(time:epoch.addingTimeInterval(24),tokens:Double.infinity)
        precondition(prefill.samples.count == prefillBeforeMissing && prefill.currentRate == nil)
        let chartSamples = [ThroughputSample(time:epoch,tokensPerSecond:10,memoryGiB:2),ThroughputSample(time:epoch.addingTimeInterval(3),tokensPerSecond:20,memoryGiB:nil),ThroughputSample(time:epoch.addingTimeInterval(30),tokensPerSecond:30,memoryGiB:3)]
        let outputPoints = TimelineValue.points(chartSamples,value:{$0.tokensPerSecond},maximumGap:8)
        precondition(outputPoints.map(\.segment) == [0,0,1])
        let delayedSamples = [ThroughputSample(time:epoch,tokensPerSecond:10,memoryGiB:2),ThroughputSample(time:epoch.addingTimeInterval(9),tokensPerSecond:20,memoryGiB:3),ThroughputSample(time:epoch.addingTimeInterval(12),tokensPerSecond:30,memoryGiB:3,startsNewSegment:true)]
        let delayedPoints = TimelineValue.points(delayedSamples,value:{$0.tokensPerSecond},maximumGap:ThroughputHistory.maximumInterval)
        precondition(delayedPoints.map(\.segment) == [0,0,1])
        let memoryPoints = TimelineValue.points(chartSamples,value:{$0.memoryGiB},maximumGap:8)
        precondition(memoryPoints.count == 2 && memoryPoints[0].segment != memoryPoints[1].segment)
        var history = ThroughputHistory()
        let time = Date(timeIntervalSince1970:1000)
        history.record(time:time,tokens:100,memoryBytes:nil)
        history.record(time:time.addingTimeInterval(3),tokens:160,memoryBytes:nil)
        precondition(history.samples.last!.tokensPerSecond == 20)
        history.record(time:time.addingTimeInterval(6),tokens:0,memoryBytes:nil)
        precondition(history.samples.count == 1 && history.currentRate == nil)
        history.record(time:time.addingTimeInterval(9),tokens:nil,memoryBytes:nil)
        history.record(time:time.addingTimeInterval(12),tokens:900,memoryBytes:nil)
        precondition(history.samples.count == 1 && history.currentRate == nil)
        history.record(time:time.addingTimeInterval(15),tokens:960,memoryBytes:nil)
        precondition(history.samples.last!.startsNewSegment && history.currentRate == 20)
        let beforeGap = history.samples.count
        history.record(time:time.addingTimeInterval(90),tokens:1500,memoryBytes:nil)
        precondition(history.samples.count == beforeGap)
        history.record(time:time.addingTimeInterval(93),tokens:1560,memoryBytes:nil)
        precondition(history.samples.last!.tokensPerSecond == 20)
        prefill.record(time:epoch.addingTimeInterval(27),tokens:0)
        prefill.record(time:epoch.addingTimeInterval(30),tokens:60)
        precondition(prefill.samples.last!.startsNewSegment && prefill.currentRate == 20)
        let beforePrefillGap = prefill.samples.count
        prefill.record(time:epoch.addingTimeInterval(90),tokens:1000)
        precondition(prefill.samples.count == beforePrefillGap)
        let profile = ConnectionProfile(name:"Local",endpoint:"http://127.0.0.1:12345",binary:"",arguments:"[\"serve\"]")
        try profile.validate()
        let encoded = try JSONEncoder().encode(profile)
        let decoded = try JSONDecoder().decode(ConnectionProfile.self,from:encoded)
        precondition(decoded == profile)
        let saved = "[\"serve\",\"--metrics\",\"--port\",\"9000\",\"--ctx-size\",\"131072\",\"--prefix-cache-disk\",\"20GB\",\"--temp\",\"0.7\"]"
        let restored = try LaunchSettings.read(saved)
        precondition(restored.port == 9000 && restored.context == "131072" && restored.diskCacheGiB == "20.0")
        let merged = try restored.mergedArguments(saved)
        precondition(merged.suffix(2) == ["--temp","0.7"])
        try LaunchConfiguration.validateServer(merged,endpoint:"http://127.0.0.1:9000")
        do { try LaunchConfiguration.validateServer(merged,endpoint:"http://127.0.0.1:8000"); fatalError("Port mismatch accepted") } catch {}
        do { try LaunchConfiguration.validateServer(["serve","--host","0.0.0.0"],endpoint:"http://127.0.0.1:12345"); fatalError("LAN bind accepted") } catch {}
        let cache = PrefixCacheStatus(["settings":["prefix_cache":["ram_enabled":true,"mem_bytes":2147483648.0,"disk_bytes":1073741824.0,"disk_used_bytes":536870912.0,"disk_entries":12]]])
        precondition(cache.available && cache.diskFraction == 0.5 && cache.diskEntries == 12)
        precondition(!PrefixCacheStatus([:]).available)
        let disabled = PrefixCacheStatus(["settings":["prefix_cache":["ram_enabled":false,"disk_bytes":0.0]]])
        precondition(disabled.ramEnabled == false && disabled.diskFraction == nil)
        let invalid = PrefixCacheStatus(["settings":["prefix_cache":["disk_bytes":Double.nan,"disk_used_bytes":-1.0]]])
        precondition(invalid.diskBudget == nil && invalid.diskUsed == nil)
        let serving = ServingStats(["counters":["prompt_tokens_total":10000,"prefill_tokens_total":1000,"prefix_cache_tokens_total":9000,"generation_tokens_total":200,"requests_success_total":4,"requests_cancelled_total":1,"requests_failed_total":1,"requests_rejected_total":1,"prefix_cache_queries_total":10,"prefix_cache_hits_total":8],"histograms":["prefill_time_seconds":["sum":5.0,"count":4],"decode_time_seconds":["sum":10.0,"count":4],"time_to_first_token_seconds":["sum":1.6,"count":4]]])
        precondition(serving.prefillRate == 200 && serving.decodeRate == 20)
        precondition(serving.meanTTFT == 0.4 && serving.cachedFraction == 0.9 && serving.cacheHitFraction == 0.8)
        precondition(serving.totalRequests == 7 && serving.successfulRequests == 4)
        precondition(serving.csv.contains("Forwarded prompt tokens,1000") && !serving.csv.contains("Authorization"))
        let noRequests = ServingStats(["counters":["prompt_tokens_total":0,"generation_tokens_total":0],"histograms":["prefill_time_seconds":["sum":0,"count":0]]])
        precondition(noRequests.prefillRate == nil && noRequests.cachedFraction == nil && noRequests.totalRequests == nil)
        let legacy = ServingStats(["counters":["prompt_tokens_total":10000,"prefix_cache_tokens_total":9000],"histograms":["prefill_time_seconds":["sum":5,"count":1]]])
        precondition(legacy.prefillRate == 200)
        let malformed = ServingStats(["counters":["prompt_tokens_total":Double.infinity,"prefill_tokens_total":-1,"generation_tokens_total":true],"histograms":["prefill_time_seconds":["sum":Double.nan,"count":1]]])
        precondition(malformed.promptTokens == nil && malformed.outputTokens == nil && malformed.prefillRate == nil)
        let invalidForwarded = ServingStats(["counters":["prompt_tokens_total":10000,"prefix_cache_tokens_total":9000,"prefill_tokens_total":-1],"histograms":["prefill_time_seconds":["sum":5]]])
        precondition(invalidForwarded.prefillRate == nil)
        let largeCount = ServingStats(["counters":["prompt_tokens_total":123456789]])
        precondition(largeCount.csv.contains("123456789"))
        precondition(!ServingStats([:]).available)
        let launchCommand = try AgentLaunchCommand.make(binary:"",client:"pi",endpoint:"http://127.0.0.1:8000",model:"model with spaces")
        precondition(launchCommand.contains("launch pi --url") && launchCommand.contains("--model 'model with spaces'"))
        let defaultLaunch = try AgentLaunchCommand.make(binary:"",client:"hermes",endpoint:"http://localhost:12345",model:nil)
        precondition(!defaultLaunch.contains("--model"))
        do { _ = try AgentLaunchCommand.make(binary:"",client:"unknown",endpoint:"http://localhost:12345",model:nil); fatalError("Unknown client accepted") } catch {}
        do { _ = try AgentLaunchCommand.make(binary:"",client:"pi",endpoint:"https://example.com",model:nil); fatalError("Remote endpoint accepted") } catch {}
        let scriptDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:scriptDirectory,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:scriptDirectory) }
        let script = scriptDirectory.appendingPathComponent("test ' cli")
        try "#!/bin/sh\nexec /usr/bin/python3 -I -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \"$@\"\n".write(to:script,atomically:true,encoding:.utf8)
        try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:script.path)
        let sentinel = scriptDirectory.appendingPathComponent("should-not-exist")
        let hostileModel = "model'; touch " + sentinel.path + "; #$(id)"
        let quotedCommand = try AgentLaunchCommand.make(binary:script.path,client:"pi",endpoint:"http://127.0.0.1:8000",model:hostileModel)
        let shell = Process(), captured = Pipe()
        shell.executableURL = URL(fileURLWithPath:"/bin/sh"); shell.arguments = ["-c",quotedCommand]; shell.standardOutput = captured
        try shell.run()
        let deadline = Date().addingTimeInterval(5)
        while shell.isRunning && Date() < deadline { Thread.sleep(forTimeInterval:0.02) }
        guard !shell.isRunning else { shell.terminate(); throw ServerFailure(message:"Command quoting test timed out") }
        let words = try JSONDecoder().decode([String].self,from:captured.fileHandleForReading.readDataToEndOfFile())
        precondition(shell.terminationStatus == 0 && words == ["launch","pi","--url","http://127.0.0.1:8000","--model",hostileModel])
        precondition(!FileManager.default.fileExists(atPath:sentinel.path))
        print("Client behavior tests passed")
    }
}
