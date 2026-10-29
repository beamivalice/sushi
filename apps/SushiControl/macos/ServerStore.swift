import SwiftUI
import AppKit

@MainActor final class ServerStore: ObservableObject {
    static let shared = ServerStore(benchmark:BenchmarkRunner(storageURL:FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Sushi Control/benchmarks.json")),tool:ToolTask(storageURL:FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Sushi Control/tasks.json")),usage:UsageHistory(storageURL:FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Sushi Control/usage.json")))
    let benchmark:BenchmarkRunner
    let usage:UsageHistory
    @Published var usageEnabled = false { didSet { defaults.set(usageEnabled,forKey:"recordUsage"); usage.resetSampling() } }
    let tool:ToolTask
    private var systemMonitor = SystemMonitor()
    @Published var system = SystemSnapshot.capture()
    @Published var page:Page? = .status
    @Published var conversionPlan = ConversionPlan() { didSet { if let data = try? JSONEncoder().encode(conversionPlan) { defaults.set(data,forKey:"conversionPlan") } } }
    @Published var setupPresented = false
    func finishSetup() { defaults.set(true,forKey:"setupCompleted"); setupPresented = false }
    @Published var serverSection = "Connection"
    @Published var endpoint = "http://127.0.0.1:12345"
    @Published var apiKey = ""
    @Published var startServerAtLogin = false { didSet { defaults.set(startServerAtLogin,forKey:"startServerAtLogin") } }
    @Published var recoverServerAfterCrash = false { didSet { defaults.set(recoverServerAfterCrash,forKey:"recoverServerAfterCrash") } }
    @Published var binary = ""
    @Published var toolBinary = ""
    @Published var arguments = "[\"serve\", \"--metrics\"]"
    @Published var profiles:[ConnectionProfile] = []
    @Published var profileID = ""
    @Published var profileName = ""
    @Published var rememberKey = false
    @Published var favoriteModelIDs:Set<String> = [] { didSet { defaults.set(favoriteModelIDs.sorted(),forKey:"favoriteModels") } }
    @Published var menuFavoritesOnly = false { didSet { defaults.set(menuFavoritesOnly,forKey:"menuFavoritesOnly") } }
    @Published var librarySort = "Name" { didSet { defaults.set(librarySort,forKey:"librarySort") } }
    var menuModels:[ModelEntry] {
        let menuFavorites = menuFavoritesOnly ? favoriteModelIDs.union(selected.isEmpty ? [] : [selected]) : favoriteModelIDs
        return ModelEntry.library(models,scope:menuFavoritesOnly ? "Menu" : "All",favorites:menuFavorites,sort:"Menu")
    }
    static var menuOptionalMetricOptions:[String] { menuMetricOptions.filter { $0 != "Output" } }
    func toggleFavorite(_ id:String) {
        if favoriteModelIDs.contains(id) { favoriteModelIDs.remove(id) } else { favoriteModelIDs.insert(id) }
    }
    func copyModelID(_ id:String) {
        copyText(id,label:"Model ID")
    }
    @Published var models:[ModelEntry] = []
    @Published var selected = ""
    @Published var rescanAvailable = true
    @Published var props:[String:Any] = [:]
    @Published var metrics:[String:Any] = [:]
    @Published var metricsTemporarilyUnavailable = false
    @Published var online = false
    @Published var backendSleeping = false
    private var refreshing = false
    private var refreshPending = false
    @Published var checking = false
    @Published var busy = false
    @Published var error = ""
    @Published var notice = ""
    @Published var owned = false
    @Published private(set) var serviceKnown = false
    @Published var serviceRunning = false
    @Published var serviceExitCode:Int?
    @Published var connectionError = ""
    @Published var controllerToken = "" { didSet { controllerState = nil; controllerError = "" } }
    @Published var controllerState:DesktopControllerState?
    @Published var controllerError = ""
    func chooseControllerToken() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let values = try url.resourceValues(forKeys:[.fileSizeKey,.isRegularFileKey])
            guard values.isRegularFile == true, (values.fileSize ?? 0) <= 4096 else { throw ServerFailure(message:"Choose a token file smaller than 4 KiB.") }
            let token = try String(contentsOf:url,encoding:.utf8).trimmingCharacters(in:.whitespacesAndNewlines)
            guard !token.isEmpty, !token.contains(where:{$0.isWhitespace || $0.isNewline}) else { throw ServerFailure(message:"The controller token must be a single nonempty value.") }
            controllerToken = token; refreshNow()
        } catch { controllerError = error.localizedDescription }
    }
    func controllerAction(_ action:String) {
        guard controllerState != nil, !controllerToken.isEmpty, !busy, !tool.exclusiveServerUse,
              ["pause","resume","load","unload"].contains(action) else { return }
        generation += 1; busy = true; operation = "Updating controller…"
        let client = Connection(endpoint:connectedEndpoint,apiKey:controllerToken)
        Task {
            do { _ = try await client.fetch("control/action",body:["action":action]); controllerError = "" }
            catch { controllerError = error.localizedDescription }
            busy = false; operation = ""; await refresh()
        }
    }

    @Published var diagnosticText = ""
    @Published var testingConnection = false
    @Published var operation = ""
    @Published var lastUpdated:Date?
    @Published var history = ThroughputHistory()
    @Published var prefillHistory = PrefillHistory()
    @Published var logText = ""
    @Published var logError = ""
    @Published var logStream = "output"
    @Published var refreshInterval:Double = 3 {
        didSet {
            guard [2.0,3,5,10].contains(refreshInterval) else { refreshInterval = oldValue; return }
            defaults.set(refreshInterval,forKey:"refreshInterval")
            if timer != nil { scheduleRefresh() }
        }
    }
    @Published var showDockIcon = false {
        didSet { defaults.set(showDockIcon,forKey:"showDockIcon"); AppDelegate.instance?.applyActivationPolicy(keepDockVisible:showDockIcon) }
    }
    @Published var sidebarVisible = true { didSet { defaults.set(sidebarVisible,forKey:"sidebarVisible") } }
    @Published var appearance = "system"
    static let menuMetricOptions = ["GPU","Output","Requests","Cache reuse"]
    @Published var menuBarMetric = "None" {
        didSet {
            guard ["None","Requests","Output","GPU"].contains(menuBarMetric) else { menuBarMetric = oldValue; return }
            defaults.set(menuBarMetric,forKey:"menuBarMetric")
        }
    }
    @Published var menuShowGraph = true { didSet { defaults.set(menuShowGraph,forKey:"menuShowGraph") } }
    @Published var menuMetricNames = ["GPU","Requests","Cache reuse"] {
        didSet {
            guard menuMetricNames.allSatisfy(Self.menuMetricOptions.contains), Set(menuMetricNames).count == menuMetricNames.count else { menuMetricNames = oldValue; return }
            defaults.set(menuMetricNames,forKey:"menuMetricNames")
        }
    }
    var menuSecondaryMetrics:[String] { menuMetricNames.filter { $0 != "Output" && $0 != "Prefill" } }
    var menuBarText:String {
        guard online && !backendSleeping, menuBarMetric != "None" else { return "" }
        let value = menuValue(menuBarMetric)
        return value == "—" ? "" : " " + value
    }
    func menuValue(_ name:String) -> String {
        guard online && !backendSleeping else { return "—" }
        switch name { case "GPU":return memory; case "Output":return outputRate; case "Requests":return activity; case "Cache reuse":return cacheReuse; default:return "—" }
    }
    func toggleMenuMetric(_ name:String) {
        guard Self.menuMetricOptions.contains(name) else { return }
        if menuMetricNames.contains(name) { menuMetricNames.removeAll { $0 == name } } else { menuMetricNames.append(name) }
    }
    private var service:ManagedServer
    @Published var existingAgentPath = ""
    var usesExistingAgent:Bool { service.externalPlist != nil }
    var canStart:Bool { usesExistingAgent || !binary.isEmpty }
    var managedDescription:String { usesExistingAgent ? "selected launch agent" : "app-managed server" }
    func chooseExistingAgent() {
        guard !busy && !owned else { return }
        let panel = NSOpenPanel(); panel.canChooseFiles = true; panel.canChooseDirectories = false
        panel.message = "Choose the launch agent that starts this Sushi server. Its file will be preserved."
        panel.directoryURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/LaunchAgents")
        if panel.runModal() == .OK, let url = panel.url {
            do {
                let candidate = try ManagedServer.existingAgent(url)
                service = candidate; existingAgentPath = url.path
                defaults.set(url.path,forKey:"existingLaunchAgent")
                generation += 1; notice = "Existing launch agent selected. Its configuration is unchanged."
                Task { await refresh() }
            } catch { self.error = error.localizedDescription }
        }
    }
    func detachExistingAgent() {
        guard !busy && usesExistingAgent else { return }
        service = ManagedServer(); existingAgentPath = ""; owned = false; serviceRunning = false
        defaults.removeObject(forKey:"existingLaunchAgent"); generation += 1
        notice = "Launch agent detached. Its server keeps running."
        Task { await refresh() }
    }
    private let defaults:UserDefaults
    private var timer:Timer?
    private var generation = 0
    private var connection = Connection(endpoint:"http://127.0.0.1:12345")
    var outputRate:String { guard online && !backendSleeping && !metricsTemporarilyUnavailable else { return "—" }; return history.currentRate.map { String(format:"%.1f tok/s",$0) } ?? "—" }
    var prefillRate:String { guard online && !backendSleeping && !metricsTemporarilyUnavailable else { return "—" }; return prefillHistory.currentRate.map { String(format:"%.0f tok/s",$0) } ?? "—" }
    var canRunOfflineMaintenance:Bool { serviceKnown && !owned && !refreshing && !checking && !online && !serviceRunning && !busy && !tool.running }
    var canTrashLocalModels:Bool { canRunOfflineMaintenance }
    var connectedEndpoint:String { connection.endpoint }
    var selectedModel:ModelEntry? { models.first(where:{$0.id == selected}) }
    var statusColor:Color { !online ? .secondary : controllerState?.paused == true ? .orange : backendSleeping ? .orange : (active ?? 0) > 0 ? .blue : .green }
    var gauges:[String:Any] { metrics["gauges"] as? [String:Any] ?? [:] }
    var active:Int? { (gauges["requests_running"] as? NSNumber)?.intValue }
    var activity:String { active.map(String.init) ?? "—" }
    var state:String { !online ? (owned || usesExistingAgent ? (serviceRunning ? "Connecting" : serviceExitCode == 0 || serviceExitCode == nil ? "Stopped" : "Failed") : "Disconnected") : controllerState?.paused == true ? "Paused" : backendSleeping ? "Sleeping" : models.contains(where:{$0.loaded}) ? "Ready" : "No model loaded" }
    var memory:String {
        let bytes = (props["memory"] as? [String:Any])?["active_bytes"] as? Double
        return bytes.map { String(format:"%.1f GiB",$0 / 1073741824) } ?? "—"
    }
    var cacheReuse:String {
        let counters = metrics["counters"] as? [String:Any] ?? [:]
        let total = counters["prompt_tokens_total"] as? Double ?? 0
        return total > 0 ? String(format:"%.0f%%",(counters["prefix_cache_tokens_total"] as? Double ?? 0)/total*100) : "—"
    }
    var logURL:URL { logStream == "error" ? service.errorLogURL : service.logURL }
    init(defaults:UserDefaults = .standard,automaticallyRefresh:Bool = true,service:ManagedServer = ManagedServer(),benchmark:BenchmarkRunner? = nil,tool:ToolTask? = nil,usage:UsageHistory? = nil) {
        self.defaults = defaults; self.service = service; self.tool = tool ?? ToolTask(); self.benchmark = benchmark ?? BenchmarkRunner(); self.usage = usage ?? UsageHistory()
        usageEnabled = defaults.bool(forKey:"recordUsage")
        existingAgentPath = service.externalPlist?.path ?? ""
        if let data = defaults.data(forKey:"conversionPlan"), let plan = try? JSONDecoder().decode(ConversionPlan.self,from:data) { conversionPlan = plan }
        endpoint = defaults.string(forKey:"serverEndpoint") ?? endpoint
        binary = defaults.string(forKey:"serverBinary") ?? binary
        arguments = defaults.string(forKey:"serverArguments") ?? arguments
        toolBinary = defaults.string(forKey:"toolBinary") ?? binary
        startServerAtLogin = defaults.bool(forKey:"startServerAtLogin")
        recoverServerAfterCrash = defaults.bool(forKey:"recoverServerAfterCrash")
        profiles = defaults.data(forKey:"connectionProfiles").flatMap { try? JSONDecoder().decode([ConnectionProfile].self,from:$0) } ?? []
        profileID = defaults.string(forKey:"selectedProfile") ?? ""
        sidebarVisible = defaults.object(forKey:"sidebarVisible") as? Bool ?? true
        appearance = defaults.string(forKey:"appearance") ?? "system"
        let interval = defaults.double(forKey:"refreshInterval")
        refreshInterval = [2.0,3,5,10].contains(interval) ? interval : 3
        showDockIcon = defaults.bool(forKey:"showDockIcon")
        menuBarMetric = defaults.string(forKey:"menuBarMetric").flatMap { ["None","Requests","Output","GPU"].contains($0) ? $0 : nil } ?? (defaults.bool(forKey:"showActivity") ? "Requests" : "None")
        menuShowGraph = defaults.object(forKey:"menuShowGraph") as? Bool ?? true
        if let metrics = defaults.stringArray(forKey:"menuMetricNames"), metrics.allSatisfy(Self.menuMetricOptions.contains), Set(metrics).count == metrics.count { menuMetricNames = metrics }
        favoriteModelIDs = Set(defaults.stringArray(forKey:"favoriteModels") ?? [])
        menuFavoritesOnly = defaults.bool(forKey:"menuFavoritesOnly")
        librarySort = defaults.string(forKey:"librarySort").flatMap { ["Name","Memory","Context"].contains($0) ? $0 : nil } ?? "Name"
        if let path = defaults.string(forKey:"existingLaunchAgent"), let existing = try? ManagedServer.existingAgent(URL(fileURLWithPath:path)) {
            self.service = existing; existingAgentPath = path
        }
        connection = Connection(endpoint:endpoint)
        if let profile = profiles.first(where:{$0.id == profileID}) {
            profileName = profile.name; endpoint = profile.endpoint; binary = profile.binary; arguments = profile.arguments; rememberKey = profile.rememberKey
            apiKey = profile.rememberKey ? CredentialStore().read(profile.id) ?? "" : ""
            connection = Connection(endpoint:endpoint,apiKey:apiKey)
        }
        let configured = defaults.object(forKey:"serverEndpoint") != nil || !profiles.isEmpty || !binary.isEmpty || usesExistingAgent
        setupPresented = !defaults.bool(forKey:"setupCompleted") && !configured
        if automaticallyRefresh {
        scheduleRefresh()
        Task { await refresh() }
        }
    }
    private func scheduleRefresh() {
        timer?.invalidate()
        let next = Timer(timeInterval:refreshInterval,repeats:true) { [weak self] _ in Task { @MainActor in await self?.refresh() } }
        RunLoop.main.add(next,forMode:.common)
        timer = next
    }
    func applyAppearance() {
        defaults.set(appearance,forKey:"appearance")
        NSApp.appearance = appearance == "light" ? NSAppearance(named:.aqua) : appearance == "dark" ? NSAppearance(named:.darkAqua) : nil
        AppDelegate.instance?.popover.appearance = NSApp.appearance
    }
    func saveProfile() {
        guard !owned && !busy else { return }
        var profile = ConnectionProfile(name:profileName.trimmingCharacters(in:.whitespacesAndNewlines),endpoint:endpoint,binary:binary,arguments:arguments,rememberKey:rememberKey)
        if !profileID.isEmpty { profile.id = profileID }
        if profile.name.isEmpty { profile.name = endpoint }
        do {
            try profile.validate()
            if rememberKey { try CredentialStore().write(apiKey,id:profile.id) }
            else { try CredentialStore().remove(profile.id) }
            if let index = profiles.firstIndex(where:{$0.id == profile.id}) { profiles[index] = profile } else { profiles.append(profile) }
            profileID = profile.id; profileName = profile.name
            defaults.set(profile.id,forKey:"selectedProfile")
            try persistProfiles(); saveConnection()
            notice = "Profile saved."
        } catch { self.error = error.localizedDescription }
    }
    func selectProfile(_ id:String) {
        guard !owned && !busy, let profile = profiles.first(where:{$0.id == id}) else { return }
        defaults.set(id,forKey:"selectedProfile")
        profileID = id; profileName = profile.name; endpoint = profile.endpoint; binary = profile.binary; arguments = profile.arguments; rememberKey = profile.rememberKey
        apiKey = profile.rememberKey ? CredentialStore().read(id) ?? "" : ""
        saveConnection()
    }
    func newProfile() {
        guard !owned && !busy else { return }
        profileID = ""; profileName = ""; rememberKey = false; apiKey = ""
        defaults.removeObject(forKey:"selectedProfile")
    }
    func deleteProfile() {
        guard !owned && !busy && !profileID.isEmpty else { return }
        do {
            try CredentialStore().remove(profileID)
            profiles.removeAll { $0.id == profileID }; try persistProfiles()
            profileID = ""; profileName = ""; rememberKey = false
            defaults.removeObject(forKey:"selectedProfile")
            notice = "Profile removed."
        } catch { self.error = error.localizedDescription }
    }
    private func persistProfiles() throws {
        defaults.set(try JSONEncoder().encode(profiles),forKey:"connectionProfiles")
    }
    func saveConnection() {
        let updated = Connection(endpoint:endpoint,apiKey:apiKey)
        do {
            _ = try updated.request("health")
            _ = try LaunchConfiguration.decode(arguments)
        } catch { self.error = error.localizedDescription; return }
        if let saved = profiles.first(where:{$0.id == profileID}), saved.endpoint != endpoint || saved.binary != binary || saved.arguments != arguments {
            defaults.removeObject(forKey:"selectedProfile")
        }
        generation += 1; connection = updated; history.reset(); prefillHistory.reset(); usage.resetSampling(); rescanAvailable = true; controllerToken = ""
        defaults.set(endpoint,forKey:"serverEndpoint")
        defaults.set(binary,forKey:"serverBinary")
        defaults.set(arguments,forKey:"serverArguments")
        error = ""; connectionError = ""; online = false; backendSleeping = false; lastUpdated = nil; props = [:]; metrics = [:]; metricsTemporarilyUnavailable = false; models = []; selected = ""
        notice = "Connection saved. Checking the server…"
        Task { await refresh() }
    }
    func refreshNow() {
        guard !checking else { return }
        checking = true
        Task { await refresh(); checking = false }
    }
    func refresh() async {
        system = systemMonitor.sample()
        guard !busy else { return }
        guard !refreshing else { refreshPending = true; return }
        refreshing = true
        defer {
            refreshing = false
            if refreshPending {
                refreshPending = false
                Task { await refresh() }
            }
        }
        let client = connection, version = generation
        let serviceState = await service.snapshot()
        guard version == generation && !busy else { return }
        serviceKnown = serviceState.known
        if serviceState.known {
            owned = serviceState.registered; serviceRunning = serviceState.running; serviceExitCode = serviceState.exitCode
        } else if error.isEmpty {
            error = "Cannot check the app-managed server. Its last known state has been kept. Try refreshing before using server controls."
        }
        do {
            let health = try? await client.fetch("health")
            let result = try await client.fetch("v1/models")
            guard version == generation && !busy else { return }
            backendSleeping = health?["model_state"] as? String == "sleeping"
            models = ModelEntry.parse(result).map { entry in
                var model = entry
                if backendSleeping { model.loaded = false; model.state = "sleeping"; model.residentBytes = nil }
                return model
            }; online = true
            selected = ModelEntry.selection(selected,in:models)
            let nextProps = (try? await client.fetch("props")) ?? [:]
            let nextMetrics:[String:Any]?
            var metricsUnsupported = false
            do { nextMetrics = try await client.fetch("metrics.json") }
            catch {
                nextMetrics = nil
                if let failure = error as? ServerFailure, let status = failure.status {
                    metricsUnsupported = (400..<500).contains(status) || (status == 503 && failure.message.localizedCaseInsensitiveContains("metrics not enabled"))
                }
            }
            guard version == generation && !busy else { return }
            props = nextProps
            if let nextMetrics { metrics = nextMetrics; metricsTemporarilyUnavailable = false }
            else if metricsUnsupported { metrics = [:]; metricsTemporarilyUnavailable = false }
            else { metrics = [:]; metricsTemporarilyUnavailable = true }
            let snapshot = nextMetrics?["gauges"] as? [String:Any] ?? [:]
            prefillHistory.record(time:Date(),tokens:ServingStats.number(snapshot["prefill_tokens_live"]))
            history.record(time:Date(),tokens:ServingStats.number(snapshot["generation_tokens_live"]),memoryBytes:(nextProps["memory"] as? [String:Any])?["active_bytes"] as? Double)
            if !controllerToken.isEmpty {
                let token = controllerToken
                do {
                    let snapshot = try await Connection(endpoint:client.endpoint,apiKey:token).fetch("control/status")
                    guard version == generation && token == controllerToken && !busy else { return }
                    controllerState = try DesktopControllerState(snapshot); controllerError = ""
                    if let controllerState {
                        for id in controllerState.models.keys.sorted() where !models.contains(where:{$0.id == id}) {
                            models.append(ModelEntry(id:id,loaded:false,state:"available",architecture:"",quantization:""))
                        }
                        selected = ModelEntry.selection(selected,in:models)
                    }
                } catch {
                    guard version == generation && token == controllerToken && !busy else { return }
                    controllerState = nil; controllerError = error.localizedDescription
                }
            }
            if usageEnabled { usage.record(endpoint:client.endpoint,metrics:nextMetrics ?? [:]) }
            connectionError = ""; lastUpdated = Date()
        } catch {
            guard version == generation && !busy else { return }
            online = false; backendSleeping = false; controllerState = nil; models = []; selected = ""; props = [:]; metrics = [:]; metricsTemporarilyUnavailable = false; history.reset(); prefillHistory.reset(); usage.resetSampling()
            if let network = error as? URLError, [.cannotConnectToHost,.networkConnectionLost,.timedOut].contains(network.code) {
                connectionError = "Cannot reach Sushi. Check that the server is running and its port matches Server settings."
            } else { connectionError = error.localizedDescription }
        }
    }
    func runBenchmark(_ plan:BenchmarkPlan) {
        guard !busy, !tool.running, online else { error = "Use an idle connected server before starting a benchmark."; return }
        do {
            var configured = plan
            if let model = models.first(where:{$0.id == plan.model}) {
                configured.configuration = [model.architecture,model.quantization,"context \(model.contextTokens.map(String.init) ?? "unknown")","KV \(model.kvFormat ?? "unknown")","DFlash \(model.drafterLoaded.map(String.init) ?? "unknown")","MTP \(model.mtpLoaded.map(String.init) ?? "unknown")","Sushi \((props["settings"] as? [String:Any])?["version"] as? String ?? "unreported")"].joined(separator:" · ")
            }
            try benchmark.start(plan:configured,connection:Connection(endpoint:endpoint,apiKey:apiKey),onFinished:{ [weak self] in
                self?.busy = false; self?.operation = ""; self?.refreshNow()
            })
            busy = true; operation = "Benchmarking…"; error = ""
        } catch { self.error = error.localizedDescription }
    }
    func canModelAction(_ path:String,model:String? = nil) -> Bool {
        guard online, !busy, !tool.exclusiveServerUse,
              controllerToken.isEmpty || controllerState != nil,
              path != "models/rescan" || rescanAvailable else { return false }
        return (try? ModelActionPlan.make(path,model:model,controller:controllerState)) != nil
    }
    func modelAction(_ path:String, model:String? = nil) {
        guard !busy && !tool.exclusiveServerUse else { return }
        if !controllerToken.isEmpty && controllerState == nil { error = "Reconnect the desktop controller before changing models, or disconnect it to use the standard Sushi API."; return }
        let plan:ModelActionPlan
        do { plan = try ModelActionPlan.make(path,model:model,controller:controllerState) }
        catch { self.error = error.localizedDescription; return }
        generation += 1; busy = true; error = ""; notice = ""
        operation = path == "load-model" ? "Loading model…" : path == "unload-model" ? "Unloading model…" : "Scanning models…"
        let client = plan.usesController ? Connection(endpoint:connectedEndpoint,apiKey:controllerToken) : connection
        Task {
            do {
                try await plan.execute(client)
                notice = path == "load-model" ? (plan.usesController ? "Model loaded and selected." : "Model loaded and set as the server default.") : path == "unload-model" ? "Model unloaded." : "Model library refreshed."
            } catch {
                if path == "models/rescan", (error as? ServerFailure)?.status == 404 {
                    rescanAvailable = false
                    notice = "This server does not expose model rescanning. Refreshing its existing model list instead."
                } else { self.error = error.localizedDescription }
            }
            busy = false; operation = ""; await refresh()
        }
    }
    func saveServerStartup() {
        guard !usesExistingAgent, canRunOfflineMaintenance else { error = "Stop the app-managed server before saving startup settings."; return }
        busy = true; operation = "Saving server startup…"; error = ""
        Task {
            do {
                let state = try await service.checkedSnapshot()
                guard !state.registered && !state.running else { throw ServerFailure(message:"Stop the app-managed server before saving startup settings.") }
                let args = try LaunchConfiguration.decode(arguments)
                try LaunchConfiguration.validateServer(args,endpoint:endpoint)
                try service.install(binary:binary,arguments:args,startAtLogin:startServerAtLogin,recoverAfterCrash:recoverServerAfterCrash)
                saveConnection()
                notice = startServerAtLogin ? "Server configured to start at your next login. It has not been started now." : "Server login startup disabled. Recovery settings apply the next time it starts."
            } catch { self.error = error.localizedDescription }
            busy = false; operation = ""; await refresh()
        }
    }
    func start() {
        guard !busy && !tool.exclusiveServerUse && !online && (!owned || !serviceRunning) else { return }
        generation += 1; busy = true; operation = "Starting server…"; error = ""; notice = ""
        Task {
            do {
                _ = try Connection(endpoint:endpoint,apiKey:apiKey).request("health")
                let args = usesExistingAgent ? [] : try LaunchConfiguration.decode(arguments)
                if !usesExistingAgent { try LaunchConfiguration.validateServer(args,endpoint:endpoint) }
                let marker = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".sushi/runs/QUIET")
                guard !FileManager.default.fileExists(atPath:marker.path) else { throw ServerFailure(message:"A Sushi benchmark quiet window is active.") }
                let current = try await service.checkedSnapshot()
                guard !current.running else { throw ServerFailure(message:"The app-managed server is already running. Refresh its status before starting it again.") }
                if !usesExistingAgent {
                    if current.registered { try await service.stop(); owned = false }
                    try service.install(binary:binary,arguments:args,startAtLogin:startServerAtLogin,recoverAfterCrash:recoverServerAfterCrash); saveConnection()
                }
                try await service.start()
                owned = true; serviceRunning = true; notice = "Server started."
            } catch { self.error = error.localizedDescription }
            busy = false; operation = ""; await refresh()
        }
    }
    func stop(restart:Bool) {
        guard !busy && !tool.exclusiveServerUse && owned else { return }
        generation += 1; busy = true; operation = restart ? "Restarting server…" : "Stopping server…"; error = ""; notice = ""
        Task {
            do {
                let restartArguments = restart && !usesExistingAgent ? try LaunchConfiguration.decode(arguments) : []
                if restart && !usesExistingAgent {
                    try LaunchConfiguration.validateServer(restartArguments,endpoint:endpoint)
                    guard binary.hasPrefix("/"), FileManager.default.isExecutableFile(atPath:binary) else { throw ServerFailure(message:"Choose an installed Sushi executable before restarting.") }
                }
                if restart { try service.validateExistingAgent() }
                try await service.stop(); owned = false; online = false; history.reset(); prefillHistory.reset()
                if restart {
                    let marker = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".sushi/runs/QUIET")
                    guard !FileManager.default.fileExists(atPath:marker.path) else { throw ServerFailure(message:"Server stopped. Wait for the benchmark to finish before starting it again.") }
                    if !usesExistingAgent {
                        try service.install(binary:binary,arguments:restartArguments,startAtLogin:startServerAtLogin,recoverAfterCrash:recoverServerAfterCrash)
                        saveConnection()
                    }
                    try await service.start(); owned = true
                }
                notice = restart ? "Server restarted." : "Server stopped."
            } catch { self.error = error.localizedDescription }
            busy = false; operation = ""; await refresh()
        }
    }
    func importModel() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = "Choose a model folder on this Mac. Loading registers it with Sushi."
        if panel.runModal() == .OK, let path = panel.url?.path { modelAction("load-model",model:path) }
    }
    func setToolBinary(_ path:String) throws {
        guard !tool.running, !busy else { throw ServerFailure(message:"Wait for the current operation to finish before changing the CLI executable.") }
        guard path.hasPrefix("/"), FileManager.default.isExecutableFile(atPath:path) else { throw ServerFailure(message:"Choose an installed Sushi executable.") }
        toolBinary = path; defaults.set(path,forKey:"toolBinary")
    }
    func chooseToolBinary() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.canChooseFiles = true
        panel.message = "Choose the Sushi CLI for downloads and runtime tools. This does not change the server launcher."
        if panel.runModal() == .OK, let path = panel.url?.path {
            do { try setToolBinary(path) } catch { self.error = error.localizedDescription }
        }
    }
    func chooseBinary() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.canChooseFiles = true
        if panel.runModal() == .OK, let path = panel.url?.path { binary = path }
    }
    func testConnection() {
        guard !testingConnection else { return }
        testingConnection = true; diagnosticText = "Checking connection…"
        let client = connection
        Task {
            var lines = ["Sushi Control 0.1", "Server: \(client.endpoint)"]
            for path in ["health","v1/models","props","metrics.json"] {
                let start = Date()
                do {
                    _ = try await client.fetch(path)
                    lines.append("/\(path): OK (\(Int(Date().timeIntervalSince(start)*1000)) ms)")
                } catch { lines.append("/\(path): \(error.localizedDescription)") }
            }
            lines.append("Server management: \(await service.registered() ? "Available" : "External")")
            let report = lines.joined(separator:"\n")
            diagnosticText = client.apiKey.isEmpty ? report : report.replacingOccurrences(of:client.apiKey,with:"[redacted]"); testingConnection = false
        }
    }
    func exportServingStats() {
        let stats = ServingStats(metrics)
        guard stats.available else { return }
        let panel = NSSavePanel(); panel.nameFieldStringValue = "Sushi Serving Stats.csv"
        if panel.runModal() == .OK, let url = panel.url {
            do { try stats.csv.write(to:url,atomically:true,encoding:.utf8); notice = "Serving statistics exported." }
            catch { self.error = error.localizedDescription }
        }
    }
    func exportDiagnostics() {
        let panel = NSSavePanel(); panel.nameFieldStringValue = "Sushi Diagnostics.txt"
        if panel.runModal() == .OK, let url = panel.url {
            do { try diagnosticText.write(to:url,atomically:true,encoding:.utf8) }
            catch { self.error = error.localizedDescription }
        }
    }
    func copyText(_ text:String,label:String) {
        NSPasteboard.general.clearContents(); NSPasteboard.general.setString(text,forType:.string)
        notice = label + " copied."
    }
    func copyEndpoint() {
        guard let url = try? connection.request("v1").url else { return }
        copyText(url.absoluteString,label:"API endpoint")
    }
    func revealLogs() { NSWorkspace.shared.activateFileViewerSelecting([logURL]) }
    func dashboard() { if let url = try? connection.request("dashboard").url { NSWorkspace.shared.open(url) } }
    func readLog() {
        logError = ""
        do {
            let handle = try FileHandle(forReadingFrom:logURL)
            defer { try? handle.close() }
            let size = try handle.seekToEnd()
            try handle.seek(toOffset:size > 100000 ? size-100000 : 0)
            logText = String(decoding:try handle.readToEnd() ?? Data(),as:UTF8.self)
        } catch {
            logText = ""
            logError = "Cannot read \(logURL.path): \(error.localizedDescription)"
        }
    }
}
