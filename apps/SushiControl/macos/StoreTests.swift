import SwiftUI
import Darwin

@main struct StoreTests {
    @MainActor static func main() async throws {
        if let endpoint = ProcessInfo.processInfo.environment["SUSHI_METRICS_FIXTURE_URL"] {
            try await metricsRefreshFixture(endpoint)
            return
        }
        let suite = "org.sushi.store.test." + UUID().uuidString
        let defaults = UserDefaults(suiteName:suite)!
        defer { defaults.removePersistentDomain(forName:suite) }
        let startupRoot = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:startupRoot) }
        let startupService = ManagedServer(label:"org.sushi.startup.store."+UUID().uuidString.lowercased(),directory:startupRoot.appendingPathComponent("config"),loginDirectory:startupRoot.appendingPathComponent("LoginAgents"))
        let startupDefaults = UserDefaults(suiteName:suite+".startup")!
        defer { startupDefaults.removePersistentDomain(forName:suite+".startup") }
        let startupStore = ServerStore(defaults:startupDefaults,automaticallyRefresh:false,service:startupService)
        precondition(!startupStore.startServerAtLogin && !startupStore.recoverServerAfterCrash)
        precondition(startupStore.sidebarVisible)
        startupStore.sidebarVisible = false
        let reopenedSidebar = ServerStore(defaults:startupDefaults,automaticallyRefresh:false,service:startupService)
        precondition(!reopenedSidebar.sidebarVisible)
        startupStore.sidebarVisible = true
        startupStore.endpoint = "http://127.0.0.1:1"; startupStore.binary = "/bin/sleep"
        startupStore.arguments = "[\"serve\",\"--port\",\"1\"]"
        startupStore.startServerAtLogin = true; startupStore.recoverServerAfterCrash = true
        let startupReopened = ServerStore(defaults:startupDefaults,automaticallyRefresh:false,service:startupService)
        precondition(startupReopened.startServerAtLogin && startupReopened.recoverServerAfterCrash)
        await startupStore.refresh()
        startupStore.saveServerStartup(); try await settle(startupStore)
        precondition(startupStore.error.isEmpty && FileManager.default.fileExists(atPath:startupService.loginPlistURL.path))
        let startupState = await startupService.snapshot(); precondition(!startupState.registered,"Saving startup unexpectedly launched a process")
        let refreshDeadline = Date().addingTimeInterval(15)
        while !startupStore.canRunOfflineMaintenance && Date() < refreshDeadline { try await Task.sleep(nanoseconds:50_000_000) }
        precondition(startupStore.canRunOfflineMaintenance)
        startupStore.startServerAtLogin = false; startupStore.recoverServerAfterCrash = false
        startupStore.saveServerStartup(); try await settle(startupStore)
        precondition(startupStore.error.isEmpty && !FileManager.default.fileExists(atPath:startupService.loginPlistURL.path),startupStore.error)
        precondition(SystemSnapshot.cpuPercent(previous:nil,current:[100,50,200,0]) == nil)
        precondition(SystemSnapshot.cpuPercent(previous:[100,50,200,0],current:[150,75,275,0]) == 50)
        precondition(SystemSnapshot.cpuPercent(previous:[1,1,1,1],current:[1,1,1,1]) == nil)
        precondition(SystemSnapshot.cpuPercent(previous:[UInt32.max-5,0,10,0],current:[4,0,20,0]) == 50)
        var pages = vm_statistics64_data_t()
        pages.free_count = 100; pages.speculative_count = 40; pages.wire_count = 10; pages.compressor_page_count = 5
        let memory = SystemSnapshot.memoryBytes(pages,pageSize:16384)
        precondition(memory.free == 100*16384 && memory.wired == 10*16384 && memory.compressed == 5*16384)
        let gpu = SystemSnapshot.gpuInfo(["model":"Apple test GPU","gpu-core-count":40,"PerformanceStatistics":["Device Utilization %":73]])
        precondition(gpu.name == "Apple test GPU" && gpu.cores == 40 && gpu.percent == 73)
        precondition(SystemSnapshot.gpuInfo([:]).percent == nil)
        precondition(SystemSnapshot.gpuInfo(["gpu-core-count":true,"PerformanceStatistics":["Device Utilization %":101]]).cores == nil)
        precondition(SystemSnapshot.gpuInfo(["PerformanceStatistics":["Device Utilization %":101]]).percent == nil)
        precondition(SystemSnapshot.gpuInfo(["PerformanceStatistics":["Device Utilization %":false]]).percent == nil)
        let machine = SystemSnapshot.capture()
        precondition(machine.physicalBytes > 0 && machine.uptime >= 0 && !machine.thermal.isEmpty)
        if let wired = machine.wiredBytes { precondition(wired >= 0 && wired <= machine.physicalBytes) }
        let setupSuite = "org.sushi.setup-tests."+UUID().uuidString
        let setupDefaults = UserDefaults(suiteName:setupSuite)!
        defer { setupDefaults.removePersistentDomain(forName:setupSuite) }
        let controllerStore = ServerStore(defaults:setupDefaults,automaticallyRefresh:false)
        controllerStore.online = true
        controllerStore.controllerState = try DesktopControllerState(["model":"current","desktop":["paused":true,"models":[]],"idle_controller":["state":"ready","active_requests":0]])
        precondition(controllerStore.state == "Paused")
        precondition(controllerStore.canModelAction("load-model",model:"current"))
        precondition(!controllerStore.canModelAction("models/rescan"))
        controllerStore.busy = true
        precondition(!controllerStore.canModelAction("load-model",model:"current"))
        controllerStore.busy = false
        controllerStore.controllerToken = "temporary-control-secret"
        precondition(controllerStore.controllerState == nil)
        precondition(!controllerStore.canModelAction("load-model",model:"current"))
        precondition(ServerStore(defaults:setupDefaults,automaticallyRefresh:false).controllerToken.isEmpty)
        let usagePreferences = ServerStore(defaults:setupDefaults,automaticallyRefresh:false)
        precondition(!usagePreferences.usageEnabled)
        usagePreferences.usageEnabled = true
        precondition(ServerStore(defaults:setupDefaults,automaticallyRefresh:false).usageEnabled)
        usagePreferences.usageEnabled = false
        let fileSuite = suite+".local-files"
        let fileDefaults = UserDefaults(suiteName:fileSuite)!
        defer { fileDefaults.removePersistentDomain(forName:fileSuite) }
        fileDefaults.set("http://127.0.0.1:1",forKey:"serverEndpoint")
        let fileControls = ServerStore(defaults:fileDefaults,automaticallyRefresh:false)
        precondition(!fileControls.canTrashLocalModels)
        await fileControls.refresh()
        precondition(fileControls.canTrashLocalModels)
        fileControls.owned = true; precondition(!fileControls.canTrashLocalModels)
        fileControls.owned = false
        fileControls.online = true; precondition(!fileControls.canTrashLocalModels)
        fileControls.online = false; fileControls.serviceRunning = true; precondition(!fileControls.canTrashLocalModels)
        fileControls.serviceRunning = false; fileControls.busy = true; precondition(!fileControls.canTrashLocalModels)
        let conversionPreferences = ServerStore(defaults:setupDefaults,automaticallyRefresh:false)
        conversionPreferences.conversionPlan.expertBits = 2.625
        let conversionReopened = ServerStore(defaults:setupDefaults,automaticallyRefresh:false)
        precondition(conversionReopened.conversionPlan.expertBits == 2.625)
        let freshSetup = ServerStore(defaults:setupDefaults,automaticallyRefresh:false)
        precondition(freshSetup.setupPresented)
        freshSetup.finishSetup()
        precondition(!ServerStore(defaults:setupDefaults,automaticallyRefresh:false).setupPresented)
        setupDefaults.removeObject(forKey:"setupCompleted")
        setupDefaults.set("http://127.0.0.1:12345",forKey:"serverEndpoint")
        precondition(!ServerStore(defaults:setupDefaults,automaticallyRefresh:false).setupPresented)
        let discoveryDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:discoveryDir,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:discoveryDir) }
        let discoveryCLI = discoveryDir.appendingPathComponent("sushi")
        try Data("fixture".utf8).write(to:discoveryCLI)
        precondition(SushiDiscovery.candidates(paths:[discoveryCLI.path]).isEmpty)
        try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:discoveryCLI.path)
        let discoveryLink = discoveryDir.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at:discoveryLink,withDestinationURL:discoveryCLI)
        precondition(SushiDiscovery.candidates(paths:[discoveryCLI.path,discoveryLink.path,discoveryDir.path]).count == 1)
        let saved = ConnectionProfile(name:"Saved",endpoint:"http://127.0.0.1:12345",binary:"",arguments:"[\"serve\"]")
        defaults.set(try JSONEncoder().encode([saved]),forKey:"connectionProfiles")
        defaults.set(saved.id,forKey:"selectedProfile")
        let store = ServerStore(defaults:defaults,automaticallyRefresh:false)
        precondition(store.profileID == saved.id)
        defaults.removeObject(forKey:"menuBarMetric")
        defaults.set(true,forKey:"showActivity")
        let migratedMenu = ServerStore(defaults:defaults,automaticallyRefresh:false)
        precondition(migratedMenu.menuBarMetric == "Requests")
        store.menuBarMetric = "Output"; store.menuShowGraph = false
        store.menuMetricNames = ["Output","GPU"]
        precondition(store.menuSecondaryMetrics == ["GPU"])
        precondition(!ServerStore.menuOptionalMetricOptions.contains("Output"))
        store.menuMetricNames = ["Requests","Output","Cache reuse","GPU"]
        precondition(store.menuSecondaryMetrics == ["Requests","Cache reuse","GPU"])
        store.menuMetricNames = ["Output","GPU"]
        let menuReopened = ServerStore(defaults:defaults,automaticallyRefresh:false)
        precondition(menuReopened.menuBarMetric == "Output" && !menuReopened.menuShowGraph && menuReopened.menuMetricNames == ["Output","GPU"])
        store.menuBarMetric = "invalid"
        precondition(store.menuBarMetric == "Output")
        precondition(store.menuBarText.isEmpty)
        let menuValues = ServerStore(defaults:defaults,automaticallyRefresh:false)
        menuValues.online = true; menuValues.metrics = ["gauges":["requests_running":2]]
        menuValues.props = ["memory":["active_bytes":3.0*1073741824]]
        let menuTime = Date()
        menuValues.history.record(time:menuTime,tokens:0,memoryBytes:nil)
        menuValues.history.record(time:menuTime.addingTimeInterval(3),tokens:60,memoryBytes:nil)
        precondition(menuValues.menuBarText == " 20.0 tok/s")
        menuValues.metricsTemporarilyUnavailable = true
        precondition(menuValues.outputRate == "—")
        menuValues.metricsTemporarilyUnavailable = false
        menuValues.menuBarMetric = "Requests"; precondition(menuValues.menuBarText == " 2")
        menuValues.menuBarMetric = "GPU"; precondition(menuValues.menuBarText == " 3.0 GiB")
        menuValues.backendSleeping = true; precondition(menuValues.menuBarText.isEmpty && menuValues.menuValue("GPU") == "—")
        menuValues.toggleMenuMetric("Output"); precondition(!menuValues.menuMetricNames.contains("Output"))
        menuValues.menuMetricNames = ["invalid"]; precondition(menuValues.menuMetricNames == ["GPU"])
        store.refreshInterval = 2; store.showDockIcon = true
        store.refreshInterval = .infinity
        precondition(store.refreshInterval == 2)
        let appearanceReopened = ServerStore(defaults:defaults,automaticallyRefresh:false)
        precondition(appearanceReopened.refreshInterval == 2 && appearanceReopened.showDockIcon)
        store.toggleFavorite("favorite-model")
        store.menuFavoritesOnly = true; store.librarySort = "Context"
        let favoriteReopened = ServerStore(defaults:defaults,automaticallyRefresh:false)
        precondition(favoriteReopened.favoriteModelIDs == ["favorite-model"] && favoriteReopened.menuFavoritesOnly && favoriteReopened.librarySort == "Context")
        favoriteReopened.models = [ModelEntry(id:"chosen-model",loaded:false,state:"unloaded",architecture:"",quantization:"",contextTokens:nil,residentBytes:nil),ModelEntry(id:"favorite-model",loaded:false,state:"unloaded",architecture:"",quantization:"",contextTokens:nil,residentBytes:nil),ModelEntry(id:"loaded-model",loaded:true,state:"ready",architecture:"",quantization:"",contextTokens:nil,residentBytes:nil)]
        favoriteReopened.selected = "chosen-model"
        precondition(Set(favoriteReopened.menuModels.map(\.id)) == ["chosen-model","favorite-model","loaded-model"])
        store.toggleFavorite("favorite-model")
        precondition((defaults.stringArray(forKey:"favoriteModels") ?? []).isEmpty)
        store.endpoint = "http://127.0.0.1:9000"
        store.saveConnection()
        let reopened = ServerStore(defaults:defaults,automaticallyRefresh:false)
        precondition(reopened.endpoint == "http://127.0.0.1:9000" && reopened.profileID.isEmpty)
        store.arguments = "invalid"
        store.endpoint = "http://127.0.0.1:8000"
        store.saveConnection()
        precondition(!store.error.isEmpty && defaults.string(forKey:"serverEndpoint") == "http://127.0.0.1:9000")
        store.models = [ModelEntry(id:"offline-model",loaded:true,state:"loaded",architecture:"",quantization:"",contextTokens:nil,residentBytes:nil)]
        store.selected = "offline-model"
        await store.refresh()
        precondition(!store.online && store.models.isEmpty && store.selected.isEmpty)
        try store.setToolBinary("/bin/echo")
        let toolReopened = ServerStore(defaults:defaults,automaticallyRefresh:false)
        precondition(toolReopened.toolBinary == "/bin/echo" && toolReopened.binary.isEmpty)
        do { try store.setToolBinary("/missing/sushi"); fatalError("Invalid CLI accepted") }
        catch { precondition(store.toolBinary == "/bin/echo") }
        let logDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:logDirectory,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:logDirectory) }
        let logService = ManagedServer(label:"org.sushi.log.test",directory:logDirectory)
        let logStore = ServerStore(defaults:defaults,automaticallyRefresh:false,service:logService)
        try (String(repeating:"old line\n",count:20000)+"latest output").write(to:logService.logURL,atomically:true,encoding:.utf8)
        logStore.readLog()
        precondition(logStore.logError.isEmpty && logStore.logText.utf8.count == 100000 && logStore.logText.hasSuffix("latest output"))
        try FileManager.default.removeItem(at:logService.logURL)
        logStore.readLog()
        precondition(logStore.logText.isEmpty && logStore.logError.contains(logService.logURL.path))
        let stderrURL = logDirectory.appendingPathComponent("stderr.log")
        let sourceURL = logDirectory.appendingPathComponent("agent.plist")
        let agent:[String:Any] = ["Label":"org.sushi.log.test","ProgramArguments":["/bin/sleep","1"],"StandardOutPath":logService.logURL.path,"StandardErrorPath":stderrURL.path]
        try PropertyListSerialization.data(fromPropertyList:agent,format:.xml,options:0).write(to:sourceURL)
        try "server diagnostic".write(to:stderrURL,atomically:true,encoding:.utf8)
        let externalLogs = ServerStore(defaults:defaults,automaticallyRefresh:false,service:try ManagedServer.existingAgent(sourceURL))
        externalLogs.logStream = "error"; externalLogs.readLog()
        precondition(externalLogs.logURL == stderrURL && externalLogs.logText == "server diagnostic")
        precondition(externalLogs.state == "Stopped")
        externalLogs.owned = true; externalLogs.serviceRunning = true
        precondition(externalLogs.state == "Connecting")
        try store.tool.run(binary:"/bin/echo",arguments:["update"],title:"Runtime update",timeout:5)
        store.start()
        precondition(!store.busy, "Server start overlapped runtime update")
        store.modelAction("load-model",model:"example")
        precondition(!store.busy, "Model load overlapped runtime update")
        let taskDeadline = Date().addingTimeInterval(5)
        while store.tool.running && Date() < taskDeadline { try await Task.sleep(nanoseconds:50_000_000) }
        precondition(!store.tool.running)
        try store.tool.run(binary:"/bin/echo",arguments:["convert","fixture"],title:"Conversion guard",timeout:5)
        precondition(store.tool.exclusiveServerUse)
        store.modelAction("load-model",model:"example")
        precondition(!store.busy,"Model load overlapped conversion")
        let conversionDeadline = Date().addingTimeInterval(5)
        while store.tool.running && Date() < conversionDeadline { try await Task.sleep(nanoseconds:50_000_000) }
        precondition(!store.tool.running && !store.tool.exclusiveServerUse)
        if CommandLine.arguments.count > 2 { try await lifecycle(binary:CommandLine.arguments[1],port:CommandLine.arguments[2]) }
        defaults.set(false,forKey:"showDockIcon")
        _ = NSApplication.shared
        let delegate = AppDelegate()
        let previousMenu = NSApp.mainMenu, previousHelp = NSApp.helpMenu, previousWindows = NSApp.windowsMenu
        delegate.configureMenus()
        let edit = NSApp.mainMenu?.items.first{$0.submenu?.title == "Edit"}?.submenu
        let selectAll = edit?.items.first{$0.keyEquivalent == "a"}
        let editor = NSTextView(); editor.string = "Select the complete source path"
        precondition(NSApp.sendAction(selectAll!.action!,to:editor,from:selectAll))
        precondition(editor.selectedRange() == NSRange(location:0,length:editor.string.utf16.count))
        let close = NSApp.mainMenu?.items.first{$0.submenu?.title == "File"}?.submenu?.items.first{$0.keyEquivalent == "w"}
        precondition(close?.action == #selector(NSWindow.performClose(_:)) && close?.target == nil)
        precondition(NSApp.windowsMenu?.items.contains{$0.keyEquivalent == "m" && $0.action == #selector(NSWindow.performMiniaturize(_:))} == true)
        NSApp.mainMenu = previousMenu; NSApp.helpMenu = previousHelp; NSApp.windowsMenu = previousWindows
        precondition(delegate.popover.delegate === delegate)
        let focusWindow = FocusWindow()
        delegate.window = focusWindow
        focusWindow.shown = true
        delegate.popoverDidClose(Notification(name:NSPopover.didCloseNotification,object:delegate.popover))
        precondition(focusWindow.keyRequests == 1,"Popover dismissal did not restore the visible control centre")
        focusWindow.shown = false
        delegate.popoverDidClose(Notification(name:NSPopover.didCloseNotification,object:delegate.popover))
        precondition(focusWindow.keyRequests == 1,"Popover dismissal focused a hidden control centre")
        delegate.window = nil
        let withDelegate = ServerStore(defaults:defaults,automaticallyRefresh:false)
        precondition(!withDelegate.showDockIcon)
        let previousAppearance = NSApp.appearance
        for (selection,name) in [("light",NSAppearance.Name.aqua),("dark",NSAppearance.Name.darkAqua)] {
            withDelegate.appearance = selection; withDelegate.applyAppearance()
            precondition(delegate.popover.appearance?.name == name,"Menu appearance did not follow the selected app theme")
        }
        withDelegate.appearance = "system"; withDelegate.applyAppearance()
        precondition(delegate.popover.appearance == nil)
        NSApp.appearance = previousAppearance
        AppDelegate.instance = nil
        withExtendedLifetime(delegate) {}
        print("Store regressions passed: profile reopening, invalid-save isolation and offline model cleanup")
    }
    @MainActor static func lifecycle(binary:String,port:String) async throws {
        let id = UUID().uuidString.lowercased()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(id)
        let service = ManagedServer(label:"org.sushi.store.lifecycle."+id,directory:directory)
        let suite = "org.sushi.store.lifecycle."+id
        let defaults = UserDefaults(suiteName:suite)!
        defer { defaults.removePersistentDomain(forName:suite); try? FileManager.default.removeItem(at:directory) }
        let models = directory.appendingPathComponent("models")
        try FileManager.default.createDirectory(at:models,withIntermediateDirectories:true)
        let store = ServerStore(defaults:defaults,automaticallyRefresh:false,service:service)
        store.endpoint = "http://127.0.0.1:"+port
        store.binary = binary
        var args = ["serve","--host","127.0.0.1","--port",port,"--model-dir",models.path,"--metrics"]
        store.arguments = String(decoding:try JSONEncoder().encode(args),as:UTF8.self)
        store.saveConnection()
        do {
            store.start(); try await settle(store)
            precondition(store.error.isEmpty && store.owned)
            let client = Connection(endpoint:store.endpoint)
            try await ready(client)
            let enabled = try await client.fetch("metrics.json")
            precondition(enabled["gauges"] is [String:Any])
            let before = await service.snapshot()
            precondition(before.running)
            store.arguments = "invalid"
            store.stop(restart:true); try await settle(store)
            let retained = await service.snapshot()
            precondition(!store.error.isEmpty && retained.running)
            args.removeLast()
            store.arguments = String(decoding:try JSONEncoder().encode(args),as:UTF8.self)
            store.stop(restart:true); try await settle(store)
            precondition(store.error.isEmpty && store.owned)
            let plist = try PropertyListSerialization.propertyList(from:Data(contentsOf:service.plistURL),format:nil) as! [String:Any]
            precondition(plist["ProgramArguments"] as? [String] == [binary]+args)
            try await ready(client)
            do { _ = try await client.fetch("metrics.json"); throw ServerFailure(message:"Restart did not disable metrics") }
            catch { guard error.localizedDescription.contains("503") else { throw error } }
            store.stop(restart:false); try await settle(store)
            let remaining = await service.registered(); precondition(!remaining)
            let sourceBytes = try Data(contentsOf:service.plistURL)
            defaults.set(service.plistURL.path,forKey:"existingLaunchAgent")
            let externalStore = ServerStore(defaults:defaults,automaticallyRefresh:false)
            precondition(externalStore.usesExistingAgent && externalStore.canStart)
            externalStore.arguments = "invalid"
            externalStore.binary = ""
            externalStore.start(); try await settle(externalStore)
            precondition(externalStore.error.isEmpty && externalStore.owned)
            try await ready(client)
            externalStore.stop(restart:true); try await settle(externalStore)
            precondition(externalStore.error.isEmpty && externalStore.owned)
            try await ready(client)
            externalStore.stop(restart:false); try await settle(externalStore)
            let externalRemaining = await service.registered(); precondition(!externalRemaining)
            let finalBytes = try Data(contentsOf:service.plistURL); precondition(sourceBytes == finalBytes)
        } catch {
            let log = (try? String(contentsOf:service.logURL,encoding:.utf8)) ?? "No server log"
            let launchState = (try? await service.command(["print",service.service]).text) ?? "No launchd state"
            if await service.registered() { try? await service.stop() }
            throw ServerFailure(message:error.localizedDescription + "\n" + String(log.suffix(8000)) + "\n" + launchState)
        }
        print("Store lifecycle passed: invalid restart preserves service; updated arguments applied")
    }
    @MainActor static func metricsRefreshFixture(_ endpoint:String) async throws {
        let suite = "org.sushi.metrics.test." + UUID().uuidString
        let defaults = UserDefaults(suiteName:suite)!
        defer { defaults.removePersistentDomain(forName:suite) }
        defaults.set(endpoint,forKey:"serverEndpoint")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let service = ManagedServer(label:"org.sushi.metrics.test."+UUID().uuidString.lowercased(),directory:root.appendingPathComponent("config"),loginDirectory:root.appendingPathComponent("login"))
        let store = ServerStore(defaults:defaults,automaticallyRefresh:false,service:service)
        await store.refresh(); await store.refresh()
        precondition(store.online && !store.metricsTemporarilyUnavailable && store.history.samples.count == 1 && store.prefillHistory.samples.count == 1)
        await store.refresh()
        precondition(store.online && store.metricsTemporarilyUnavailable && store.metrics.isEmpty)
        precondition(store.history.samples.count == 1 && store.prefillHistory.samples.count == 1 && store.outputRate == "—" && store.prefillRate == "—")
        await store.refresh()
        precondition(!store.metricsTemporarilyUnavailable && store.history.samples.count == 1 && store.outputRate == "—")
        await store.refresh()
        precondition(store.history.samples.count == 2 && store.history.samples.last!.startsNewSegment && store.prefillHistory.samples.last!.startsNewSegment)
        await store.refresh()
        precondition(!store.metricsTemporarilyUnavailable && store.metrics.isEmpty && store.outputRate == "—")
        print("Metrics refresh passed: transient failure preserves chart history, resumes with a gap and recognizes disabled metrics")
    }
    static func ready(_ client:Connection) async throws {
        let deadline = Date().addingTimeInterval(20)
        while Date() < deadline {
            if (try? await client.fetch("health")) != nil { return }
            try await Task.sleep(nanoseconds:100_000_000)
        }
        throw ServerFailure(message:"Private server did not become ready")
    }
    @MainActor static func settle(_ store:ServerStore) async throws {
        let deadline = Date().addingTimeInterval(45)
        while store.busy && Date() < deadline { try await Task.sleep(nanoseconds:50_000_000) }
        guard !store.busy else { throw ServerFailure(message:"Store operation timed out") }
    }
}

@MainActor final class FocusWindow:NSWindow {
    var shown = false
    var keyRequests = 0
    override var isVisible:Bool { shown }
    override func makeKey() { keyRequests += 1 }
}
