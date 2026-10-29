import SwiftUI
import AppKit
import ServiceManagement
import Charts
import Combine


struct Actions:View {
    @ObservedObject var store:ServerStore
    @ObservedObject private var task:ToolTask
    var compact:Bool
    init(store:ServerStore,compact:Bool = false) { self.store = store; self.task = store.tool; self.compact = compact }
    @State private var confirmation = false
    @State private var restart = false
    var body:some View {
        HStack {
            if !compact || (!store.online && store.canStart) {
                Button("Start",systemImage:"play.fill") { store.start() }.disabled(store.online || (store.owned && store.serviceRunning) || !store.canStart).help(!store.canStart ? "Choose a Sushi executable in Server settings first." : "Start the \(store.managedDescription).")
            }
            if !compact || store.owned {
                Button("Restart",systemImage:"arrow.clockwise") { restart = true; confirmation = true }.disabled(!store.owned).help("Restart the \(store.managedDescription).")
                Button("Stop",systemImage:"stop.fill") { restart = false; confirmation = true }.disabled(!store.owned)
            }
        }.disabled(store.busy || task.exclusiveServerUse)
        .confirmationDialog(restart ? "Restart the \(store.managedDescription)?" : "Stop the \(store.managedDescription)?",isPresented:$confirmation,titleVisibility:.visible) {
            Button(restart ? "Restart Server" : "Stop Server",role:.destructive) { store.stop(restart:restart) }
        } message: { Text("This ends any active requests for the selected server.") }
    }
}
struct ConnectionBadge:View {
    @ObservedObject var store:ServerStore
    var body:some View {
        HStack(spacing:6) { Circle().fill(store.statusColor).frame(width:7,height:7); Text(store.state).font(.caption.weight(.medium)) }
            .padding(.horizontal,10).padding(.vertical,6).background(store.statusColor.opacity(0.1),in:Capsule())
    }
}
struct Feedback:View {
    var text:String
    var failure = false
    var body:some View {
        HStack(alignment:.top,spacing:10) {
            Image(systemName:failure ? "exclamationmark.circle" : "info.circle").foregroundStyle(failure ? Color.orange : Color.sushiAccent)
            Text(text).font(.callout).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
        }.padding(12).background((failure ? Color.orange : Color.sushiAccent).opacity(0.08),in:RoundedRectangle(cornerRadius:10))
    }
}
private struct MenuPanelContentHeight:PreferenceKey {
    static var defaultValue:CGFloat { 0 }
    static func reduce(value:inout CGFloat,nextValue:()->CGFloat) { value = max(value,nextValue()) }
}
private struct MenuPanelHeaderHeight:PreferenceKey {
    static var defaultValue:CGFloat { 0 }
    static func reduce(value:inout CGFloat,nextValue:()->CGFloat) { value = max(value,nextValue()) }
}
private struct MenuPanelFooterHeight:PreferenceKey {
    static var defaultValue:CGFloat { 0 }
    static func reduce(value:inout CGFloat,nextValue:()->CGFloat) { value = max(value,nextValue()) }
}
private struct MenuGroupSurface:ViewModifier {
    func body(content:Content)->some View {
        content.background(Color.primary.opacity(0.045),in:RoundedRectangle(cornerRadius:12))
            .overlay(RoundedRectangle(cornerRadius:12).strokeBorder(Color.primary.opacity(0.06)))
    }
}
enum MenuPanelSection:String,CaseIterable,Identifiable {
    case overview = "Overview", models = "Models", server = "Server"
    var id:String { rawValue }
    var icon:String { switch self { case .overview:"waveform.path"; case .models:"square.stack.3d.up"; case .server:"network" } }
}
struct MenuPanel:View {
    @ObservedObject var store:ServerStore
    var maximumHeight:CGFloat = 650
    var preferredHeightChanged:(CGFloat)->Void = { _ in }
    @State private var section:MenuPanelSection = .overview
    @State private var contentHeight:CGFloat = 350
    @State private var sectionHeight:CGFloat = 200
    @State private var headerHeight:CGFloat = 90
    @State private var footerHeight:CGFloat = 60
    var menuSelection:String { ModelEntry.selection(store.selected,in:store.menuModels) }
    var body:some View {
        VStack(spacing:0) {
            VStack(alignment:.leading,spacing:12) {
                HStack { Text("Sushi Control").font(.headline); Spacer(); ConnectionBadge(store:store) }
                Picker("Menu section",selection:$section) {
                    ForEach(MenuPanelSection.allCases) { item in Label(item.rawValue,systemImage:item.icon).tag(item) }
                }.pickerStyle(.segmented).labelsHidden().frame(maxWidth:.infinity).accessibilityLabel("Menu section")
            }.padding(.horizontal,16).padding(.top,16).padding(.bottom,12)
                .background(GeometryReader { geometry in Color.clear.preference(key:MenuPanelHeaderHeight.self,value:geometry.size.height) })
            Divider()
            SushiScrollView {
                VStack(alignment:.leading,spacing:14) {
                    switch section {
                    case .overview: overview
                    case .models: modelControls
                    case .server: serverControls
                    }
                    if store.busy { HStack { ProgressView().controlSize(.small); Text(store.operation).font(.caption) } }
                    if !store.error.isEmpty { Feedback(text:store.error,failure:true) }
                    if !store.controllerError.isEmpty { Feedback(text:store.controllerError,failure:true) }
                    if !store.online && !store.connectionError.isEmpty { Text(store.connectionError).font(.caption).foregroundStyle(.secondary) }
                }.padding(16).frame(maxWidth:.infinity,alignment:.leading)
                    .background(GeometryReader { geometry in Color.clear.preference(key:MenuPanelContentHeight.self,value:geometry.size.height) })
            }
            Divider()
            HStack(spacing:8) {
                Button { AppDelegate.instance?.showCenter() } label: { Label("Control Center",systemImage:"slider.horizontal.3").frame(maxWidth:.infinity) }
                    .keyboardShortcut(",").help("Open the main window")
                Button { store.dashboard() } label: { Image(systemName:"arrow.up.right.square") }
                    .disabled(!store.online).help("Open Dashboard").accessibilityLabel("Open Dashboard")
                Button { store.refreshNow() } label: { Image(systemName:"arrow.clockwise") }
                    .disabled(store.checking || store.busy).help("Refresh status").accessibilityLabel("Refresh status")
                Button { NSApp.terminate(nil) } label: { Image(systemName:"power") }
                    .disabled(store.busy).help("Quit Sushi Control").accessibilityLabel("Quit Sushi Control")
            }.buttonStyle(.bordered).controlSize(.regular).padding(.horizontal,16).padding(.vertical,12)
                .background(GeometryReader { geometry in Color.clear.preference(key:MenuPanelFooterHeight.self,value:geometry.size.height) })
        }
        .tint(.sushiAccent).frame(width:390,height:min(maximumHeight,contentHeight))
        .onPreferenceChange(MenuPanelContentHeight.self) { height in if height > 0 { sectionHeight = height; updateHeight() } }
        .onPreferenceChange(MenuPanelHeaderHeight.self) { height in if height > 0 { headerHeight = height; updateHeight() } }
        .onPreferenceChange(MenuPanelFooterHeight.self) { height in if height > 0 { footerHeight = height; updateHeight() } }
        .background(Color(nsColor:.windowBackgroundColor))
    }
    private func updateHeight() {
        let next = min(maximumHeight,max(220,headerHeight+sectionHeight+footerHeight+2))
        if abs(contentHeight-next) > 0.5 { contentHeight = next; preferredHeightChanged(next) }
    }
    private var overview:some View {
        VStack(alignment:.leading,spacing:14) {
            VStack(alignment:.leading,spacing:6) {
                Text("MENU SELECTION").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                if let model = store.selectedModel {
                    Text(model.id).font(.subheadline.weight(.semibold)).lineLimit(2).help(model.id).accessibilityLabel("Menu selection: " + model.id)
                    Label(model.loaded ? "Loaded" : "Not loaded",systemImage:model.loaded ? "checkmark.circle.fill" : "circle")
                        .font(.caption).foregroundStyle(model.loaded ? Color.green : Color.secondary)
                } else { Text("No model selected").font(.subheadline).foregroundStyle(.secondary) }
            }.frame(maxWidth:.infinity,alignment:.leading).padding(12).modifier(SushiControlSurface())
            HStack(alignment:.top,spacing:16) {
                metric("Output",store.outputRate == "—" ? "— tok/s" : store.outputRate)
                metric("Prefill",store.prefillRate == "—" ? "— tok/s" : store.prefillRate)
            }
            if !store.menuSecondaryMetrics.isEmpty {
                LazyVGrid(columns:Array(repeating:GridItem(.flexible(),alignment:.leading),count:min(3,store.menuSecondaryMetrics.count)),alignment:.leading,spacing:12) {
                    ForEach(store.menuSecondaryMetrics,id:\.self) { name in metric(name,store.menuValue(name)) }
                }
            }
            if store.menuShowGraph { MenuPerformancePanel(store:store) }
        }
    }
    private var modelControls:some View {
        VStack(alignment:.leading,spacing:12) {
            Text("MODEL CONTROLS").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            if store.online && !store.menuModels.isEmpty {
                VStack(alignment:.leading,spacing:12) {
                    Picker("Model",selection:Binding(get:{ menuSelection },set:{ store.selected = $0 })) {
                        ForEach(store.menuModels) { Text(($0.loaded ? "● " : store.favoriteModelIDs.contains($0.id) ? "★ " : "") + $0.id).tag($0.id) }
                    }.disabled(store.busy).help("Choose which model the Load and Unload buttons act on.")
                    Text(store.controllerState == nil ? "Load makes this the default for requests without a model." : "Load makes this the controller’s active model.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
                    HStack {
                        Button("Load") { store.modelAction("load-model",model:menuSelection) }
                            .buttonStyle(SushiPrimaryButtonStyle()).disabled(!store.canModelAction("load-model",model:menuSelection))
                            .help(store.controllerState == nil ? "Load this model and make it the default for requests that don’t name a model." : "Load this model and make it active in the controller.")
                        Button("Unload") { store.modelAction("unload-model",model:menuSelection) }.buttonStyle(.bordered)
                            .disabled(store.menuModels.first(where:{$0.id == menuSelection})?.loaded != true || !store.canModelAction("unload-model",model:menuSelection))
                            .help("Unload the chosen model after any active requests finish.")
                    }.disabled(store.busy || menuSelection.isEmpty)
                }.padding(14).frame(maxWidth:.infinity,alignment:.leading).modifier(MenuGroupSurface())
            } else {
                Text(store.online ? "No models are available in the menu." : "Connect to a server to manage models.").font(.callout).foregroundStyle(.secondary)
            }
            Button("Open Model Library",systemImage:"square.stack.3d.up") { store.page = .models; AppDelegate.instance?.showCenter() }.buttonStyle(.bordered)
        }
    }
    private var serverControls:some View {
        VStack(alignment:.leading,spacing:12) {
            Text("SERVER CONTROLS").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            VStack(alignment:.leading,spacing:12) {
                Text("Local address").font(.callout.weight(.medium))
                Text(store.connectedEndpoint).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).help(store.connectedEndpoint)
                if let controller = store.controllerState {
                    Divider()
                    Button(controller.paused ? "Resume Requests" : "Pause New Requests",systemImage:controller.paused ? "play.circle" : "pause.circle") { store.controllerAction(controller.paused ? "resume" : "pause") }
                        .disabled(store.busy || store.tool.exclusiveServerUse)
                    if controller.paused { Text("New requests paused; existing work can finish.").font(.caption).foregroundStyle(.secondary) }
                }
                if store.owned || (!store.online && store.canStart) {
                    Divider()
                    Actions(store:store,compact:true).buttonStyle(.bordered)
                } else if store.online {
                    Text("Managed outside Sushi Control").font(.caption).foregroundStyle(.secondary)
                }
            }.padding(14).frame(maxWidth:.infinity,alignment:.leading).modifier(MenuGroupSurface())
            Button("Server Settings",systemImage:"gearshape") { store.page = .server; AppDelegate.instance?.showCenter() }.buttonStyle(.bordered)
        }
    }
    private func metric(_ title:String,_ value:String)->some View {
        VStack(alignment:.leading,spacing:4) { Text(title).font(.caption).foregroundStyle(.secondary); Text(value).font(.headline).monospacedDigit().lineLimit(1).minimumScaleFactor(0.75) }.frame(maxWidth:.infinity,alignment:.leading)
    }
}

enum Page:String,CaseIterable,Identifiable {
    case status = "Status", activity = "Activity", models = "Models", downloads = "Downloads", integrations = "Integrations", runtime = "Runtime", server = "Server", logs = "Logs", diagnostics = "Diagnostics", general = "General", benchmarks = "Benchmarks", conversion = "Conversion"
    var id:String { rawValue }
    var icon:String { switch self { case .conversion:"arrow.triangle.2.circlepath"; case .benchmarks:"speedometer"; case .status:"gauge.with.dots.needle.50percent"; case .activity:"waveform.path"; case .models:"square.stack.3d.up"; case .downloads:"arrow.down.circle"; case .integrations:"link"; case .runtime:"shippingbox"; case .server:"network"; case .logs:"doc.text"; case .diagnostics:"stethoscope"; case .general:"gearshape" } }
    var subtitle:String { switch self { case .conversion:"Create Sushi packs with the native EXL3 converter."; case .benchmarks:"Measure prompt processing, generation and cache reuse."; case .status:"Your local inference server, at a glance."; case .activity:"Live requests and cached conversations."; case .models:"Manage the models registered with Sushi."; case .downloads:"Download models using your Sushi installation."; case .integrations:"Connect your tools to local models."; case .runtime:"Check and maintain your Sushi installation."; case .server:"Connect to Sushi or launch your own server."; case .logs:"Output from the selected server launcher."; case .diagnostics:"Check your connection and server endpoints."; case .general:"Make Sushi Control feel at home on your Mac." } }
}
struct ControlCenter:View {
    @ObservedObject var store:ServerStore
    var page:Page? { store.page }
    @State private var loginEnabled = SMAppService.mainApp.status == .enabled
    @State private var search = ""
    @State private var advanced = false
    @State private var launch = LaunchSettings()
    @State private var showModelSettings = false
    @State private var showLocalFiles = false
    @State private var modelFilter = "All"
    @State private var logSearch = ""
    @State private var removeProfile = false
    private var sidebarVisibility:Binding<NavigationSplitViewVisibility> {
        Binding(get:{ store.sidebarVisible ? .all : .detailOnly },set:{ value in
            if value == .all { store.sidebarVisible = true }
            else if value == .detailOnly { store.sidebarVisible = false }
        })
    }
    var filteredModels:[ModelEntry] { ModelEntry.library(store.models,query:search,scope:modelFilter,favorites:store.favoriteModelIDs,sort:store.librarySort) }
    var body:some View {
        NavigationSplitView(columnVisibility:sidebarVisibility) {
            VStack(spacing:0) {
                HStack(spacing:10) { VStack(alignment:.leading) { Text("Sushi Control").font(.headline); Text("Version 0.1").font(.caption2).foregroundStyle(.secondary) }; Spacer() }.padding(18)
                List(selection:$store.page) {
                    Section("Workspace") { ForEach([Page.status,.activity,.models,.downloads]) { item in Label(item.rawValue,systemImage:item.icon).padding(.vertical,1).listItemTint(.monochrome).tag(item) } }
                    Section("Tools") { ForEach([Page.integrations,.benchmarks,.conversion]) { item in Label(item.rawValue,systemImage:item.icon).padding(.vertical,1).listItemTint(.monochrome).tag(item) } }
                    Section("Server & App") { ForEach([Page.server,.runtime,.logs,.diagnostics,.general]) { item in Label(item.rawValue,systemImage:item.icon).padding(.vertical,1).listItemTint(.monochrome).tag(item) } }
                }.listStyle(.sidebar)
                Divider()
                HStack { ConnectionBadge(store:store); Spacer() }.padding(14)
            }.frame(width:210).navigationSplitViewColumnWidth(210).toolbar(removing:.sidebarToggle)
        } detail: {
            VStack(alignment:.leading,spacing:0) {
                HStack(alignment:.top) {
                    VStack(alignment:.leading,spacing:6) { Text(page?.rawValue ?? "Status").font(.system(size:26,weight:.semibold)).tracking(-0.5).accessibilityAddTraits(.isHeader); Text((page ?? .status).subtitle).foregroundStyle(.secondary) }
                    Spacer()
                    if store.checking || store.busy { ProgressView().controlSize(.small).padding(.top,8) }
                    Button { store.refreshNow() } label: { Image(systemName:"arrow.clockwise") }.help("Refresh server status").disabled(store.checking || store.busy)
                }.padding(20)
                SushiScrollView {
                    VStack(alignment:.leading,spacing:16) {
                        if !store.error.isEmpty { HStack(alignment:.top) { Feedback(text:store.error,failure:true); Button { store.error = "" } label: { Image(systemName:"xmark") }.buttonStyle(.borderless).help("Dismiss error").accessibilityLabel("Dismiss error") } }
                        if store.busy { HStack(spacing:10) { ProgressView().controlSize(.small); Text(store.operation).foregroundStyle(.secondary) } }
                        else if !store.notice.isEmpty {
                            HStack(alignment:.top) {
                                Feedback(text:store.notice)
                                Button { store.notice = "" } label: { Image(systemName:"xmark") }.buttonStyle(.borderless).accessibilityLabel("Dismiss notification")
                            }.task(id:store.notice) {
                                let message = store.notice
                                guard !NSWorkspace.shared.isVoiceOverEnabled else { return }
                                do { try await Task.sleep(for:.seconds(10)) } catch { return }
                                if store.notice == message { store.notice = "" }
                            }
                        }
                        content
                    }.padding(.horizontal,20).padding(.bottom,20)
                        .frame(maxWidth:1280,alignment:.leading)
                        .frame(maxWidth:.infinity,alignment:.top)
                }
            }.background(Color(nsColor:.windowBackgroundColor))
        }.navigationSplitViewStyle(.balanced).tint(.sushiAccent).frame(minWidth:store.sidebarVisible ? 940 : 720,minHeight:680)
        .sheet(isPresented:$showModelSettings) { ModelSettingsPanel() }
        .sheet(isPresented:$showLocalFiles) { LocalModelFilesPanel(store:store,task:store.tool) }
        .confirmationDialog("Remove this connection profile?",isPresented:$removeProfile,titleVisibility:.visible) {
            Button("Remove Profile",role:.destructive) { store.deleteProfile() }
        } message: { Text("Its saved API key will also be removed. The server will keep running.") }
    }
    @ViewBuilder var content:some View {
        switch page ?? .status {
        case .conversion:
            ConversionPanel(store:store,task:store.tool)
        case .benchmarks:
            BenchmarkPanel(store:store,runner:store.benchmark)
        case .status:
            VStack(alignment:.leading,spacing:20) {
                if store.setupPresented { SetupGuide(store:store) }
                StatusDashboard(store:store)
            }
        case .activity:
            UsageHistoryPanel(store:store,history:store.usage)
            ActivityPanel(store:store)
        case .models:
            VStack(alignment:.leading,spacing:16) {
                HStack { Text("Per-model defaults for local Sushi installations").font(.caption).foregroundStyle(.secondary); Spacer(); Button("Manage Files…",systemImage:"folder") { showLocalFiles = true }; Button("Model Settings…",systemImage:"slider.horizontal.3") { showModelSettings = true } }
                HStack { TextField("Search models",text:$search).textFieldStyle(.roundedBorder); Button("Rescan",systemImage:"arrow.clockwise") { store.modelAction("models/rescan") }.disabled(!store.canModelAction("models/rescan")).help(store.controllerState != nil ? "The controller catalog refreshes automatically." : store.rescanAvailable ? "Scan the server’s configured model directory." : "This server does not expose model rescanning."); Button("Load Folder…",systemImage:"folder.badge.plus") { store.importModel() } }.disabled(!store.online || store.busy)
                if !store.online { empty("Server disconnected","Connect in Server settings to see its model library.","network") }
                else if filteredModels.isEmpty { empty(store.models.isEmpty ? "No models registered" : modelFilter == "Favorites" ? "No matching favorites" : modelFilter == "Loaded" ? "No loaded models" : "No matching models",store.models.isEmpty ? "Choose a model folder, or rescan the server’s configured model directory." : modelFilter == "Favorites" ? "Star models in All models to keep them here. Clear the search to see all your favorites." : "Try a different name or architecture.","square.stack.3d.up") }
                HStack {
                    Picker("Show",selection:$modelFilter) { Text("All models").tag("All"); Text("Loaded").tag("Loaded"); Text("Favorites").tag("Favorites") }.pickerStyle(.segmented).labelsHidden().frame(maxWidth:360)
                    Spacer()
                    Picker("Sort by",selection:$store.librarySort) { Text("Name").tag("Name"); Text("Memory").tag("Memory"); Text("Context").tag("Context") }.frame(width:190)
                    Text("\(filteredModels.count) \(filteredModels.count == 1 ? "model" : "models")").font(.caption).foregroundStyle(.secondary)
                }
                ForEach(filteredModels) { model in
                    SushiCard {
                        VStack(alignment:.leading,spacing:14) {
                            HStack(alignment:.top) { Image(systemName:"cube").font(.title2).foregroundStyle(.secondary); VStack(alignment:.leading,spacing:5) { Text(model.id).font(.headline).textSelection(.enabled).fixedSize(horizontal:false,vertical:true); Text(model.architecture.isEmpty ? model.state.capitalized : "\(model.architecture) · \(model.state.capitalized)").font(.caption).foregroundStyle(.secondary) }; Spacer(); Button { store.toggleFavorite(model.id) } label: { Image(systemName:store.favoriteModelIDs.contains(model.id) ? "star.fill" : "star") }.buttonStyle(.borderless).foregroundStyle(store.favoriteModelIDs.contains(model.id) ? Color.orange : Color.secondary).help(store.favoriteModelIDs.contains(model.id) ? "Remove favorite" : "Add favorite").accessibilityLabel("Favorite " + model.id).accessibilityValue(store.favoriteModelIDs.contains(model.id) ? "Favorite" : "Not a favorite"); if model.loaded { Label("Loaded",systemImage:"circle.fill").font(.caption).foregroundStyle(.green) } }
                            HStack(spacing:12) {
                                if let context = model.contextTokens { Label("\(context.formatted()) tokens",systemImage:"text.alignleft") }
                                if let bytes = model.residentBytes, bytes > 0 { Label(String(format:"%.1f GiB",bytes/1073741824),systemImage:"memorychip") }
                            }.font(.caption).foregroundStyle(.secondary)
                            if !model.quantization.isEmpty { Text(model.quantization).font(.caption).foregroundStyle(.secondary) }
                            DisclosureGroup("Model details") {
                                VStack(alignment:.leading,spacing:10) {
                                    if !model.capabilities.isEmpty { LabeledContent("Capabilities",value:model.capabilities.map { $0.replacingOccurrences(of:"_",with:" ").capitalized }.joined(separator:", ")) }
                                    if !model.inputModalities.isEmpty { LabeledContent("Inputs",value:model.inputModalities.map { $0.capitalized }.joined(separator:", ")) }
                                    if let maximum = model.maximumContext { LabeledContent("Model context ceiling",value:maximum.formatted()+" tokens") }
                                    if let bytes = model.diskBytes { LabeledContent("On disk",value:String(format:"%.1f GiB",bytes/1073741824)) }
                                    if let kv = model.kvFormat { LabeledContent("KV cache",value:kv) }
                                    if let loaded = model.drafterLoaded { LabeledContent("DFlash assistant",value:loaded ? "Loaded" : "Not loaded") }
                                    if let loaded = model.mtpLoaded { LabeledContent("MTP",value:loaded ? "Loaded" : "Not loaded") }
                                    if !model.reasoningEfforts.isEmpty { LabeledContent("Reasoning levels",value:model.reasoningEfforts.joined(separator:", ")) }
                                    if let value = model.recommendedTemperature { LabeledContent("Author temperature",value:value.formatted()) }
                                    if let value = model.recommendedTopP { LabeledContent("Author top-p",value:value.formatted()) }
                                    if let value = model.recommendedTopK { LabeledContent("Author top-k",value:value.formatted()) }
                                    if model.recommendedTemperature != nil || model.recommendedTopP != nil || model.recommendedTopK != nil { Text("Recommendations from generation_config.json. Request parameters and server launch defaults may override them.").foregroundStyle(.secondary) }
                                    if let effort = model.defaultReasoningEffort { LabeledContent("Default reasoning",value:effort) }
                                    Text("Reported by the server. Available details depend on the model and Sushi version.").foregroundStyle(.secondary)
                                }.font(.caption).padding(.top,8).textSelection(.enabled)
                            }
                            HStack { Button("Copy ID",systemImage:"doc.on.doc") { store.copyModelID(model.id) }; Spacer(); Button("Load",systemImage:"play.fill") { store.selected = model.id; store.modelAction("load-model",model:model.id) }.buttonStyle(.bordered).disabled(!store.canModelAction("load-model",model:model.id)).help(store.controllerState == nil ? "Load this model and make it the default for requests that don’t name a model." : "Load this model and make it active in the controller."); Button("Unload",systemImage:"eject") { store.modelAction("unload-model",model:model.id) }.disabled(!model.loaded || !store.canModelAction("unload-model",model:model.id)); Spacer(); if model.id == store.selected { Label("Selected in menu",systemImage:"checkmark.circle").font(.caption).foregroundStyle(.secondary) } else { Button("Select in Menu") { store.selected = model.id } } }
                        }
                    }.overlay(RoundedRectangle(cornerRadius:14).strokeBorder(model.loaded ? Color.sushiAccent.opacity(0.35) : .clear,lineWidth:1)).disabled(!store.online || store.busy)
                }
                Text(store.controllerState != nil ? "Switching models changes the controller’s saved selection and then loads it. Wait for active requests to finish first. Loading a model keeps paused admission paused." : "Setting a default applies to requests that omit a model. Requests naming another model keep using that model. Unloading may wait for the server to finish using the model.").font(.caption).foregroundStyle(.secondary)
            }
        case .integrations:
            IntegrationsPanel(store:store)
        case .server:
            VStack(alignment:.leading,spacing:20) {
                SushiCard("Server launcher") {
                    VStack(alignment:.leading,spacing:10) {
                        if store.usesExistingAgent {
                            Text(store.existingAgentPath).font(.caption.monospaced()).textSelection(.enabled)
                            Text("Start, Stop and Restart use this launch agent. Its configuration is preserved.").font(.caption).foregroundStyle(.secondary)
                            Button("Detach Launcher") { store.detachExistingAgent() }
                        } else {
                            Text("Use an existing launch agent, or configure a server started by Sushi Control.").font(.caption).foregroundStyle(.secondary)
                            Button("Use Existing Launch Agent…") { store.chooseExistingAgent() }.disabled(store.owned)
                        }
                    }.padding(12)
                }
                Picker("Settings",selection:$store.serverSection) { Text("Connection").tag("Connection"); Text("Launch & Performance").tag("Launch") }.pickerStyle(.segmented).labelsHidden().frame(maxWidth:360)
                if store.serverSection == "Connection" {
                SushiCard("Connection profiles") {
                    VStack(alignment:.leading,spacing:12) {
                        HStack {
                            Menu(store.profiles.first(where:{$0.id == store.profileID})?.name ?? "Choose a profile") { ForEach(store.profiles) { profile in Button(profile.name) { store.selectProfile(profile.id) } } }
                            Spacer(); Button("New") { store.newProfile() }
                            Button("Remove",role:.destructive) { removeProfile = true }.disabled(store.profileID.isEmpty)
                        }
                        TextField("Profile name",text:$store.profileName)
                        Toggle("Remember API key in Keychain",isOn:$store.rememberKey)
                        Button("Save Profile") { store.saveProfile() }
                    }
                }.disabled(store.owned || store.usesExistingAgent)
                DesktopControllerPanel(store:store)
                SushiCard("Connection") {
                    VStack(alignment:.leading,spacing:14) {
                        LabeledContent("Server URL") { TextField("http://127.0.0.1:12345",text:$store.endpoint).frame(minWidth:260) }
                        LabeledContent("API key") { SecureField("Optional",text:$store.apiKey).frame(minWidth:260) }
                        Text("Use a local server root without /v1. Sushi’s default port is 12345.").font(.caption).foregroundStyle(.secondary)
                    }
                }.disabled(store.owned || store.usesExistingAgent)
                } else {
                SushiCard("Start a server from this app") {
                    VStack(alignment:.leading,spacing:14) {
                        HStack { TextField("Choose your installed sushi executable",text:$store.binary); Button("Choose…") { store.chooseBinary() } }
                        GroupBox("Launch settings") {
                            VStack(spacing:14) {
                                LabeledContent("Port") { TextField("12345",value:$launch.port,format:.number.grouping(.never)).frame(width:110) }
                                LabeledContent("Context tokens") { TextField("Automatic",text:$launch.context).frame(width:160) }
                                LabeledContent("Resident models (GiB)") { TextField("Automatic",text:$launch.memoryGiB).frame(width:160) }
                                LabeledContent("Unload after (minutes)") { Stepper(value:$launch.idleMinutes,in:0...10080,step:5) { Text(launch.idleMinutes == 0 ? "Never" : String(launch.idleMinutes)).frame(width:80) } }
                                Picker("KV cache",selection:$launch.kvBits) { Text("8-bit").tag(8); Text("6-bit").tag(6); Text("4-bit").tag(4); Text("16-bit").tag(16) }
                                DisclosureGroup("Cache & acceleration") {
                                LabeledContent("RAM prefix cache (GiB)") { TextField("Default",text:$launch.ramCacheGiB).frame(width:160) }
                                LabeledContent("SSD prefix cache (GiB)") { TextField("Default",text:$launch.diskCacheGiB).frame(width:160) }
                                LabeledContent("Prefill chunk tokens") { TextField("Automatic",text:$launch.prefillChunk).frame(width:160) }
                                TextField("DFlash assistant folder (optional)",text:$launch.drafter)
                                }.padding(.vertical,4)
                                Toggle("Enable live metrics",isOn:$launch.metrics)
                                HStack { TextField("Models folder (optional)",text:$launch.modelDirectory); Button("Choose…") { let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; if panel.runModal() == .OK { launch.modelDirectory = panel.url?.path ?? "" } } }
                                Button("Use These Launch Settings") {
                                    do {
                                        let args = try launch.mergedArguments(store.arguments)
                                        store.arguments = String(decoding:try JSONSerialization.data(withJSONObject:args,options:[.prettyPrinted,.withoutEscapingSlashes]),as:UTF8.self)
                                        store.endpoint = "http://127.0.0.1:\(launch.port)"; advanced = true
                                        store.notice = "Launch settings prepared. Save to apply them."
                                    } catch { store.error = error.localizedDescription }
                                }
                            }.padding(10)
                        }
                        DisclosureGroup("Advanced arguments",isExpanded:$advanced) {
                            VStack(alignment:.leading,spacing:10) {
                                TextEditor(text:$store.arguments).font(.system(.body,design:.monospaced)).frame(height:120).padding(6).background(Color(nsColor:.textBackgroundColor),in:RoundedRectangle(cornerRadius:6)).overlay(RoundedRectangle(cornerRadius:6).stroke(.quaternary))
                                Text("Example: [\"serve\", \"--metrics\", \"--port\", \"12345\"]. Enter full model paths.").font(.caption).foregroundStyle(.secondary)
                            }.padding(.top,10)
                        }
                        Text(store.usesExistingAgent ? "Launch settings are controlled by the selected launch agent." : "Changes take effect the next time you start the server.").font(.caption).foregroundStyle(.secondary)
                    }.padding(12)
                }.disabled(store.usesExistingAgent)
                SushiCard("Server startup & recovery") {
                    Toggle("Start Sushi server when I log in",isOn:$store.startServerAtLogin)
                    Toggle("Restart server after a crash",isOn:$store.recoverServerAfterCrash)
                    Text("Separate from opening Sushi Control at login. Crash recovery restarts unsuccessful exits with a 30-second minimum interval. Stop removes the running job until you start it again or log in again. Quitting this app leaves the server running.").font(.caption).foregroundStyle(.secondary)
                    Button("Save Server Startup") { store.saveServerStartup() }.disabled(store.usesExistingAgent || !store.canRunOfflineMaintenance)
                    Text(store.usesExistingAgent ? "The existing launch agent controls its own startup and recovery policy." : "Stop the server before saving. Saving does not start it; Start and Restart also apply these choices.").font(.caption).foregroundStyle(.secondary)
                }.disabled(store.usesExistingAgent)
                }
                if store.usesExistingAgent {
                    Feedback(text:"Launch settings are read from the selected launch agent. Detach it to change the connection or configure a different launcher.")
                } else {
                    HStack { Button(store.owned ? "Save Changes" : "Save & Connect") { store.saveConnection() }.buttonStyle(SushiPrimaryButtonStyle()); Text(store.owned ? "Restart the server to apply saved launch changes." : "The server does not start automatically.").font(.caption).foregroundStyle(.secondary) }
                    if store.owned { Feedback(text:"Connection profiles stay locked while the app-managed server is registered. Launch changes apply on restart.") }
                }
            }.textFieldStyle(.roundedBorder).disabled(store.busy)
            .onAppear { restoreLaunchForm() }
            .onChange(of:store.profileID) { _,_ in restoreLaunchForm() }
            .onChange(of:store.serverSection) { _,section in if section == "Launch" { restoreLaunchForm() } }
        case .logs:
            VStack(alignment:.leading,spacing:14) {
                HStack { Button("Refresh",systemImage:"arrow.clockwise") { store.readLog() }; Button("Show in Finder",systemImage:"folder") { store.revealLogs() }; Spacer(); Text("Last 100 KB").font(.caption).foregroundStyle(.secondary) }
                Picker("Stream",selection:$store.logStream) { Text("Standard output").tag("output"); Text("Standard error").tag("error") }.frame(maxWidth:320).onChange(of:store.logStream) { _,_ in store.readLog() }
                Text(store.logURL.path).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                if !store.logError.isEmpty { Feedback(text:store.logError,failure:true) }
                else if store.logText.isEmpty { empty("Log is empty","New output appears here when the selected launcher writes to its log.","doc.text") }
                else {
                    TextField("Filter log lines",text:$logSearch).textFieldStyle(.roundedBorder)
                    Text(store.logText.components(separatedBy:"\n").filter { logSearch.isEmpty || $0.localizedCaseInsensitiveContains(logSearch) }.joined(separator:"\n")).font(.system(size:11,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(14).background(Color(nsColor:.textBackgroundColor),in:RoundedRectangle(cornerRadius:10)) }
            }.onAppear { store.readLog() }
        case .downloads:
            ToolsPanel(store:store,task:store.tool)
        case .runtime:
            ToolsPanel(store:store,task:store.tool,runtime:true)
        case .diagnostics:
            VStack(alignment:.leading,spacing:18) {
                HStack {
                    Button("Test Connection",systemImage:"stethoscope") { store.testConnection() }.buttonStyle(SushiPrimaryButtonStyle()).disabled(store.testingConnection)
                    Button("Export Report…",systemImage:"square.and.arrow.up") { store.exportDiagnostics() }.disabled(store.diagnosticText.isEmpty || store.testingConnection)
                    if store.testingConnection { ProgressView().controlSize(.small) }
                }
                if store.diagnosticText.isEmpty { empty("Check server connectivity","Test the connection without loading a model.","stethoscope") }
                else { Text(store.diagnosticText).font(.system(.callout,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(16).background(Color(nsColor:.textBackgroundColor),in:RoundedRectangle(cornerRadius:10)) }
            }
        case .general:
            VStack(alignment:.leading,spacing:18) {
                SushiCard("Appearance") {
                    VStack(alignment:.leading,spacing:12) {
                        Text("Choose how Sushi Control looks on this Mac.").font(.caption).foregroundStyle(.secondary)
                        HStack(spacing:10) {
                            ForEach([("System","system"),("Light","light"),("Dark","dark")],id:\.1) { title,value in
                                AppearanceOption(title:title,value:value,selected:store.appearance == value) { store.appearance = value }
                            }
                        }.onChange(of:store.appearance) { _,_ in store.applyAppearance() }
                    }
                }
                SushiCard("Menu bar") {
                    VStack(alignment:.leading,spacing:0) {
                        PreferenceRow("Menu bar display",detail:"Choose what appears beside the sushi icon.",icon:"menubar.rectangle") {
                            Picker("Menu bar display",selection:$store.menuBarMetric) { Text("Icon only").tag("None"); Text("Active requests").tag("Requests"); Text("Output speed").tag("Output"); Text("GPU allocation").tag("GPU") }.labelsHidden().frame(width:170)
                        }
                        Divider()
                        PreferenceRow("Live throughput graph",detail:"Show output and prefill in the menu panel.",icon:"waveform.path") {
                            Toggle("Live throughput graph",isOn:$store.menuShowGraph).labelsHidden().toggleStyle(.switch)
                        }
                        Divider()
                        PreferenceRow("Favorites in the menu",detail:"Keep your selected and loaded models available too.",icon:"star") {
                            Toggle("Favorites in the menu",isOn:$store.menuFavoritesOnly).labelsHidden().toggleStyle(.switch)
                        }
                        Divider()
                        VStack(alignment:.leading,spacing:10) {
                            Text("Additional menu metrics").font(.callout.weight(.medium))
                            Text("Output and prefill are always shown.").font(.caption).foregroundStyle(.secondary)
                            LazyVGrid(columns:[GridItem(.adaptive(minimum:120),alignment:.leading)],alignment:.leading,spacing:10) {
                                ForEach(ServerStore.menuOptionalMetricOptions,id:\.self) { name in
                                    Toggle(name,isOn:Binding(get:{store.menuMetricNames.contains(name)},set:{ _ in store.toggleMenuMetric(name) })).toggleStyle(.checkbox)
                                }
                            }
                        }.padding(.vertical,12)
                    }
                }
                SushiCard("App & startup") {
                    VStack(spacing:0) {
                        PreferenceRow("Refresh interval",detail:"How often Sushi Control updates its status.",icon:"clock") {
                            Picker("Refresh interval",selection:$store.refreshInterval) { ForEach([2,3,5,10],id:\.self) { seconds in Text("Every \(seconds) seconds").tag(Double(seconds)) } }.labelsHidden().frame(width:170)
                        }
                        Divider()
                        PreferenceRow("Keep Dock icon visible",detail:"Keep Sushi Control in the Dock when its window is closed.",icon:"dock.rectangle") {
                            Toggle("Keep Dock icon visible",isOn:$store.showDockIcon).labelsHidden().toggleStyle(.switch)
                        }
                        Divider()
                        PreferenceRow("Open at login",detail:"Start Sushi Control when you sign in.",icon:"laptopcomputer") {
                            Toggle("Open Sushi Control at login",isOn:$loginEnabled).labelsHidden().toggleStyle(.switch).onChange(of:loginEnabled) { _,value in
                                do { if value { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() } }
                                catch { store.error = error.localizedDescription; loginEnabled = SMAppService.mainApp.status == .enabled }
                            }
                        }
                        Text("Server startup is configured in Server → Launch & Performance. Quitting Sushi Control leaves the server running.")
                            .font(.caption).foregroundStyle(.secondary).frame(maxWidth:.infinity,alignment:.leading).padding(.vertical,10)
                    }
                }
                SushiCard {
                    PreferenceRow("Getting started",detail:"Connect a server, choose a CLI or review model setup.",icon:"sparkles") {
                        Button("Open Guide") { store.setupPresented = true; store.page = .status }
                    }
                    Divider()
                    PreferenceRow("Sushi Control 0.1",detail:"About this app and its licenses.",icon:"info.circle") {
                        Button("About…") { AppDelegate.instance?.showAbout() }
                    }
                }
            }

        }
    }
    func restoreLaunchForm() {
        do { launch = try LaunchSettings.read(store.arguments) }
        catch { advanced = true; store.error = error.localizedDescription }
    }
    func empty(_ title:String,_ detail:String,_ icon:String)->some View {
        VStack(spacing:10) { Image(systemName:icon).font(.system(size:30)).foregroundStyle(.secondary); Text(title).font(.headline); Text(detail).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth:420) }.frame(maxWidth:.infinity).padding(.vertical,30)
    }
}

@MainActor final class AppDelegate:NSObject,NSApplicationDelegate,NSWindowDelegate,NSPopoverDelegate,NSToolbarDelegate {
    static var instance:AppDelegate?
    private static let sidebarToolbarID = NSToolbarItem.Identifier("org.sushi.sidebar")
    var window:NSWindow?
    var item:NSStatusItem?
    let popover = NSPopover()
    var aboutWindow:NSWindow?
    var statusSubscription:AnyCancellable?
    override init() { super.init(); Self.instance = self; popover.delegate = self }
    func applicationDidFinishLaunching(_ notification:Notification) {
        ServerStore.shared.applyAppearance()
        configureMenus()
        let item = NSStatusBar.system.statusItem(withLength:NSStatusItem.variableLength)
        item.button?.image = sushiImage(size:18)
        item.button?.imageScaling = .scaleNone
        item.button?.imagePosition = .imageLeading
        item.button?.setAccessibilityLabel("Sushi Control"); item.button?.toolTip = "Sushi Control"
        item.button?.image?.isTemplate = true; item.button?.target = self; item.button?.action = #selector(togglePanel)
        self.item = item
        popover.behavior = .transient; popover.contentViewController = NSHostingController(rootView:MenuPanel(store:ServerStore.shared))
        item.button?.font = .monospacedDigitSystemFont(ofSize:12,weight:.medium)
        let store = ServerStore.shared
        statusSubscription = store.objectWillChange.debounce(for:.milliseconds(80),scheduler:RunLoop.main).sink { [weak self,weak store] _ in
            Task { @MainActor in if let store { self?.updateStatusItem(store) } }
        }
        updateStatusItem(store)
        showCenter()
    }
    func configureMenus() {
        let main = NSMenu(), appItem = NSMenuItem(); main.addItem(appItem)
        let appMenu = NSMenu(title:"Sushi Control")
        let about = NSMenuItem(title:"About Sushi Control…",action:#selector(showAbout),keyEquivalent:""); about.target = self; appMenu.addItem(about); appMenu.addItem(.separator())
        let center = NSMenuItem(title:"Control Center…",action:#selector(showCenter),keyEquivalent:","); center.target = self; appMenu.addItem(center)
        let menuControls = NSMenuItem(title:"Menu Bar Controls",action:#selector(togglePanel),keyEquivalent:"m")
        menuControls.keyEquivalentModifierMask = [.command,.shift]; menuControls.target = self; appMenu.addItem(menuControls)
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle:"Hide Sushi Control",action:#selector(NSApplication.hide(_:)),keyEquivalent:"h")
        let hideOthers = appMenu.addItem(withTitle:"Hide Others",action:#selector(NSApplication.hideOtherApplications(_:)),keyEquivalent:"h"); hideOthers.keyEquivalentModifierMask = [.command,.option]
        appMenu.addItem(withTitle:"Show All",action:#selector(NSApplication.unhideAllApplications(_:)),keyEquivalent:"")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle:"Quit Sushi Control",action:#selector(NSApplication.terminate(_:)),keyEquivalent:"q")
        appItem.submenu = appMenu
        let fileItem = NSMenuItem(); main.addItem(fileItem)
        let fileMenu = NSMenu(title:"File")
        fileMenu.addItem(withTitle:"Close Window",action:#selector(NSWindow.performClose(_:)),keyEquivalent:"w")
        fileItem.submenu = fileMenu
        let editItem = NSMenuItem(); main.addItem(editItem)
        let editMenu = NSMenu(title:"Edit")
        editMenu.addItem(withTitle:"Undo",action:Selector(("undo:")),keyEquivalent:"z")
        let redo = editMenu.addItem(withTitle:"Redo",action:Selector(("redo:")),keyEquivalent:"z"); redo.keyEquivalentModifierMask = [.command,.shift]
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle:"Cut",action:#selector(NSText.cut(_:)),keyEquivalent:"x")
        editMenu.addItem(withTitle:"Copy",action:#selector(NSText.copy(_:)),keyEquivalent:"c")
        editMenu.addItem(withTitle:"Paste",action:#selector(NSText.paste(_:)),keyEquivalent:"v")
        editMenu.addItem(withTitle:"Select All",action:#selector(NSText.selectAll(_:)),keyEquivalent:"a")
        editItem.submenu = editMenu
        let viewItem = NSMenuItem(); main.addItem(viewItem)
        let viewMenu = NSMenu(title:"View")
        for (index,page) in Page.allCases.enumerated() {
            let item = NSMenuItem(title:page.rawValue,action:#selector(navigate(_:)),keyEquivalent:index < 9 ? String(index+1) : index == 9 ? "0" : "")
            item.tag = index; item.target = self; viewMenu.addItem(item)
        }
        viewMenu.addItem(.separator())
        let sidebar = NSMenuItem(title:"Toggle Sidebar",action:#selector(toggleSidebar),keyEquivalent:"s")
        sidebar.keyEquivalentModifierMask = [.command,.control]; sidebar.target = self; viewMenu.addItem(sidebar)
        viewItem.submenu = viewMenu
        let windowItem = NSMenuItem(); main.addItem(windowItem)
        let windowMenu = NSMenu(title:"Window")
        windowMenu.addItem(withTitle:"Minimize",action:#selector(NSWindow.performMiniaturize(_:)),keyEquivalent:"m")
        windowMenu.addItem(withTitle:"Zoom",action:#selector(NSWindow.performZoom(_:)),keyEquivalent:"")
        windowMenu.addItem(.separator())
        windowMenu.addItem(withTitle:"Bring All to Front",action:#selector(NSApplication.arrangeInFront(_:)),keyEquivalent:"")
        windowItem.submenu = windowMenu; NSApp.windowsMenu = windowMenu
        let helpItem = NSMenuItem(); main.addItem(helpItem)
        let help = NSMenu(title:"Help")
        let docs = NSMenuItem(title:"Sushi Documentation",action:#selector(openDocumentation),keyEquivalent:""); docs.target = self; help.addItem(docs)
        let licenses = NSMenuItem(title:"Licenses & Acknowledgements…",action:#selector(showAbout),keyEquivalent:""); licenses.target = self; help.addItem(licenses)
        helpItem.submenu = help; NSApp.helpMenu = help; NSApp.mainMenu = main
    }
    func toolbarDefaultItemIdentifiers(_ toolbar:NSToolbar)->[NSToolbarItem.Identifier] { [Self.sidebarToolbarID,.flexibleSpace] }
    func toolbarAllowedItemIdentifiers(_ toolbar:NSToolbar)->[NSToolbarItem.Identifier] { toolbarDefaultItemIdentifiers(toolbar) }
    func toolbar(_ toolbar:NSToolbar,itemForItemIdentifier identifier:NSToolbarItem.Identifier,willBeInsertedIntoToolbar flag:Bool)->NSToolbarItem? {
        guard identifier == Self.sidebarToolbarID else { return nil }
        let control = NSToolbarItem(itemIdentifier:identifier)
        control.label = "Toggle Sidebar"; control.paletteLabel = "Toggle Sidebar"; control.toolTip = "Show or hide the sidebar"
        control.image = NSImage(systemSymbolName:"sidebar.left",accessibilityDescription:"Toggle Sidebar")
        control.target = self; control.action = #selector(toggleSidebar); control.isBordered = true
        return control
    }
    @objc func toggleSidebar() { ServerStore.shared.sidebarVisible.toggle() }
    private func updateStatusItem(_ store:ServerStore) {
        if let sidebar = window?.toolbar?.items.first(where:{$0.itemIdentifier == Self.sidebarToolbarID}) {
            sidebar.label = store.sidebarVisible ? "Hide Sidebar" : "Show Sidebar"
            sidebar.toolTip = sidebar.label
        }
        item?.button?.title = store.menuBarText
        item?.button?.toolTip = "Sushi · \(store.state) · \(store.menuValue("Output")) · GPU \(store.menuValue("GPU"))"
    }
    @objc func showAbout() {
        if aboutWindow == nil {
            let window = NSWindow(contentViewController:NSHostingController(rootView:AboutPanel()))
            window.title = "About Sushi Control"; window.styleMask = [.titled,.closable,.resizable]; window.setContentSize(NSSize(width:680,height:600)); window.isReleasedWhenClosed = false; window.center()
            aboutWindow = window
        }
        NSApp.activate(ignoringOtherApps:true); aboutWindow?.makeKeyAndOrderFront(nil)
    }
    @objc func openDocumentation() { NSWorkspace.shared.open(URL(string:"https://github.com/beamivalice/sushi/blob/main/docs/usage.md")!) }
    @objc func navigate(_ sender:NSMenuItem) {
        guard Page.allCases.indices.contains(sender.tag) else { return }
        ServerStore.shared.page = Page.allCases[sender.tag]; showCenter()
    }
    @objc func togglePanel() {
        if popover.isShown { popover.performClose(nil) }
        else if let button = item?.button {
            NSApp.activate(ignoringOtherApps:true)
            let maximumHeight = max(180,min(650,(button.window?.screen?.visibleFrame.height ?? 750)-60))
            popover.contentViewController = NSHostingController(rootView:MenuPanel(store:ServerStore.shared,maximumHeight:maximumHeight,preferredHeightChanged:{ [weak self] height in self?.popover.contentSize = NSSize(width:390,height:height) }))
            popover.contentSize = NSSize(width:390,height:min(maximumHeight,350))
            popover.show(relativeTo:button.bounds,of:button,preferredEdge:.minY)
            popover.contentViewController?.view.window?.makeKey()
        }
    }
    func popoverDidClose(_ notification:Notification) {
        if let window, window.isVisible { window.makeKey() }
    }
    func applicationShouldHandleReopen(_ sender:NSApplication,hasVisibleWindows:Bool)->Bool { showCenter(); return true }
    func applicationShouldTerminate(_ sender:NSApplication)->NSApplication.TerminateReply {
        if ServerStore.shared.busy {
            ServerStore.shared.error = ServerStore.shared.benchmark.running ? "Cancel or finish the benchmark before quitting." : "Wait for the server operation to finish before quitting."
            showCenter(); return .terminateCancel
        }
        if ServerStore.shared.tool.running {
            ServerStore.shared.error = "Wait for the current task to finish or cancel it before quitting."
            showCenter(); return .terminateCancel
        }
        return .terminateNow
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool { false }
    func applyActivationPolicy(keepDockVisible:Bool) {
        NSApp.setActivationPolicy(keepDockVisible || window?.isVisible == true ? .regular : .accessory)
    }
    func windowWillClose(_ notification:Notification) { NSApp.setActivationPolicy(ServerStore.shared.showDockIcon ? .regular : .accessory) }
    @objc func showCenter() {
        NSApp.setActivationPolicy(.regular); popover.performClose(nil)
        if window == nil {
            let window = NSWindow(contentRect:NSRect(x:0,y:0,width:960,height:720),styleMask:[.titled,.closable,.miniaturizable,.resizable],backing:.buffered,defer:false)
            window.titlebarAppearsTransparent = true
            window.titleVisibility = .hidden
            window.toolbarStyle = .unifiedCompact
            let restored = window.setFrameUsingName("SushiControlCenter")
            window.setFrameAutosaveName("SushiControlCenter")
            window.title = "Sushi Control"; window.delegate = self; window.isReleasedWhenClosed = false
            window.contentView = NSHostingView(rootView:ControlCenter(store:ServerStore.shared))
            let toolbar = NSToolbar(identifier:"SushiControlToolbar")
            toolbar.delegate = self; toolbar.displayMode = .iconOnly; toolbar.allowsUserCustomization = false
            window.toolbar = toolbar
            if !restored { window.center() }; self.window = window
            updateStatusItem(ServerStore.shared)
        }
        window?.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps:true)
    }
}
#if !DESIGN_PREVIEW
@main struct SushiControlMain {
    @MainActor static func main() {
        let app = NSApplication.shared; app.setActivationPolicy(.accessory)
        let delegate = AppDelegate(); app.delegate = delegate; app.run(); _ = delegate
    }
}

#endif

struct DesktopControllerPanel:View {
    @ObservedObject var store:ServerStore
    var body:some View {
        SushiCard("Optional desktop controller") {
            VStack(alignment:.leading,spacing:12) {
                Text("Connect a compatible local controller to pause new requests or load and unload its model. Standard Sushi servers do not require this extension.").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Choose Control Token…") { store.chooseControllerToken() }
                    if !store.controllerToken.isEmpty { Button("Disconnect") { store.controllerToken = "" } }
                }.disabled(store.busy)
                Text("The control token stays in memory until you disconnect, change servers, or quit. It is separate from the inference API key.").font(.caption).foregroundStyle(.secondary)
                if let state = store.controllerState {
                    LabeledContent("Controller",value:state.state.capitalized)
                    LabeledContent("Active requests",value:String(state.activeRequests))
                    HStack {
                        Button(state.paused ? "Resume Requests" : "Pause New Requests") { store.controllerAction(state.paused ? "resume" : "pause") }
                        Button("Load Model") { store.controllerAction("load") }.disabled(state.state != "sleeping")
                        Button("Unload Model") { store.controllerAction("unload") }.disabled(state.activeRequests > 0 || state.state != "ready")
                    }.disabled(store.busy || store.tool.exclusiveServerUse)
                    Text(state.paused ? "New requests are paused. Existing requests can finish." : "New requests are accepted. Unloading keeps the controller available for the next load.").font(.caption).foregroundStyle(.secondary)
                }
                if !store.controllerError.isEmpty { Text(store.controllerError).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            }.padding(12)
        }
    }
}
