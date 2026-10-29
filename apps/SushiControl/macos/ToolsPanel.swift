import SwiftUI

struct ToolsPanel:View {
    @ObservedObject var store:ServerStore
    @ObservedObject var task:ToolTask
    var runtime = false
    @State private var model = ""
    @State private var timeoutMinutes = 120
    @State private var catalogSearch = ""
    @State private var confirmUpdate = false
    @State private var showPublisher = false
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            HStack {
                Text(store.toolBinary.isEmpty ? "No CLI selected" : store.toolBinary).font(.caption.monospaced()).lineLimit(2).textSelection(.enabled)
                Spacer()
                Button("Choose Sushi CLI…") { store.chooseToolBinary() }.disabled(task.running || store.busy)
            }
            if store.toolBinary.isEmpty {
                ContentUnavailableView("Choose a Sushi executable",systemImage:"terminal",description:Text("Choose your installed Sushi CLI to use these tools."))
            } else {
                SushiCard(runtime ? "Sushi installation" : "Model downloads") {
                    VStack(alignment:.leading,spacing:14) {
                        if runtime {
                            Text(store.toolBinary).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                            HStack {
                                Button("Installed Version") { run(["--version"],title:"Installed version",timeout:20) }
                                Button("Check for Updates") { run(["update","--check"],title:"Update check",timeout:90) }
                                Button("Update Runtime…") { confirmUpdate = true }.disabled(!store.canRunOfflineMaintenance)
                            }
                            Text("Stop servers using this installation before updating it.").font(.caption).foregroundStyle(.secondary)
                        } else {
                            HStack { TextField("Model alias or beamster/repository",text:$model).textFieldStyle(.roundedBorder); Button("Download",systemImage:"arrow.down.circle") { run(["pull",model.trimmingCharacters(in:.whitespacesAndNewlines)],title:"Download \(model)",timeout:Double(timeoutMinutes*60)) }.disabled(model.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty || task.running)
                                Button("Add to Queue",systemImage:"text.badge.plus") { enqueue(model) }.disabled(model.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
                            }
                            HStack { Button("Show Available Models",systemImage:"list.bullet") { run(["list"],title:"Available models",timeout:45) }.disabled(task.running); Spacer(); Picker("Time limit",selection:$timeoutMinutes) { Text("30 minutes").tag(30); Text("2 hours").tag(120); Text("4 hours").tag(240) }.frame(width:240) }
                            Button("Browse Publisher Models…",systemImage:"globe") { showPublisher = true }
                            Text("Downloads go to ~/.sushi/models. Repeating a download resumes incomplete files.").font(.caption).foregroundStyle(.secondary)
                        }
                    }.padding(12)
                }.disabled(store.busy || (runtime && task.running))
            }
            if let error = task.persistenceError { Label(error,systemImage:"exclamationmark.triangle").font(.caption).foregroundStyle(.orange) }
            if let notice = task.recoveryNotice { Label(notice,systemImage:"arrow.clockwise.circle").font(.caption).foregroundStyle(.secondary) }
            if !runtime, task.catalogBinary == store.toolBinary, !task.catalog.isEmpty {
                SushiCard("Available packs") {
                    VStack(alignment:.leading,spacing:12) {
                        TextField("Search packs",text:$catalogSearch).textFieldStyle(.roundedBorder)
                        Text("Sizes and hardware recommendations come from your Sushi CLI. Context and runtime memory require additional headroom.").font(.caption).foregroundStyle(.secondary)
                        ForEach(task.catalog.filter { catalogSearch.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty || $0.id.localizedCaseInsensitiveContains(catalogSearch.trimmingCharacters(in:.whitespacesAndNewlines)) }) { pack in
                            HStack(alignment:.top,spacing:12) {
                                Image(systemName:pack.downloaded ? "internaldrive.fill" : "shippingbox").font(.title2).foregroundStyle(.secondary)
                                VStack(alignment:.leading,spacing:5) {
                                    Text(pack.id).font(.subheadline.weight(.semibold)).textSelection(.enabled)
                                    Text("\(pack.size) · \(pack.verdict)").font(.caption).foregroundStyle(.secondary)
                                    if pack.recommended { Label("Recommended by Sushi",systemImage:"sparkles").font(.caption).foregroundStyle(.tint) }
                                    if !pack.published { Text("Not yet published").font(.caption).foregroundStyle(.secondary) }
                                }
                                Spacer()
                                Button(pack.downloaded ? "Verify / Resume" : "Download",systemImage:pack.downloaded ? "arrow.clockwise" : "arrow.down.circle") {
                                    run(["pull",pack.id],title:"Download \(pack.id)",timeout:Double(timeoutMinutes*60))
                                }.disabled(!pack.published || task.running || store.busy)
                                Button("Queue",systemImage:"text.badge.plus") { enqueue(pack.id) }.disabled(!pack.published || store.busy)
                            }.padding(.vertical,5)
                            Divider()
                        }
                    }.padding(12)
                }
            }
            if !runtime, !task.queue.isEmpty || task.queueError != nil {
                SushiCard("Download queue") {
                    VStack(alignment:.leading,spacing:12) {
                        HStack {
                            Text(task.queuePaused ? "Paused" : "Running in order").font(.subheadline.weight(.medium))
                            Spacer()
                            Button("Clear Queue") { task.clearQueue() }
                            if task.queuePaused {
                                Button("Start Queue",systemImage:"play.fill") {
                                    guard !store.busy else { return }
                                    do { try task.startQueue() } catch { store.error = error.localizedDescription }
                                }.disabled(task.running || store.busy || task.queue.isEmpty)
                            } else { Button("Pause After Current",systemImage:"pause.fill") { task.pauseQueue() } }
                        }
                        Text("Each download keeps the CLI and time limit selected when added. Cancelling or failing a download pauses the queue. Queued tasks are saved and reopen paused.").font(.caption).foregroundStyle(.secondary)
                        if let error = task.queueError { Text(error).font(.caption).foregroundStyle(.orange) }
                        ForEach(Array(task.queue.enumerated()),id:\.element.id) { index, item in
                            HStack {
                                Text("\(index+1)").monospacedDigit().foregroundStyle(.secondary)
                                VStack(alignment:.leading,spacing:3) { Text(item.model).font(.subheadline); Text(item.binary).font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(1).help(item.binary) }
                                Spacer()
                                Button("Move Up",systemImage:"arrow.up") { task.moveQueued(item.id,offset:-1) }.labelStyle(.iconOnly).disabled(index == 0)
                                Button("Move Down",systemImage:"arrow.down") { task.moveQueued(item.id,offset:1) }.labelStyle(.iconOnly).disabled(index == task.queue.count-1)
                                Button("Remove",systemImage:"xmark") { task.removeQueued(item.id) }.labelStyle(.iconOnly)
                            }
                        }
                    }.padding(12)
                }
            }
            TaskActivityPanel(store:store,task:task,allowRescan:!runtime,resumeTimeout:Double(timeoutMinutes*60))
        }.sheet(isPresented:$showPublisher) { HubBrowserPanel(store:store,task:task,downloadTimeout:Double(timeoutMinutes*60)) }
        .confirmationDialog("Update the selected Sushi installation?",isPresented:$confirmUpdate,titleVisibility:.visible) {
            Button("Update Runtime") { run(["update"],title:"Update Sushi",timeout:600) }
        } message: { Text("Sushi will download and install an update using its built-in updater.") }
    }
    func enqueue(_ model:String) {
        do { try task.enqueueDownload(binary:store.toolBinary,model:model,timeout:Double(timeoutMinutes*60)) }
        catch { store.error = error.localizedDescription }
    }
    func run(_ arguments:[String],title:String,timeout:TimeInterval) {
        guard !store.busy else { store.error = "Wait for the server operation to finish before running a tool."; return }
        if arguments == ["update"] {
            Task {
                await store.refresh()
                guard store.canRunOfflineMaintenance else { store.error = "Stop and unregister the server launcher before updating its runtime."; return }
                startTool(arguments,title:title,timeout:timeout)
            }
        } else { startTool(arguments,title:title,timeout:timeout) }
    }
    private func startTool(_ arguments:[String],title:String,timeout:TimeInterval) {
        do { try task.run(binary:store.toolBinary,arguments:arguments,title:title,timeout:timeout) }
        catch { store.error = error.localizedDescription }
    }
}
