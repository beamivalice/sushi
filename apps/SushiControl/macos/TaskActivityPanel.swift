import SwiftUI
import AppKit

struct TaskActivityPanel:View {
    @ObservedObject var store:ServerStore
    @ObservedObject var task:ToolTask
    var allowRescan = false
    var resumeTimeout:Double = 7200
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            if !task.title.isEmpty {
                SushiCard {
                    VStack(alignment:.leading,spacing:12) {
                        HStack { VStack(alignment:.leading,spacing:4) { Text(task.title).font(.headline); Text(task.status).font(.caption).foregroundStyle(.secondary) }; Spacer(); if task.running { ProgressView().controlSize(.small); Button("Cancel",role:.destructive) { task.cancel() } } }
                        if task.running, let progress = task.downloadProgress {
                            VStack(alignment:.leading,spacing:6) {
                                HStack {
                                    Text("File \(progress.index) of \(progress.total)").font(.caption).foregroundStyle(.secondary)
                                    Spacer()
                                    if let fraction = progress.fraction { Text(fraction,format:.percent.precision(.fractionLength(1))).monospacedDigit() }
                                }
                                Text(progress.file).font(.caption.monospaced()).lineLimit(2)
                                if let fraction = progress.fraction { ProgressView(value:fraction) }
                                else { ProgressView().controlSize(.small) }
                                Text("Progress for the current file").font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                        if let log = task.logURL { Button("Show Task Log in Finder",systemImage:"folder") { NSWorkspace.shared.activateFileViewerSelecting([log]) } }
                        if let started = task.started, task.running { Text(started,style:.relative).font(.caption).foregroundStyle(.secondary) }
                        if !task.output.isEmpty { DisclosureGroup("Task output") { Text(task.output).font(.system(size:11,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(12).background(Color(nsColor:.textBackgroundColor),in:RoundedRectangle(cornerRadius:8)) } }
                    }.padding(12)
                }
                if !task.running && allowRescan && store.online { Button("Refresh Model Library") { store.modelAction("models/rescan") }.disabled(store.busy || !store.rescanAvailable) }
            }
            if !task.history.isEmpty {
                SushiCard("Recent tasks") {
                    VStack(alignment:.leading,spacing:12) {
                        HStack { Text("Recent history · up to 20 tasks").font(.caption).foregroundStyle(.secondary); Spacer(); Button("Clear") { task.clearHistory() } }
                        ForEach(task.history) { record in
                            HStack(alignment:.top) {
                                Image(systemName:record.status == "Completed" ? "checkmark.circle.fill" : "exclamationmark.circle").foregroundStyle(record.status == "Completed" ? Color.green : Color.orange)
                                VStack(alignment:.leading,spacing:4) {
                                    Text(record.title).font(.subheadline.weight(.medium))
                                    Text("\(record.status) · \(record.finished.formatted(date:.abbreviated,time:.shortened))").font(.caption).foregroundStyle(.secondary)
                                    Text(Duration.seconds(max(0,record.finished.timeIntervalSince(record.started))).formatted(.time(pattern:.hourMinuteSecond))).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if let model = record.downloadModel, record.status != "Completed" {
                                    Button("Resume") { resume(model) }.disabled(task.running || store.busy || store.toolBinary.isEmpty)
                                }
                                Button("Log",systemImage:"doc.text") { NSWorkspace.shared.activateFileViewerSelecting([record.logURL]) }.disabled(!FileManager.default.fileExists(atPath:record.logURL.path))
                            }
                            if record.id != task.history.last?.id { Divider() }
                        }
                    }.padding(12)
                }
            }
        }
    }
    private func resume(_ model:String) {
        guard !store.busy else { return }
        do { try task.run(binary:store.toolBinary,arguments:["pull",model],title:"Download \(model)",timeout:resumeTimeout) }
        catch { store.error = error.localizedDescription }
    }
}
