import SwiftUI
import AppKit

struct LocalModelFilesPanel:View {
    @ObservedObject var store:ServerStore
    @ObservedObject var task:ToolTask
    @Environment(\.dismiss) private var dismiss
    @State private var selected:LocalModelFile?
    @State private var receipt:ModelTrashReceipt?
    @State private var working = false
    @State private var confirmTrash = false
    @State private var error = ""
    @State private var notice = ""
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            HStack { VStack(alignment:.leading,spacing:5) { Text("Local Model Files").font(.title2.bold()).accessibilityAddTraits(.isHeader); Text("Inspect and manage a model on this Mac").foregroundStyle(.secondary) }; Spacer(); Button("Done") { dismiss() }.keyboardShortcut(.cancelAction).disabled(working && store.busy) }
            SushiScrollView {
                VStack(alignment:.leading,spacing:16) {
            HStack { Button("Choose Model…",systemImage:"folder") { choose() }.disabled(working); if working { ProgressView().controlSize(.small); Text("Working…").font(.caption).foregroundStyle(.secondary) } }
            if let selected {
                SushiCard(selected.url.lastPathComponent) {
                    VStack(alignment:.leading,spacing:12) {
                        Text(selected.url.path).font(.caption.monospaced()).textSelection(.enabled)
                        LabeledContent("Format / architecture",value:selected.architecture)
                        LabeledContent(selected.partial ? "File bytes counted (partial)" : "File bytes",value:ByteCountFormatter.string(fromByteCount:selected.bytes,countStyle:.file))
                        LabeledContent("Files counted",value:selected.fileCount.formatted())
                        Text("Shared or symbolic-link targets are not counted. File bytes do not predict space freed by moving an item to Trash.").font(.caption).foregroundStyle(.secondary)
                        HStack { Button("Reveal in Finder",systemImage:"folder") { NSWorkspace.shared.activateFileViewerSelecting([selected.url]) }; Spacer(); Button("Move to Trash…",systemImage:"trash",role:.destructive) { confirmTrash = true }.disabled(!store.canTrashLocalModels || working) }
                    }.padding(8)
                }
            } else {
                Text("Choose an individual model folder containing config.json and weights, or a GGUF file. The folder’s contents move together.").foregroundStyle(.secondary)
            }
            if !store.canTrashLocalModels { Text("Stop the connected server and any running tools before moving model files. Other apps using the same files must also be stopped.").font(.caption).foregroundStyle(.secondary) }
            if let receipt {
                HStack { Text("Moved to Trash. Restore is available here until this dialog closes, or later in Finder.").font(.caption); Spacer(); Button("Restore Model") { restore(receipt) }.disabled(!store.canTrashLocalModels || working) }
            }
            if !notice.isEmpty { Label(notice,systemImage:"checkmark.circle").font(.callout).foregroundStyle(.secondary) }
            if !error.isEmpty { Feedback(text:error,failure:true) }
                }.frame(maxWidth:.infinity,alignment:.leading)
            }

        }.padding(24).frame(width:650,height:selected == nil ? 300 : 560)
        .confirmationDialog("Move \(selected?.url.lastPathComponent ?? "this model") to Trash?",isPresented:$confirmTrash,titleVisibility:.visible) {
            Button("Move Model to Trash",role:.destructive) { trash() }
        } message: { Text("The selected file or entire folder moves to this Mac’s Trash. Other apps and shared caches may use these files. Stop them first. You can restore the item in Finder or with Restore Model.") }
    }
    private func choose() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = true; panel.allowsMultipleSelection = false; panel.prompt = "Choose Model"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        working = true; error = ""; notice = ""
        Task {
            do { selected = try await Task.detached(priority:.utility) { try LocalModelFile.inspect(url) }.value }
            catch { selected = nil; self.error = error.localizedDescription }
            working = false
        }
    }
    private func trash() {
        guard let selected, !working else { return }
        working = true
        Task {
            await store.refresh()
            guard store.canTrashLocalModels else { error = "Stop the server and running tools before moving model files."; working = false; return }
            store.busy = true; store.operation = "Moving model to Trash…"
            do {
                receipt = try await Task.detached(priority:.utility) { try selected.moveToTrash() }.value
                self.selected = nil; error = ""; notice = "Model moved to Trash. Server registrations update when the library is rescanned."
            } catch { self.error = error.localizedDescription }
            store.busy = false; store.operation = ""; working = false; store.refreshNow()
        }
    }
    private func restore(_ receipt:ModelTrashReceipt) {
        guard !working else { return }; working = true
        Task {
            await store.refresh()
            guard store.canTrashLocalModels else { error = "Stop the server and running tools before restoring model files."; working = false; return }
            store.busy = true; store.operation = "Restoring model…"
            do { try await Task.detached(priority:.utility) { try receipt.restore() }.value; self.receipt = nil; notice = "Model restored to its original path."; error = "" }
            catch { self.error = error.localizedDescription }
            store.busy = false; store.operation = ""; working = false; store.refreshNow()
        }
    }
}
