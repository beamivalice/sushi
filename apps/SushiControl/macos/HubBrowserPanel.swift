import SwiftUI
import AppKit

struct HubBrowserPanel:View {
    @ObservedObject var store:ServerStore
    @ObservedObject var task:ToolTask
    var downloadTimeout:TimeInterval
    @StateObject private var browser = HubBrowser()
    @Environment(\.dismiss) private var dismiss
    @State private var query = "Sushi"
    @State private var notice = ""
    private var catalog:[DownloadPack] { task.catalogBinary == store.toolBinary ? task.catalog : [] }
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            HStack { VStack(alignment:.leading,spacing:5) { Text("Publisher Models").font(.title2.bold()).accessibilityAddTraits(.isHeader); Text("Public metadata from huggingface.co/beamster").font(.caption).foregroundStyle(.secondary) }; Spacer(); Button("Done") { dismiss() }.keyboardShortcut(.cancelAction) }
            HStack { TextField("Search publisher models",text:$query).textFieldStyle(.roundedBorder).onSubmit { browser.search(query) }; Button("Search",systemImage:"magnifyingglass") { browser.search(query) }; if browser.loading { ProgressView().controlSize(.small) } }
            HStack {
                Text("Download actions use a matching entry in your installed Sushi CLI catalogue. The CLI controls repository selection and runtime support.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Refresh CLI Catalogue") { refreshCatalog() }.disabled(task.running || store.busy || store.toolBinary.isEmpty)
            }
            if !browser.error.isEmpty { Feedback(text:browser.error,failure:true) }
            if !notice.isEmpty { Text(notice).font(.caption).foregroundStyle(.secondary) }
            HSplitView {
                SushiScrollView {
                    LazyVStack(alignment:.leading,spacing:8) {
                        ForEach(browser.models) { model in
                            Button { browser.select(model) } label: {
                                HStack(alignment:.top,spacing:10) {
                                    Image(systemName:"shippingbox").foregroundStyle(.secondary)
                                    VStack(alignment:.leading,spacing:5) { Text(model.id.split(separator:"/").last.map(String.init) ?? model.id).font(.subheadline.weight(.medium)).lineLimit(3); if let count = model.downloads { Text("\(count.formatted()) reported downloads").font(.caption).foregroundStyle(.secondary) } }
                                    Spacer(minLength:0)
                                }.padding(10).frame(maxWidth:.infinity,alignment:.leading).contentShape(Rectangle()).background(browser.selected?.id == model.id ? Color.sushiAccent.opacity(0.1) : Color.clear,in:RoundedRectangle(cornerRadius:10))
                            }.buttonStyle(.plain)
                        }
                        if browser.models.isEmpty && !browser.loading { Text("No publisher models found.").foregroundStyle(.secondary).padding() }
                    }.padding(4)
                }.frame(minWidth:240,idealWidth:300,maxWidth:340)
                SushiScrollView {
                    if let model = browser.selected { details(model).padding(16) }
                    else { ContentUnavailableView("Choose a model",systemImage:"shippingbox",description:Text("Inspect repository metadata, reported files and the model card before downloading.")) }
                }.frame(minWidth:390,maxWidth:.infinity)
            }
        }.padding(24).frame(width:900,height:650).task { browser.search(query) }.onDisappear { browser.cancel() }
    }
    private func details(_ model:HubModel)->some View {
        VStack(alignment:.leading,spacing:14) {
            Text(model.id).font(.headline).textSelection(.enabled)
            Link("View Model Card",destination:model.pageURL)
            if let modified = model.modified { LabeledContent("Updated",value:modified.formatted(date:.abbreviated,time:.omitted)) }
            LabeledContent("License metadata",value:model.license ?? "Not reported")
            if let likes = model.likes { LabeledContent("Reported likes",value:likes.formatted()) }
            if let pack = model.matchingPack(catalog) {
                Divider()
                Text("CLI catalogue: \(pack.id)").font(.subheadline.weight(.medium))
                Text("\(pack.size) · \(pack.verdict)").font(.caption).foregroundStyle(.secondary)
                Text("Download time limit: \(Int(downloadTimeout/60)) minutes").font(.caption).foregroundStyle(.secondary)
                if pack.recommended { Label("Recommended by your Sushi CLI",systemImage:"sparkles").font(.caption).foregroundStyle(.tint) }
                HStack { Button(pack.downloaded ? "Verify / Resume" : "Download Now") { download(pack,queued:false) }.buttonStyle(.borderedProminent).disabled(task.running || store.busy || !pack.published); Button("Add to Queue") { download(pack,queued:true) }.disabled(store.busy || !pack.published) }
            } else {
                Text(catalog.isEmpty ? "Refresh the CLI catalogue to enable matching download actions." : "This repository has no unambiguous name match in this CLI’s catalogue. Check its model card and CLI support before using it.").font(.caption).foregroundStyle(.secondary)
            }
            if !model.tags.isEmpty { DisclosureGroup("Repository tags") { Text(model.tags.joined(separator:" · ")).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) } }
            Divider()
            Text("Reported files\(model.fileCount > 100 ? " (first 100 of \(model.fileCount))" : "")").font(.subheadline.weight(.semibold))
            if model.files.isEmpty { Text(browser.loading ? "Loading file metadata…" : "No file metadata reported.").font(.caption).foregroundStyle(.secondary) }
            ForEach(model.files) { file in
                HStack(alignment:.top) { Text(file.name).font(.caption.monospaced()).textSelection(.enabled); Spacer(); if let bytes = file.bytes { Text(ByteCountFormatter.string(fromByteCount:bytes,countStyle:.file)).font(.caption).foregroundStyle(.secondary) } }
            }
            Text("Public repository metadata is informational. Downloads, resume behavior and the files fetched are managed by the selected CLI. No weights download until you choose Download or start the queue.").font(.caption).foregroundStyle(.secondary)
        }.frame(maxWidth:.infinity,alignment:.leading)
    }
    private func refreshCatalog() {
        guard !store.busy, !task.running else { return }
        do { try task.run(binary:store.toolBinary,arguments:["list"],title:"Available models",timeout:45) }
        catch { notice = error.localizedDescription }
    }
    private func download(_ pack:DownloadPack,queued:Bool) {
        guard !store.busy, pack.published else { return }
        do {
            if queued { try task.enqueueDownload(binary:store.toolBinary,model:pack.id,timeout:downloadTimeout); notice = "Queued \(pack.id). \(task.queuePaused ? "Start the queue from Downloads when ready." : "It will run in order after the current download.")" }
            else { try task.run(binary:store.toolBinary,arguments:["pull",pack.id],title:"Download \(pack.id)",timeout:downloadTimeout); notice = "Downloading \(pack.id). Progress is available in Downloads." }
        } catch { notice = error.localizedDescription }
    }
}
