import SwiftUI
import AppKit

struct ModelSettingsPanel:View {
    @Environment(\.dismiss) private var dismiss
    @State private var document:ModelSettingsDocument?
    @State private var path = ""
    @State private var overrides = ModelOverrides()
    @State private var error = ""
    @State private var notice = ""
    @State private var backup:URL?
    @State private var presets:ModelPresetLibrary?
    @State private var presetSnapshot:Data?
    @State private var presetID:UUID?
    @State private var presetName = ""
    @State private var presetError = ""
    @State private var presetUndo:ModelPresetLibrary?
    private let presetKey = "modelTuningPresets"

    private var settingsURL:URL { FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".sushi/model-settings.json") }
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            HStack {
                VStack(alignment:.leading,spacing:5) { Text("Model Settings").font(.title2.bold()).accessibilityAddTraits(.isHeader); Text("Defaults for a model on this Mac").foregroundStyle(.secondary) }
                Spacer(); Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Text("Choose the same folder or GGUF file used by your local Sushi server. These settings affect servers using this Mac’s ~/.sushi directory.").font(.callout).foregroundStyle(.secondary)
            HStack { Text(path.isEmpty ? "No model selected" : path).font(.caption.monospaced()).textSelection(.enabled).lineLimit(3); Spacer(); Button("Choose Model…") { choose() } }
            if !error.isEmpty { Label(error,systemImage:"exclamationmark.triangle").font(.callout).foregroundStyle(.orange) }
            if document != nil {
                SushiScrollView {
                VStack(alignment:.leading,spacing:16) {
                Form {
                    TextField("Context tokens",text:$overrides.context,prompt:Text("Automatic"))
                    Picker("KV cache",selection:$overrides.kv) { Text("Default").tag("Default"); ForEach(["4","6","8","16"],id:\.self) { Text($0 == "16" ? "16-bit (BF16)" : "\($0)-bit").tag($0) } }
                    choice("MTP speculation",selection:$overrides.mtp)
                    Picker("MTP acceptance",selection:$overrides.acceptance) {
                        Text("Default").tag("Default"); Text("Exact").tag("exact"); Text("Typical (lossy)").tag("typical"); Text("Token v3 (lossy)").tag("tokenv3")
                    }
                    choice("MTP greedy tail",selection:$overrides.greedyTail)
                    TextField("SSD expert-streaming budget (GB)",text:$overrides.ssdBudget,prompt:Text("Default"))
                    choice("Vision tower",selection:$overrides.vision)
                    choice("Preserve thinking",selection:$overrides.preserveThinking)
                    TextField("Thinking penalty",text:$overrides.thinkPenalty,prompt:Text("Default (0–20)"))
                }.formStyle(.columns)
                DisclosureGroup("Saved tuning presets") { presetControls.padding(.top,8) }
                DisclosureGroup("Settings help") {
                Text("Typical and Token v3 change sampled output distributions. The SSD budget controls expert streaming, separately from the prefix cache. Leave defaults unless you intend to tune these behaviors.").font(.caption).foregroundStyle(.secondary)
                Text("Launch flags override these values. Model support varies; GLM does not support 4-bit KV. Context, KV, MTP and vision changes take effect on the next model load. Saving does not unload, load or restart anything.").font(.caption).foregroundStyle(.secondary)
                }.font(.caption).foregroundStyle(.secondary)
                }.padding(.vertical,4)
                }
                HStack {
                    Button("Use Defaults") { overrides = ModelOverrides() }
                    Spacer()
                    Button("Save Settings") { save() }.buttonStyle(.borderedProminent)
                }
            }
            if !notice.isEmpty { Label(notice,systemImage:"checkmark.circle").font(.callout).foregroundStyle(.secondary) }
            if let backup { Button("Reveal Backup",systemImage:"folder") { NSWorkspace.shared.activateFileViewerSelecting([backup]) } }
            Text(settingsURL.path).font(.caption2.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
        }.padding(24).frame(width:650,height:document == nil ? 300 : 620).onAppear { loadPresets() }
    }
    var presetControls:some View {
        SushiCard("Saved tuning presets") {
            VStack(alignment:.leading,spacing:10) {
                HStack {
                    Picker("Apply preset",selection:$presetID) {
                        Text("Choose a preset…").tag(Optional<UUID>.none)
                        ForEach(presets?.presets ?? []) { preset in Text(preset.name).tag(Optional(preset.id)) }
                    }.onChange(of:presetID) { _,id in
                        if let preset = presets?.presets.first(where:{$0.id == id}) {
                            overrides = preset.values; presetName = preset.name
                            notice = "Preset applied to the editor. Review the values, then Save Settings to apply them to this model."
                        }
                    }
                    Button("Remove") { removePreset() }.disabled(presetID == nil || presets == nil)
                }
                HStack {
                    TextField("Preset name",text:$presetName)
                    Button("Save Preset") { savePreset() }.disabled(presets == nil || presetName.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
                    if presetUndo != nil { Button("Undo") { undoPreset() } }
                }
                Text("Presets save the settings shown above. Saving the same name updates it. Applying a preset only edits this form; model loading and server launch flags remain separate.").font(.caption).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
                if !presetError.isEmpty { Text(presetError).font(.caption).foregroundStyle(.orange); Button("Reload Presets") { loadPresets() } }
            }
        }
    }
    private func loadPresets() {
        let data = UserDefaults.standard.data(forKey:presetKey)
        do { presets = try ModelPresetLibrary(data:data); presetSnapshot = data; presetError = ""; presetID = nil; presetUndo = nil }
        catch { presets = nil; presetError = "Saved presets could not be opened and have been kept unchanged. \(error.localizedDescription)" }
    }
    private func writePresets(_ next:ModelPresetLibrary,undo:ModelPresetLibrary?) throws {
        guard UserDefaults.standard.data(forKey:presetKey) == presetSnapshot else { throw ServerFailure(message:"Presets changed in another window. Reload them before saving.") }
        let data = try next.encoded()
        UserDefaults.standard.set(data,forKey:presetKey); presetSnapshot = data; presets = next; presetUndo = undo; presetError = ""
    }
    private func savePreset() {
        guard let current = presets else { return }
        do {
            var next = current; try next.save(name:presetName,values:overrides)
            try writePresets(next,undo:current)
            notice = "Preset saved on this Mac. Model settings have not been written."
        } catch { presetError = error.localizedDescription }
    }
    private func removePreset() {
        guard let current = presets, let presetID else { return }
        do { var next = current; next.remove(presetID); try writePresets(next,undo:current); self.presetID = nil; presetName = "" }
        catch { presetError = error.localizedDescription }
    }
    private func undoPreset() {
        guard let presetUndo else { return }
        do { try writePresets(presetUndo,undo:nil); presetID = nil; presetName = "" }
        catch { presetError = error.localizedDescription }
    }
    func choice(_ label:String,selection:Binding<String>) -> some View {
        Picker(label,selection:selection) { ForEach(["Default","On","Off"],id:\.self) { Text($0).tag($0) } }
    }
    private func choose() {
        let panel = NSOpenPanel(); panel.canChooseFiles = true; panel.canChooseDirectories = true; panel.allowsMultipleSelection = false; panel.prompt = "Choose Model"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let directory = (try? url.resourceValues(forKeys:[.isDirectoryKey]).isDirectory) == true
            guard directory ? FileManager.default.fileExists(atPath:url.appendingPathComponent("config.json").path) : url.pathExtension.lowercased() == "gguf" else { throw ServerFailure(message:"Choose a model folder containing config.json, or a GGUF file.") }
            let next = try ModelSettingsDocument(url:settingsURL)
            let values = try next.overrides(for:url.path)
            path = url.path; overrides = values; document = next; error = ""; notice = ""; backup = nil; presetID = nil; presetName = ""
        } catch { self.error = error.localizedDescription }
    }
    private func save() {
        guard let document else { return }
        do {
            backup = try document.save(overrides,for:path)
            self.document = try ModelSettingsDocument(url:settingsURL)
            error = ""; notice = "Saved. Existing values for other models and advanced settings were preserved."
        } catch { self.error = error.localizedDescription }
    }
}
