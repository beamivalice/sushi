import SwiftUI
import AppKit

struct ConversionPanel:View {
    @ObservedObject var store:ServerStore
    @ObservedObject var task:ToolTask
    @State private var hours = 4
    @State private var confirm = false
    @State private var sourceInfo:ConversionSourceInfo?
    @State private var sourceError = ""
    private var supported:Bool { task.converterBinary == store.toolBinary && task.converterAvailable == true }
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            SushiCard("Sushi EXL3 conversion") {
                VStack(alignment:.leading,spacing:14) {
                    Text("Convert supported source checkpoints into Sushi packs with your installed CLI. Check support to see the converter’s architectures and options.").font(.callout).foregroundStyle(.secondary)
                    HStack { Text(store.toolBinary.isEmpty ? "Choose a Sushi CLI" : store.toolBinary).font(.caption.monospaced()).lineLimit(2).textSelection(.enabled); Spacer(); Button("Choose CLI…") { store.chooseToolBinary() } }
                    Button("Check Converter Support") { help() }.disabled(store.toolBinary.isEmpty)
                    if task.converterBinary == store.toolBinary, let available = task.converterAvailable {
                        Label(available ? "Native converter available" : "This CLI did not advertise EXL3 conversion",systemImage:available ? "checkmark.circle" : "exclamationmark.circle").font(.caption).foregroundStyle(.secondary)
                    }
                    HStack { TextField("Source checkpoint folder",text:$store.conversionPlan.source); Button("Choose Source…") { chooseSource() } }
                    if let sourceInfo {
                        LabeledContent("Source metadata",value:sourceInfo.architecture+" · "+(sourceInfo.precision ?? "Precision unreported")).font(.caption)
                        if sourceInfo.isSushiPack { Label("Already a Sushi pack. Choose the original checkpoint.",systemImage:"exclamationmark.triangle").font(.caption).foregroundStyle(.orange) }
                    }
                    if !sourceError.isEmpty { Text(sourceError).font(.caption).foregroundStyle(.orange) }
                    HStack { TextField("Output pack folder",text:$store.conversionPlan.output); Button("Choose Output…") { chooseOutput() } }
                    Toggle("Resume an interrupted conversion in the output folder",isOn:$store.conversionPlan.resume)
                    HStack { Text("Expert bits per weight"); Slider(value:$store.conversionPlan.expertBits,in:1.5...8,step:0.125); Text(store.conversionPlan.expertBits,format:.number.precision(.fractionLength(3))).monospacedDigit().frame(width:60) }
                    HStack { LabeledContent("Calibration tokens") { TextField("Calibration tokens",value:$store.conversionPlan.calibrationTokens,format:.number.grouping(.never)).frame(width:130) }; Spacer(); Stepper("Batch: \(store.conversionPlan.batch)",value:$store.conversionPlan.batch,in:1...64) }
                    DisclosureGroup("Advanced conversion settings") {
                        VStack(alignment:.leading,spacing:12) {
                            Picker("Codebook",selection:$store.conversionPlan.codebook) { Text("MCG").tag("mcg"); Text("Mul1").tag("mul1") }
                            Picker("Search window",selection:$store.conversionPlan.window) { ForEach(12...16,id:\.self) { Text(String($0)).tag($0) } }
                            Picker("Trunk bits",selection:$store.conversionPlan.trunkBits) { Text("Model default").tag(0); ForEach([2,3,4,5,6,8],id:\.self) { Text(String($0)).tag($0) } }
                            Picker("Trunk group",selection:$store.conversionPlan.trunkGroup) { Text("Model default").tag(0); ForEach([32,64,128],id:\.self) { Text(String($0)).tag($0) } }
                            Picker("Factor precision",selection:$store.conversionPlan.factorPrecision) { Text("64-bit CPU (default)").tag(64); Text("32-bit GPU").tag(32) }
                            Text("32-bit GPU factoring changes the resulting pack bytes.").font(.caption).foregroundStyle(.secondary)
                            HStack { TextField("Optional calibration text file",text:$store.conversionPlan.calibrationFile); Button("Choose Text…") { chooseCalibration() } }
                        }.padding(.top,10)
                    }
                    HStack {
                        Picker("Deadline",selection:$hours) { ForEach([1,4,12,24,72],id:\.self) { Text("\($0) hours").tag($0) } }.frame(width:210)
                        Spacer(); Button("Convert…",systemImage:"arrow.triangle.2.circlepath") {
                            do { _ = try store.conversionPlan.arguments(syntax:task.converterSyntax ?? .legacy); confirm = true } catch { store.error = error.localizedDescription }
                        }.buttonStyle(.borderedProminent).disabled(!supported || !store.canRunOfflineMaintenance || sourceInfo == nil || sourceInfo?.isSushiPack == true)
                    }
                    Text("Stop inference servers before conversion. Conversion uses CPU/GPU and can run for hours; incomplete output can be resumed. A new conversion requires an empty output folder.").font(.caption).foregroundStyle(.secondary)
                }.padding(12).disabled(store.busy || task.running)
            }
            TaskActivityPanel(store:store,task:task)
        }.task(id:store.conversionPlan.source) { await inspectSource(store.conversionPlan.source) }
        .confirmationDialog("Start EXL3 conversion?",isPresented:$confirm,titleVisibility:.visible) {
            Button("Start Conversion") { run() }
        } message: { Text("The selected source checkpoint will be read and the output folder will hold the converted pack. The task stops after \(hours) hours unless completed or cancelled.") }
    }
    private func help() {
        guard !store.busy else { return }
        do { try task.run(binary:store.toolBinary,arguments:["convert","--help"],title:"Converter support",timeout:20) } catch { store.error = error.localizedDescription }
    }
    private func inspectSource(_ path:String) async {
        sourceInfo = nil; sourceError = ""
        guard !path.isEmpty else { return }
        do {
            try await Task.sleep(nanoseconds:150_000_000)
            let info = try await Task.detached(priority:.utility) { try ConversionSourceInfo.read(path) }.value
            try Task.checkCancellation()
            guard path == store.conversionPlan.source else { return }
            sourceInfo = info
        } catch is CancellationError {
        } catch {
            if !Task.isCancelled && path == store.conversionPlan.source { sourceError = error.localizedDescription }
        }
    }
    private func run() {
        Task {
            await store.refresh()
            guard store.canRunOfflineMaintenance, supported else { store.error = "Stop and unregister the server launcher, wait for tools to finish, and check converter support before converting."; return }
            do { try task.run(binary:store.toolBinary,arguments:store.conversionPlan.arguments(syntax:task.converterSyntax ?? .legacy),title:"EXL3 conversion",timeout:Double(hours*3600)) } catch { store.error = error.localizedDescription }
        }
    }
    private func chooseSource() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url { store.conversionPlan.source = url.path }
    }
    private func chooseOutput() {
        if store.conversionPlan.resume {
            let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
            if panel.runModal() == .OK, let url = panel.url { store.conversionPlan.output = url.path }
        } else {
            let panel = NSSavePanel(); panel.nameFieldStringValue = "sushi-exl3-pack"; panel.title = "New output pack folder"
            if panel.runModal() == .OK, let url = panel.url { store.conversionPlan.output = url.path }
        }
    }
    private func chooseCalibration() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = false; panel.canChooseFiles = true
        if panel.runModal() == .OK, let url = panel.url { store.conversionPlan.calibrationFile = url.path }
    }
}
