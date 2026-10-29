import SwiftUI
import AppKit

struct BenchmarkPanel:View {
    @ObservedObject var store:ServerStore
    @ObservedObject var runner:BenchmarkRunner
    @State private var model = ""
    @State private var words = 256
    @State private var outputTokens = 128
    @State private var rounds = 1
    @State private var reuse = false
    @State private var timeout = 180.0
    @State private var showSummary = true
    @State private var confirmClear = false
    @State private var confirmReload = false
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            SushiCard("Run a benchmark") {
                VStack(alignment:.leading,spacing:14) {
                    Picker("Model",selection:$model) {
                        Text("Choose a model").tag("")
                        ForEach(store.models.filter { $0.capabilities.isEmpty || $0.capabilities.contains("chat") }) { Text($0.id).tag($0.id) }
                    }
                    HStack {
                        Picker("Reference words",selection:$words) { ForEach([256,2048,8192,32768],id:\.self) { Text($0.formatted()).tag($0) } }
                        Picker("Output limit",selection:$outputTokens) { ForEach([32,128,512],id:\.self) { Text("\($0) tokens").tag($0) } }
                    }
                    HStack {
                        Stepper("Runs: \(rounds)",value:$rounds,in:1...5)
                        Picker("Per-run deadline",selection:$timeout) { Text("30 seconds").tag(30.0); Text("3 minutes").tag(180.0); Text("10 minutes").tag(600.0) }
                    }
                    Toggle("Repeat the same prompt across runs",isOn:$reuse)
                    Text("Reference words are not tokens. Actual token counts come from the response. Unique prompts reduce prefix reuse; this does not flush caches or guarantee cold weights. Temperature is 0 with thinking requested off. Models may stop before the output limit.").font(.caption).foregroundStyle(.secondary)
                    Button("Run Benchmark",systemImage:"speedometer") {
                        store.runBenchmark(BenchmarkPlan(model:model,words:words,outputTokens:outputTokens,rounds:rounds,reusePrompt:reuse,timeout:timeout))
                    }.buttonStyle(.borderedProminent).disabled(model.isEmpty || !store.online || runner.historyBlocked)
                    Text("This sends real inference requests and may load or wake the selected model. Use an idle server for comparable results.").font(.caption).foregroundStyle(.secondary)
                }.padding(12).disabled(store.busy || store.tool.running || runner.running)
            }
            if !runner.status.isEmpty {
                HStack { if runner.running { ProgressView().controlSize(.small) }; Text(runner.status).font(.callout); Spacer(); if runner.running { Button("Cancel Benchmark",role:.destructive) { runner.cancel() } } }
            }
            if let error = runner.persistenceError {
                VStack(alignment:.leading,spacing:8) {
                    Label(error,systemImage:"exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                    if runner.historyBlocked {
                        HStack { Button("Reload Saved History") { if runner.results.isEmpty { runner.reloadHistory() } else { confirmReload = true } }; Button("Back Up & Reset History…") { confirmClear = true } }.disabled(runner.running)
                    }
                }
            }
            if let backup = runner.backupURL { Button("Reveal History Backup",systemImage:"folder") { NSWorkspace.shared.activateFileViewerSelecting([backup]) } }
            if runner.results.isEmpty {
                ContentUnavailableView("No benchmark results",systemImage:"speedometer",description:Text("Run a bounded test to compare prompt processing, generation and prefix reuse."))
            } else {
                HStack { Text("Recent results").font(.headline); Text("Up to 100 saved runs").font(.caption).foregroundStyle(.secondary); Spacer(); Button("Export JSON…",systemImage:"square.and.arrow.up") { export() }; Button("Clear Results",role:.destructive) { confirmClear = true }.disabled(runner.running) }
                Text("Prefill and decode use server timings when supplied. End-to-end includes queueing, tokenization, prompt processing and transfer. These rates measure different parts of a request.").font(.caption).foregroundStyle(.secondary)
                Picker("Results view",selection:$showSummary) { Text("Comparison summaries").tag(true); Text("Individual runs").tag(false) }.pickerStyle(.segmented).frame(maxWidth:400)
                if showSummary {
                    Text("Medians group matching server, model, pre-run configuration, reference size, output limit, actual token counts, cache usage and completion reason. The count beside each metric shows how many samples reported it. Older results without a server stay separate.").font(.caption).foregroundStyle(.secondary)
                    ForEach(BenchmarkSummary.group(runner.results)) { summary in summaryRow(summary) }
                } else {
                ForEach(runner.results) { result in
                    SushiCard {
                        VStack(alignment:.leading,spacing:12) {
                            HStack { Text(result.model).font(.headline).textSelection(.enabled); Spacer(); Text(result.date,format:.dateTime.month().day().hour().minute().second()).font(.caption).foregroundStyle(.secondary) }
                            HStack {
                                value("Prefill",rate(result.prefillRate)); Spacer(); value("Decode",rate(result.decodeRate)); Spacer(); value("End-to-end",rate(result.wallRate)); Spacer(); value("Duration",String(format:"%.2f s",result.wallSeconds))
                            }
                            Text("Prompt: \(result.promptTokens.map(String.init) ?? "—") tokens · cached: \(result.cachedTokens.map(String.init) ?? "—") · output: \(result.outputTokens.map(String.init) ?? "—") / \(result.outputLimit) · \(result.finishReason)").font(.caption).foregroundStyle(.secondary)
                            if !result.configuration.isEmpty { Text("Pre-run metadata: \(result.configuration)").font(.caption2).foregroundStyle(.secondary).textSelection(.enabled) }
                            Text("\(result.words.formatted()) reference words · \(result.repeatedPrompt ? "Repeated prompt" : "Unique prompt") · run \(result.round)").font(.caption).foregroundStyle(.secondary)
                        }.padding(12)
                    }
                }
                }
            }
        }.confirmationDialog("Reset benchmark history?",isPresented:$confirmClear,titleVisibility:.visible) {
            Button("Back Up & Reset",role:.destructive) { runner.clear() }
        } message: { Text("The current history file will be kept in a local backup. In-memory results will be cleared; export them first if you want to keep them separately.") }
        .confirmationDialog("Reload saved history?",isPresented:$confirmReload,titleVisibility:.visible) {
            Button("Reload Saved History") { runner.reloadHistory() }
        } message: { Text("This replaces the in-memory results with the current saved file. Export unsaved results first if you want to keep them separately.") }
    }
    private func summaryRow(_ summary:BenchmarkSummary)->some View {
        SushiCard {
            VStack(alignment:.leading,spacing:12) {
                HStack { Text(summary.key.model).font(.headline); Spacer(); Text("\(summary.samples.count) samples").font(.caption).foregroundStyle(.secondary) }
                if let first = summary.samples.map(\.date).min(), let last = summary.samples.map(\.date).max() {
                    Text(first == last ? first.formatted(date:.abbreviated,time:.shortened) : first.formatted(date:.abbreviated,time:.shortened)+" – "+last.formatted(date:.abbreviated,time:.shortened)).font(.caption2).foregroundStyle(.secondary)
                }
                Text(summary.key.endpoint ?? "Server not recorded").font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                Text("\(summary.key.words.formatted()) reference words · output cap \(summary.key.outputLimit) · \(summary.key.repeatedPrompt ? "Repeated prompt" : "Unique prompt") · \(summary.key.cache.title)").font(.caption).foregroundStyle(.secondary)
                HStack {
                    value("Median prefill (\(summary.prefill.count))",rate(summary.prefill.median)); Spacer()
                    value("Median decode (\(summary.decode.count))",rate(summary.decode.median)); Spacer()
                    value("Median end-to-end (\(summary.wall.count))",rate(summary.wall.median))
                }
                Text("Actual prompt \(summary.range(summary.samples.map(\.promptTokens))) · cached \(summary.range(summary.samples.map(\.cachedTokens))) · output \(summary.range(summary.samples.map(\.outputTokens))) tokens · \(summary.key.finishReason)").font(.caption).foregroundStyle(.secondary)
                if !summary.key.configuration.isEmpty { DisclosureGroup("Pre-run configuration") { Text(summary.key.configuration).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) } }
            }.padding(12)
        }
    }
    private func rate(_ value:Double?) -> String { value.map { String(format:"%.1f tok/s",$0) } ?? "—" }
    private func value(_ label:String,_ text:String) -> some View { VStack(alignment:.leading,spacing:4) { Text(label).font(.caption).foregroundStyle(.secondary); Text(text).font(.headline.monospacedDigit()) } }
    private func export() {
        let panel = NSSavePanel(); panel.nameFieldStringValue = "sushi-benchmarks.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try runner.exportData().write(to:url,options:.atomic) } catch { store.error = error.localizedDescription }
    }
}
