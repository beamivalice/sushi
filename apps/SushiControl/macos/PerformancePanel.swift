import SwiftUI
import Charts

struct MetricTimeline:View {
    var title:String
    var unit:String
    var points:[TimelineValue]
    var domain:ClosedRange<Date>
    var color:Color
    var maximumGap:TimeInterval
    var emptyMessage = "Collecting samples…"
    var status:String? = nil
    @State private var hovered:Date?
    var selected:TimelineValue? {
        guard let hovered, let point = points.min(by:{abs($0.time.timeIntervalSince(hovered)) < abs($1.time.timeIntervalSince(hovered))}), abs(point.time.timeIntervalSince(hovered)) <= maximumGap else { return nil }
        return point
    }
    var body:some View {
        VStack(alignment:.leading,spacing:6) {
            HStack {
                HStack(spacing:6) { Circle().fill(color).frame(width:6,height:6); Text(title).font(.caption.weight(.medium)) }
                Spacer()
                if let selected {
                    Text("\(selected.time.formatted(date:.omitted,time:.standard)) · \(selected.value,format:.number.precision(.fractionLength(1))) \(unit)").font(.caption).monospacedDigit()
                } else if let status {
                    Text(status).font(.caption).foregroundStyle(.secondary)
                } else {
                    Text(points.last.map { String(format:"%.1f %@",$0.value,unit) } ?? "—").font(.callout.weight(.semibold)).monospacedDigit()
                }
            }
            if points.isEmpty {
                Text(emptyMessage).font(.caption).foregroundStyle(.secondary).frame(maxWidth:.infinity,minHeight:84,alignment:.leading)
            } else {
                Chart {
                    ForEach(points) { point in
                        LineMark(x:.value("Time",point.time),y:.value(unit,point.value),series:.value("Segment",point.segment)).foregroundStyle(color).lineStyle(StrokeStyle(lineWidth:2))
                        PointMark(x:.value("Time",point.time),y:.value(unit,point.value)).foregroundStyle(color).symbolSize(points.count == 1 ? 18 : 3)
                    }
                    if let selected {
                        RuleMark(x:.value("Sample time",selected.time)).foregroundStyle(.secondary.opacity(0.4)).lineStyle(StrokeStyle(dash:[3,3]))
                        PointMark(x:.value("Sample time",selected.time),y:.value(unit,selected.value)).foregroundStyle(color).symbolSize(35)
                    }
                }.chartXScale(domain:domain).chartYScale(domain:0...max(1,(points.map(\.value).max() ?? 0)*1.15))
                    .chartYAxis { AxisMarks(values:.automatic(desiredCount:3)) }
                    .chartXAxis { AxisMarks(values:.automatic(desiredCount:4)) { _ in AxisValueLabel(format:.dateTime.minute().second()); AxisGridLine() } }
                    .chartOverlay { proxy in
                        GeometryReader { geometry in
                            Rectangle().fill(.clear).contentShape(Rectangle()).onContinuousHover { phase in
                                switch phase {
                                case .active(let location):
                                    guard let plot = proxy.plotFrame else { hovered = nil; return }
                                    let frame = geometry[plot]
                                    hovered = frame.contains(location) ? proxy.value(atX:location.x-frame.minX,as:Date.self) : nil
                                case .ended: hovered = nil
                                }
                            }.onTapGesture { location in
                                guard let plot = proxy.plotFrame else { return }
                                let frame = geometry[plot]
                                hovered = frame.contains(location) ? proxy.value(atX:location.x-frame.minX,as:Date.self) : nil
                            }
                        }
                    }.frame(height:84).accessibilityLabel("\(title), \(unit), measured samples")
            }
        }
    }
}

struct PerformancePanel:View {
    @ObservedObject var store:ServerStore
    @State private var seconds = 180
    var body:some View {
        let end = Date()
        let domain = end.addingTimeInterval(-Double(seconds))...end
        let samples = store.history.samples.filter { domain.contains($0.time) }
        let prefill = store.prefillHistory.samples.filter { domain.contains($0.time) }
        let gap = ThroughputHistory.maximumInterval
        SushiCard {
            VStack(alignment:.leading,spacing:14) {
                HStack {
                    Label("Live performance",systemImage:"waveform.path.ecg").font(.headline)
                    Spacer()
                    if store.metricsTemporarilyUnavailable { Label("Reconnecting",systemImage:"wifi.exclamationmark").font(.caption).foregroundStyle(.orange) }
                    Picker("History",selection:$seconds) { Text("1 min").tag(60); Text("3 min").tag(180) }.pickerStyle(.segmented).labelsHidden().frame(width:140)
                }
                if store.metrics.isEmpty && !store.metricsTemporarilyUnavailable {
                    Text(store.backendSleeping ? "The server is sleeping. Live samples resume when it wakes." : "Enable metrics when starting Sushi to see live activity.").foregroundStyle(.secondary).frame(maxWidth:.infinity,minHeight:100)
                } else {
                    MetricTimeline(title:"Output",unit:"tok/s",points:TimelineValue.points(samples,value:{$0.tokensPerSecond},maximumGap:gap),domain:domain,color:.sushiAccent,maximumGap:store.refreshInterval*2.5,status:store.metricsTemporarilyUnavailable ? "Temporarily unavailable" : store.history.currentRate == nil ? "Waiting for sample" : nil)
                    Divider()
                    MetricTimeline(title:"Prefill",unit:"tok/s",points:TimelineValue.points(prefill,value:{$0.tokensPerSecond},maximumGap:gap),domain:domain,color:.teal,maximumGap:store.refreshInterval*2.5,emptyMessage:store.gauges["prefill_tokens_live"] == nil ? "Live prefill is unavailable from this server." : "Collecting prefill samples…",status:store.metricsTemporarilyUnavailable ? "Temporarily unavailable" : store.gauges["prefill_tokens_live"] == nil ? "Unavailable" : store.prefillHistory.currentRate == nil ? "Waiting for sample" : nil)
                    Divider()
                    MetricTimeline(title:"GPU allocation",unit:"GiB",points:TimelineValue.points(samples,value:{$0.memoryGiB},maximumGap:gap),domain:domain,color:.indigo,maximumGap:store.refreshInterval*2.5,emptyMessage:(store.props["memory"] as? [String:Any])?["active_bytes"] == nil ? "GPU allocation is not reported by this server." : "Collecting GPU samples…",status:store.metricsTemporarilyUnavailable ? "Temporarily unavailable" : (store.props["memory"] as? [String:Any])?["active_bytes"] == nil ? "Unavailable" : store.history.currentRate == nil ? "Waiting for sample" : nil)
                    Text("Sampled every \(Int(store.refreshInterval))s · prefill excludes cache restores · hover or click to inspect").font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }
}
struct ActivityPanel:View {
    @ObservedObject var store:ServerStore
    @State private var mode = "Running"
    var sessions:[[String:Any]] { store.metrics["sessions"] as? [[String:Any]] ?? [] }
    var filtered:[[String:Any]] { sessions.filter { mode == "All" || (mode == "Cached" ? $0["phase"] as? String == "cached" : $0["phase"] as? String != "cached") } }
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            HStack { Picker("Show",selection:$mode) { Text("Running").tag("Running"); Text("Cached").tag("Cached"); Text("All").tag("All") }.pickerStyle(.segmented).frame(width:300); Spacer(); Text("\(filtered.count) sessions").font(.caption).foregroundStyle(.secondary) }
            if !store.online || store.metrics.isEmpty {
                ContentUnavailableView("Activity unavailable",systemImage:"waveform.path",description:Text(store.metricsTemporarilyUnavailable ? "Live metrics are temporarily unavailable." : "Connect to a Sushi server with metrics enabled."))
            } else if filtered.isEmpty {
                ContentUnavailableView(mode == "Cached" ? "No cached conversations" : "No active requests",systemImage:mode == "Cached" ? "archivebox" : "checkmark.circle",description:Text("Sessions appear here as the server handles requests."))
            }
            ForEach(Array(filtered.enumerated()),id:\.offset) { _,session in
                SushiCard {
                    VStack(alignment:.leading,spacing:12) {
                        HStack { Text(session["model"] as? String ?? "Model").font(.headline).lineLimit(2); Spacer(); Text((session["phase"] as? String ?? "Unknown").capitalized).font(.caption.weight(.medium)).padding(.horizontal,8).padding(.vertical,4).background(Color.sushiAccent.opacity(0.1),in:Capsule()) }
                        let context = session["context_tokens"] as? Double ?? 0
                        let limit = session["context_length"] as? Double ?? 0
                        if limit > 0 { ProgressView(value:min(context,limit),total:limit).tint(context/limit > 0.9 ? .orange : .sushiAccent) }
                        HStack(spacing:20) {
                            stat("Context",number(context))
                            stat("Generated",number(session["generated_tokens"] as? Double ?? 0))
                            if let bytes = session["state_bytes"] as? Double { stat("KV memory",String(format:"%.0f MiB",bytes/1048576)) }
                            if let elapsed = session["elapsed_seconds"] as? Double, elapsed > 0 { stat("Elapsed",String(format:"%.0fs",elapsed)) }
                        }
                    }.padding(12)
                }
            }
        }
    }
    func number(_ value:Double)->String { value.formatted(.number.precision(.fractionLength(0))) }
    func stat(_ label:String,_ value:String)->some View { VStack(alignment:.leading,spacing:4) { Text(label).font(.caption).foregroundStyle(.secondary); Text(value).monospacedDigit() } }
}

struct StatValue:View {
    var title:String
    var value:String
    var note:String? = nil
    var body:some View {
        VStack(alignment:.leading,spacing:6) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(size:22,weight:.semibold,design:.rounded)).monospacedDigit()
            if let note { Text(note).font(.caption2).foregroundStyle(.secondary) }
        }.frame(maxWidth:.infinity,alignment:.leading)
    }
}
struct SystemStatusPanel:View {
    var system:SystemSnapshot
    var uptime:String {
        let formatter = DateComponentsFormatter(); formatter.allowedUnits = [.day,.hour,.minute]; formatter.unitsStyle = .abbreviated; formatter.maximumUnitCount = 2
        return formatter.string(from:system.uptime) ?? "Less than a minute"
    }
    func memory(_ value:Double?) -> String { value.map { String(format:"%.1f GiB",$0/1073741824) } ?? "—" }
    var body:some View {
        SushiCard {
            VStack(alignment:.leading,spacing:18) {
                HStack {
                    Label("This Mac",systemImage:"desktopcomputer").font(.headline)
                    Spacer()
                    Text(SystemSnapshot.processorName).font(.caption).foregroundStyle(.secondary)
                }
                HStack(alignment:.top,spacing:24) {
                    VStack(alignment:.leading,spacing:12) {
                        StatValue(title:"CPU",value:system.cpuUsagePercent.map { String(format:"%.0f%%",$0) } ?? "—",note:"\(ProcessInfo.processInfo.activeProcessorCount) cores · system-wide")
                        ProgressView(value:system.cpuUsagePercent ?? 0,total:100).tint((system.cpuUsagePercent ?? 0) > 90 ? .orange : .sushiAccent).opacity(system.cpuUsagePercent == nil ? 0 : 1).accessibilityLabel("System CPU utilization").accessibilityHidden(system.cpuUsagePercent == nil)
                    }.frame(maxWidth:.infinity,alignment:.leading)
                    VStack(alignment:.leading,spacing:12) {
                        StatValue(title:"GPU",value:system.gpuUsagePercent.map { String(format:"%.0f%%",$0) } ?? "—",note:system.gpuCoreCount.map { "\($0) cores · system-wide" } ?? "System-wide utilization")
                        ProgressView(value:system.gpuUsagePercent ?? 0,total:100).tint(.teal).opacity(system.gpuUsagePercent == nil ? 0 : 1).accessibilityLabel("System GPU utilization").accessibilityHidden(system.gpuUsagePercent == nil)
                    }.frame(maxWidth:.infinity,alignment:.leading)
                }
                Divider()
                HStack(alignment:.top,spacing:24) {
                    StatValue(title:"Physical memory",value:memory(system.physicalBytes),note:"Shared CPU and GPU memory")
                    StatValue(title:"Thermal state",value:system.thermal,note:"Reported by macOS").foregroundStyle(system.thermalWarning ? Color.orange : Color.primary)
                }
                DisclosureGroup("Hardware, memory & uptime") {
                    VStack(spacing:10) {
                        LabeledContent("GPU",value:system.gpuName ?? "Not reported")
                        LabeledContent("GPU cores",value:system.gpuCoreCount.map(String.init) ?? "Not reported")
                        LabeledContent("Wired memory",value:memory(system.wiredBytes))
                        LabeledContent("Compressed memory",value:memory(system.compressedBytes))
                        LabeledContent("Free pages",value:memory(system.freeBytes))
                        LabeledContent("Mac uptime",value:uptime)
                        Text("System-wide memory counts. macOS can also reclaim cached pages; free pages are not Sushi’s memory limit.").font(.caption2).foregroundStyle(.secondary).frame(maxWidth:.infinity,alignment:.leading)
                    }.font(.caption).padding(.top,10)
                }
            }.padding(12)
        }
    }
}
struct ServingStatsPanel:View {
    @ObservedObject var store:ServerStore
    var stats:ServingStats { ServingStats(store.metrics) }
    func count(_ value:Double?) -> String { value.map { $0.formatted(.number.precision(.fractionLength(0))) } ?? "—" }
    func rate(_ value:Double?) -> String { value.map { String(format:"%.1f tok/s",$0) } ?? "—" }
    var body:some View {
        SushiCard {
            VStack(alignment:.leading,spacing:18) {
                HStack {
                    Label("Serving statistics",systemImage:"chart.bar.xaxis").font(.headline)
                    Spacer()
                    Button("Export CSV…",systemImage:"square.and.arrow.up") { store.exportServingStats() }.disabled(!stats.available)
                }
                if !stats.available {
                    Text(store.backendSleeping ? "Statistics resume when Sushi wakes. System information stays available while it sleeps." : store.metricsTemporarilyUnavailable ? "Live statistics are temporarily unavailable." : "Start Sushi with --metrics to see serving totals and averages.").font(.callout).foregroundStyle(.secondary)
                } else {
                    HStack(alignment:.top,spacing:24) {
                        StatValue(title:"Prompt tokens",value:count(stats.promptTokens),note:"Including cache restores")
                        StatValue(title:"Generated tokens",value:count(stats.outputTokens),note:"Completed requests")
                        StatValue(title:"Cached tokens",value:count(stats.cachedTokens),note:stats.cachedFraction.map { String(format:"%.0f%% of prompt tokens",$0*100) })
                    }
                    Divider()
                    HStack(alignment:.top,spacing:24) {
                        StatValue(title:"Prompt processing",value:rate(stats.prefillRate),note:"Excludes cached tokens")
                        StatValue(title:"Generation",value:rate(stats.decodeRate),note:"Completed-request average")
                        StatValue(title:"First token",value:stats.meanTTFT.map { String(format:"%.2f s",$0) } ?? "—",note:"Mean, including queue time")
                    }
                    Divider()
                    HStack(spacing:16) {
                        Label(count(stats.totalRequests)+(stats.totalRequests == 1 ? " request" : " requests"),systemImage:"arrow.left.arrow.right")
                        Spacer()
                        Text(count(stats.successfulRequests)+" successful")
                        Text(count(stats.cancelledRequests)+" cancelled")
                    }.font(.caption).foregroundStyle(.secondary)
                    HStack(spacing:16) {
                        Text(count(stats.failedRequests)+" failed · "+count(stats.rejectedRequests)+" rejected")
                        Spacer()
                        if let hits = stats.cacheHitFraction { Text(String(format:"%.0f%% prefix lookup hit rate",hits*100)) }
                    }.font(.caption).foregroundStyle(.secondary)
                    Text("Current server run. Speed averages exclude cancelled and failed requests; prompt processing counts only tokens forwarded through the model.").font(.caption2).foregroundStyle(.secondary)
                }
            }.padding(12)
        }
    }
}

struct MenuPerformancePanel:View {
    @ObservedObject var store:ServerStore
    var body:some View {
        let end = Date()
        let domain = end.addingTimeInterval(-60)...end
        let gap = ThroughputHistory.maximumInterval
        let output = TimelineValue.points(store.history.samples.filter { domain.contains($0.time) },value:{$0.tokensPerSecond},maximumGap:gap)
        let prefill = TimelineValue.points(store.prefillHistory.samples.filter { domain.contains($0.time) },value:{$0.tokensPerSecond},maximumGap:gap)
        let outputPeak = output.map(\.value).max() ?? 0
        let prefillPeak = prefill.map(\.value).max() ?? 0
        let outputScale = max(1,outputPeak)
        let prefillScale = max(1,prefillPeak)
        let sampleMessage = !store.online ? "Connect to Sushi to view live speeds." : store.backendSleeping ? "Sampling resumes when Sushi wakes." : store.metricsTemporarilyUnavailable ? "Live metrics temporarily unavailable · showing history" : store.metrics.isEmpty ? "Start Sushi with --metrics to view live speeds." : store.gauges["prefill_tokens_live"] == nil ? "Prefill unavailable · last minute" : "Last minute · sampled every \(Int(store.refreshInterval))s"
        VStack(alignment:.leading,spacing:8) {
            Label("Live throughput",systemImage:"waveform.path").font(.caption.weight(.medium))
            HStack(spacing:14) {
                Label("Output",systemImage:"minus").foregroundStyle(Color.sushiAccent)
                Label("Prefill",systemImage:"line.horizontal.3").foregroundStyle(Color.teal)
                Spacer()
                Text("tok/s").foregroundStyle(.secondary)
            }.font(.caption2)
            if !store.online || store.backendSleeping || (store.metrics.isEmpty && !store.metricsTemporarilyUnavailable) {
                Text(store.backendSleeping ? "Server sleeping" : "Live metrics unavailable").font(.caption).foregroundStyle(.secondary).frame(maxWidth:.infinity,minHeight:60)
            } else if output.isEmpty && prefill.isEmpty { Text(store.metricsTemporarilyUnavailable ? "Live metrics temporarily unavailable" : "Collecting samples…").font(.caption).foregroundStyle(.secondary).frame(maxWidth:.infinity,minHeight:60) }
            else {
                Chart {
                    ForEach(output) { point in
                        LineMark(x:.value("Time",point.time),y:.value("Output scale",point.value/outputScale),series:.value("Series","output-\(point.segment)"))
                            .foregroundStyle(Color.sushiAccent).lineStyle(StrokeStyle(lineWidth:1.5))
                        if output.count == 1 { PointMark(x:.value("Time",point.time),y:.value("Output scale",point.value/outputScale)).foregroundStyle(Color.sushiAccent).symbolSize(20) }
                    }
                    ForEach(prefill) { point in
                        LineMark(x:.value("Time",point.time),y:.value("Prefill scale",point.value/prefillScale),series:.value("Series","prefill-\(point.segment)"))
                            .foregroundStyle(Color.teal).lineStyle(StrokeStyle(lineWidth:1.5,dash:[4,3]))
                        if prefill.count == 1 { PointMark(x:.value("Time",point.time),y:.value("Prefill scale",point.value/prefillScale)).foregroundStyle(Color.teal).symbolSize(20) }
                    }
                }.chartXScale(domain:domain).chartYScale(domain:0...1.1).chartXAxis(.hidden).chartYAxis(.hidden).frame(height:60)
                    .accessibilityElement(children:.ignore)
                    .accessibilityLabel("Live output and uncached prefill throughput during the last minute; each line uses its own scale")
                    .accessibilityValue("Output peak \(output.isEmpty ? "unavailable" : String(format:"%.0f tokens per second",outputPeak)); prefill peak \(prefill.isEmpty ? "unavailable" : String(format:"%.0f tokens per second",prefillPeak))")
            }
            Text(sampleMessage).font(.caption2).foregroundStyle(.secondary)
            if !output.isEmpty || !prefill.isEmpty { Text("Each line uses its own scale.").font(.caption2).foregroundStyle(.secondary) }
        }.padding(12).background(.quaternary.opacity(0.3),in:RoundedRectangle(cornerRadius:10))
    }
}
