import SwiftUI

private struct ScrollContentHeight:PreferenceKey {
    static var defaultValue:CGFloat { 0 }
    static func reduce(value:inout CGFloat,nextValue:()->CGFloat) { value = max(value,nextValue()) }
}
private struct ScrollViewportHeight:PreferenceKey {
    static var defaultValue:CGFloat { 0 }
    static func reduce(value:inout CGFloat,nextValue:()->CGFloat) { value = max(value,nextValue()) }
}
struct SushiScrollView<Content:View>:View {
    let content:Content
    @State private var contentHeight:CGFloat = 0
    @State private var viewportHeight:CGFloat = 0
    private var overflows:Bool { viewportHeight > 0 && contentHeight > viewportHeight+1 }
    init(@ViewBuilder content:()->Content) { self.content = content() }
    var body:some View {
        if #available(macOS 26.0,*) {
            scroll.scrollEdgeEffectStyle(.soft,for:.bottom).scrollEdgeEffectHidden(!overflows,for:.bottom)
        } else {
            scroll
        }
    }
    private var scroll:some View {
        ScrollView(.vertical,showsIndicators:true) {
            VStack(alignment:.leading,spacing:0) { content }
                .background(GeometryReader { geometry in Color.clear.preference(key:ScrollContentHeight.self,value:geometry.size.height) })
        }
            .background(GeometryReader { geometry in Color.clear.preference(key:ScrollViewportHeight.self,value:geometry.size.height) })
            .onPreferenceChange(ScrollContentHeight.self) { contentHeight = $0 }
            .onPreferenceChange(ScrollViewportHeight.self) { viewportHeight = $0 }
            .scrollIndicators(overflows ? .visible : .never,axes:.vertical)
            .scrollIndicatorsFlash(trigger:overflows)
    }
}

extension Color {
    static let sushiAccent = Color(nsColor:NSColor(name:nil) { appearance in
        appearance.bestMatch(from:[.darkAqua,.aqua]) == .darkAqua
            ? NSColor(calibratedRed:0.40,green:0.68,blue:1,alpha:1)
            : NSColor(calibratedRed:0.08,green:0.34,blue:0.72,alpha:1)
    })
}
struct SushiControlSurface:ViewModifier {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast
    func body(content:Content)->some View {
        if #available(macOS 26.0,*), !reduceTransparency, contrast != .increased {
            content.glassEffect(.regular,in:RoundedRectangle(cornerRadius:14))
        } else {
            content.background(Color(nsColor:.controlBackgroundColor),in:RoundedRectangle(cornerRadius:14))
        }
    }
}

struct SushiCard<Content:View>:View {
    @Environment(\.colorScheme) private var scheme
    var title:String?
    var content:Content
    init(_ title:String? = nil,@ViewBuilder content:()->Content) {
        self.title = title; self.content = content()
    }
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            if let title { Text(title).font(.subheadline.weight(.semibold)).foregroundStyle(.primary).accessibilityAddTraits(.isHeader) }
            content
        }.padding(18).frame(maxWidth:.infinity,alignment:.leading)
            .background(scheme == .dark ? Color.white.opacity(0.035) : Color.clear,in:RoundedRectangle(cornerRadius:14))
            .background(Color(nsColor:.controlBackgroundColor),in:RoundedRectangle(cornerRadius:14))
            .overlay(RoundedRectangle(cornerRadius:14).strokeBorder(.primary.opacity(0.06)))
            .shadow(color:.black.opacity(scheme == .dark ? 0.1 : 0.025),radius:6,x:0,y:2)
    }
}
struct PreferenceRow<Accessory:View>:View {
    var title:String
    var detail:String
    var icon:String
    var accessory:Accessory
    init(_ title:String,detail:String,icon:String,@ViewBuilder accessory:()->Accessory) {
        self.title = title; self.detail = detail; self.icon = icon; self.accessory = accessory()
    }
    var body:some View {
        HStack(alignment:.center,spacing:12) {
            Image(systemName:icon).font(.system(size:15,weight:.medium)).foregroundStyle(Color.sushiAccent)
                .frame(width:34,height:34).background(Color.sushiAccent.opacity(0.12),in:RoundedRectangle(cornerRadius:9)).accessibilityHidden(true)
            VStack(alignment:.leading,spacing:3) {
                Text(title).font(.callout.weight(.medium))
                Text(detail).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
            }.frame(maxWidth:.infinity,alignment:.leading)
            accessory
        }.padding(.vertical,10)
    }
}
struct AppearanceOption:View {
    var title:String
    var value:String
    var selected:Bool
    var action:()->Void
    var body:some View {
        Button(action:action) {
            VStack(spacing:8) {
                AppearanceSwatch(value:value).frame(height:64)
                Text(title).font(.caption.weight(selected ? .semibold : .medium)).foregroundStyle(selected ? Color.sushiAccent : Color.primary)
            }.padding(8).frame(maxWidth:.infinity)
                .background(selected ? Color.sushiAccent.opacity(0.1) : .clear,in:RoundedRectangle(cornerRadius:12))
                .overlay(RoundedRectangle(cornerRadius:12).strokeBorder(selected ? Color.sushiAccent : Color.primary.opacity(0.08),lineWidth:selected ? 1.5 : 1))
                .contentShape(RoundedRectangle(cornerRadius:12))
        }.buttonStyle(.plain).accessibilityLabel("\(title) appearance").accessibilityValue(selected ? "Selected" : "Not selected")
    }
}
private struct AppearanceSwatch:View {
    var value:String
    var body:some View {
        ZStack(alignment:.topLeading) {
            HStack(spacing:0) {
                (value == "dark" ? Color(red:0.16,green:0.17,blue:0.19) : Color(red:0.92,green:0.93,blue:0.95))
                (value == "light" ? Color.white : Color(red:0.16,green:0.17,blue:0.19))
            }
            VStack(alignment:.leading,spacing:7) {
                HStack(spacing:3) { Circle().fill(.red.opacity(0.75)); Circle().fill(.orange.opacity(0.75)); Circle().fill(.green.opacity(0.75)) }.frame(width:18,height:4)
                HStack(alignment:.top,spacing:7) {
                    RoundedRectangle(cornerRadius:2).fill(Color.gray.opacity(0.35)).frame(width:16,height:31)
                    VStack(alignment:.leading,spacing:4) {
                        RoundedRectangle(cornerRadius:2).fill(Color.gray.opacity(0.55)).frame(width:34,height:4)
                        RoundedRectangle(cornerRadius:2).fill(Color.gray.opacity(0.25)).frame(width:52,height:4)
                        RoundedRectangle(cornerRadius:2).fill(Color.gray.opacity(0.25)).frame(width:42,height:4)
                    }
                }
            }.padding(8)
        }.clipShape(RoundedRectangle(cornerRadius:8)).overlay(RoundedRectangle(cornerRadius:8).strokeBorder(Color.primary.opacity(0.1)))
            .accessibilityHidden(true)
    }
}
struct ServerHero:View {
    @ObservedObject var store:ServerStore
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack(alignment:.top,spacing:12) {
                VStack(alignment:.leading,spacing:5) {
                    HStack { ConnectionBadge(store:store); Text(store.backendSleeping ? "Engine sleeping" : "Local inference").font(.caption).foregroundStyle(.secondary) }
                    Text(store.selectedModel?.id ?? "No model selected").font(.headline).lineLimit(2).textSelection(.enabled).help(store.selectedModel?.id ?? "Select a model in Models")
                    HStack(spacing:8) {
                        Text(store.selectedModel.map { $0.loaded ? "Menu selection · loaded" : "Menu selection · not loaded" } ?? (store.online ? "Choose a model in Models" : "Connect to see models"))
                        Text("·")
                        Text(store.connectedEndpoint).lineLimit(1).truncationMode(.middle).help(store.connectedEndpoint)
                    }.font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength:0)
                Button("Models",systemImage:"square.stack.3d.up") { store.page = .models }
            }
            HStack {
                if store.owned || (!store.online && store.canStart) { Actions(store:store,compact:true) }
                else if !store.online { Button("Connect a Server",systemImage:"network") { store.page = .server; store.serverSection = "Connection" }.buttonStyle(SushiPrimaryButtonStyle()) }
                else { Text("Managed outside Sushi Control").font(.caption).foregroundStyle(.secondary) }
                Spacer()
                if store.online { Button("Dashboard",systemImage:"arrow.up.right.square") { store.dashboard() } }
            }
        }.padding(18).frame(maxWidth:.infinity,alignment:.leading).modifier(SushiControlSurface())
    }
}

struct MetricCard:View {
    var title:String
    var value:String
    var detail:String
    var icon:String
    var body:some View {
        SushiCard {
            VStack(alignment:.leading,spacing:12) {
                HStack(spacing:7) { Image(systemName:icon).foregroundStyle(Color.sushiAccent); Text(title).font(.caption.weight(.medium)).foregroundStyle(.secondary); Spacer(minLength:0) }
                Text(value).font(.system(size:28,weight:.semibold)).monospacedDigit().lineLimit(1).minimumScaleFactor(0.7)
                Text(detail).font(.caption2).foregroundStyle(.secondary)
            }.frame(maxWidth:.infinity,alignment:.leading).padding(4)
        }
    }
}
struct StatusDashboard:View {
    @ObservedObject var store:ServerStore
    var body:some View {
            VStack(alignment:.leading,spacing:16) {
                ServerHero(store:store)
                if !store.online {
                    if store.usesExistingAgent {
                        Feedback(text:store.serviceRunning ? "Waiting for the selected launch agent to accept connections." : "The selected launch agent is stopped. Use Start to reconnect.")
                    } else {
                    SushiCard {
                        HStack(alignment:.top,spacing:24) {
                            SetupOption(icon:"network",title:"Already running Sushi?",detail:"Connect using your server’s local address.",button:"Connection Settings") { store.page = .server; store.serverSection = "Connection" }
                            Divider()
                            SetupOption(icon:"play.circle",title:"Start a local server",detail:"Choose your Sushi installation and model folder.",button:"Set Up Sushi") { store.page = .server; store.serverSection = "Launch" }
                        }.padding(12)
                    }
                    }
                    if !store.connectionError.isEmpty && (!store.usesExistingAgent || store.serviceRunning) { Feedback(text:store.connectionError,failure:true) }
                } else {
                    if store.backendSleeping {
                        SushiCard("Ready when you need it") {
                            Text("The inference engine is sleeping. Live performance returns when it wakes; model management remains available.").font(.callout).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
                        }
                    } else {
                    HStack(spacing:12) { MetricCard(title:"Output",value:store.outputRate,detail:"Server-wide generation",icon:"waveform.path"); MetricCard(title:"Prefill",value:store.prefillRate,detail:"Uncached prompt work",icon:"text.alignleft"); MetricCard(title:"GPU allocation",value:store.memory,detail:"In use by Sushi",icon:"memorychip") }
                    PerformancePanel(store:store)
                    ServingStatsPanel(store:store)
                    PrefixCachePanel(props:store.props)
                    }
                    SushiCard("Model library") {
                        HStack { VStack(alignment:.leading,spacing:5) { Text("\(store.models.filter{$0.loaded}.count) loaded · \(store.models.count) registered").font(.headline); Text(store.selectedModel?.id ?? "No model selected").font(.caption).foregroundStyle(.secondary).lineLimit(2) }; Spacer(); Button("Manage Models") { store.page = .models } }.padding(12)
                    }
                    if store.metrics.isEmpty && !store.backendSleeping && !store.metricsTemporarilyUnavailable { Feedback(text:"For live request counts and cache statistics, launch Sushi with --metrics.") }
                }
                SystemStatusPanel(system:store.system)
                HStack { Button("Copy API URL",systemImage:"doc.on.doc") { store.copyEndpoint() }; Spacer(); if let updated = store.lastUpdated { Text("Updated \(updated.formatted(date:.omitted,time:.standard))").font(.caption).foregroundStyle(.secondary) } }
            }
    }
}

struct SetupOption:View {
    var icon:String
    var title:String
    var detail:String
    var button:String
    var action:()->Void
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            Image(systemName:icon).font(.title2).foregroundStyle(Color.sushiAccent).padding(12).background(Color.sushiAccent.opacity(0.08),in:RoundedRectangle(cornerRadius:12))
            VStack(alignment:.leading,spacing:6) { Text(title).font(.headline); Text(detail).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true) }
            Button(button,action:action)
        }.frame(maxWidth:.infinity,alignment:.leading)
    }
}

struct SushiPrimaryButtonStyle:ButtonStyle {
    @Environment(\.colorScheme) private var scheme
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration:Configuration)->some View {
        configuration.label.font(.callout.weight(.semibold)).padding(.horizontal,14).padding(.vertical,9)
            .foregroundStyle(scheme == .dark ? Color(red:0.03,green:0.09,blue:0.17) : .white)
            .background(Color.sushiAccent.opacity(enabled ? (configuration.isPressed ? 0.8 : 1) : 0.3),in:RoundedRectangle(cornerRadius:9))
            .opacity(enabled ? 1 : 0.6)
    }
}

struct PrefixCachePanel:View {
    var props:[String:Any]
    var cache:PrefixCacheStatus { PrefixCacheStatus(props) }
    func memory(_ bytes:Double)->String { String(format:"%.1f GiB",bytes/1073741824) }
    var body:some View {
        if cache.available {
            SushiCard {
                VStack(alignment:.leading,spacing:14) {
                    Label("Prefix cache",systemImage:"externaldrive").font(.headline)
                    HStack {
                        Text("RAM budget").foregroundStyle(.secondary)
                        Spacer()
                        if cache.ramEnabled == false { Text("Disabled") }
                        else if let budget = cache.ramBudget { Text(memory(budget)).monospacedDigit() }
                        else { Text(cache.ramEnabled == true ? "Enabled" : "Unavailable") }
                    }
                    if let budget = cache.diskBudget {
                        HStack {
                            Text("SSD cache").foregroundStyle(.secondary)
                            Spacer()
                            if budget == 0 { Text("Disabled") }
                            else if let used = cache.diskUsed { Text(memory(used)+" / "+memory(budget)).monospacedDigit() }
                            else { Text(memory(budget)+" budget").monospacedDigit() }
                        }
                        if let fraction = cache.diskFraction {
                            GeometryReader { geometry in
                                Capsule().fill(Color.secondary.opacity(0.12))
                                    .overlay(alignment:.leading) { Capsule().fill(fraction > 0.9 ? Color.orange : .sushiAccent).frame(width:geometry.size.width*fraction) }
                            }.frame(height:7).accessibilityLabel("SSD cache used").accessibilityValue(String(format:"%.0f percent",fraction*100))
                        }
                    }
                    if let entries = cache.diskEntries { Text("\(entries.formatted()) entries on SSD").font(.caption).foregroundStyle(.secondary) }
                    Text("Cache allocation for the server’s default model. RAM shows its budget, not current usage.").font(.caption2).foregroundStyle(.secondary)
                }.padding(12)
            }
        }
    }
}
