import SwiftUI

struct SushiDiscovery {
    static func candidates(paths:[String]) -> [String] {
        var seen = Set<String>()
        return paths.compactMap { path in
            let url = URL(fileURLWithPath:path).resolvingSymlinksInPath()
            guard FileManager.default.isExecutableFile(atPath:url.path), (try? url.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile) == true, seen.insert(url.path).inserted else { return nil }
            return url.path
        }
    }
    static var installedCandidates:[String] {
        candidates(paths:[FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".local/bin/sushi").path,"/opt/homebrew/bin/sushi","/usr/local/bin/sushi"])
    }
}

struct SetupGuide:View {
    @ObservedObject var store:ServerStore
    @ObservedObject private var task:ToolTask
    init(store:ServerStore) { self.store = store; self.task = store.tool }
    private var cliReady:Bool { !store.toolBinary.isEmpty && FileManager.default.isExecutableFile(atPath:store.toolBinary) }
    var body:some View {
        SushiCard {
            VStack(alignment:.leading,spacing:18) {
                HStack(alignment:.top) {

                    VStack(alignment:.leading,spacing:5) { Text("Welcome to Sushi Control").font(.title2.bold()).accessibilityAddTraits(.isHeader); Text("Connect your server, choose your CLI and add a model.").foregroundStyle(.secondary) }
                    Spacer()
                }
                step("1. Connect a server",ready:store.online)
                if store.online { Text("Connected to \(store.connectedEndpoint)").font(.caption.monospaced()).textSelection(.enabled) }
                else { Text("Connect to a server already running on this Mac, select its launch agent, or configure Sushi Control to launch it. Sushi’s default port is 12345.").font(.caption).foregroundStyle(.secondary) }
                HStack {
                    Button("Connection Settings") { store.serverSection = "Connection"; store.page = .server }
                    Button("Use Launch Agent…") { store.chooseExistingAgent() }.disabled(store.owned)
                    Button("Configure Launcher") {
                        if cliReady && store.binary.isEmpty && !store.usesExistingAgent && !store.owned { store.binary = store.toolBinary }
                        store.serverSection = "Launch"; store.page = .server
                    }
                }
                Divider()
                step("2. Choose the Sushi CLI",ready:cliReady)
                if store.toolBinary.isEmpty {
                    Text("Choose your installed executable for downloads and runtime tools. Candidate paths are checked without running them.").font(.caption).foregroundStyle(.secondary)
                    ForEach(SushiDiscovery.installedCandidates,id:\.self) { path in
                        HStack { Text(path).font(.caption.monospaced()).lineLimit(2).textSelection(.enabled); Spacer(); Button("Use This CLI") { do { try store.setToolBinary(path) } catch { store.error = error.localizedDescription } } }
                    }
                } else { Text(store.toolBinary).font(.caption.monospaced()).textSelection(.enabled) }
                HStack {
                    Button("Choose CLI…") { store.chooseToolBinary() }
                    Button("Check CLI Version") {
                        do { try task.run(binary:store.toolBinary,arguments:["--version"],title:"Installed version",timeout:20); store.page = .runtime }
                        catch { store.error = error.localizedDescription }
                    }.disabled(!cliReady)
                    Spacer()
                    Link("Installation Guide",destination:URL(string:"https://github.com/beamivalice/sushi#install")!)
                }
                Divider()
                step("3. Add a model",ready:!store.models.isEmpty)
                Text(store.models.isEmpty ? "Download a supported pack, then rescan the server’s model library or load its folder. Model downloads can be large; review the CLI’s size and hardware recommendation first." : "\(store.models.count) registered · \(store.models.filter(\.loaded).count) loaded").font(.caption).foregroundStyle(.secondary)
                HStack { Button("Browse Downloads") { store.page = .downloads }; Button("Model Library") { store.page = .models }; Spacer(); Button(store.online && cliReady && !store.models.isEmpty ? "Finish Setup" : "Dismiss Guide") { store.finishSetup() }.buttonStyle(.borderedProminent) }
                Text("These shortcuts do not install software, download a model or start a server automatically.").font(.caption2).foregroundStyle(.secondary)
            }.padding(12).disabled(store.busy || task.running)
        }
    }
    private func step(_ title:String,ready:Bool) -> some View {
        Label(title,systemImage:ready ? "checkmark.circle.fill" : "circle").font(.headline).foregroundStyle(ready ? Color.green : Color.primary)
    }
}
