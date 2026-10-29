import SwiftUI

struct IntegrationsPanel:View {
    @ObservedObject var store:ServerStore
    @AppStorage("integrationClient") private var client = "pi"
    @State private var model = ""
    var chatModels:[ModelEntry] { store.models.filter { $0.capabilities.isEmpty || $0.capabilities.contains("chat") } }
    var command:Result<String,Error> { Result { try AgentLaunchCommand.make(binary:store.toolBinary,client:client,endpoint:store.connectedEndpoint,model:model.isEmpty ? nil : model) } }
    var openAIEndpoint:String { (try? Connection(endpoint:store.connectedEndpoint).request("v1").url?.absoluteString) ?? "—" }
    var body:some View {
        VStack(alignment:.leading,spacing:20) {
            SushiCard("Launch a client") {
                VStack(alignment:.leading,spacing:16) {
                    Picker("Client",selection:$client) { ForEach(LaunchClient.allCases) { Text($0.title).tag($0.rawValue) } }
                    Picker("Model",selection:$model) {
                        Text("Server default").tag("")
                        ForEach(ModelEntry.library(chatModels)) { Text($0.id).tag($0.id) }
                    }
                    HStack {
                        Text(store.toolBinary.isEmpty ? "Uses sushi from your terminal’s PATH" : store.toolBinary).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                        Spacer()
                        Button("Choose CLI…") { store.chooseToolBinary() }.disabled(store.busy || store.tool.running)
                    }
                    switch command {
                    case .success(let text):
                        Text(text).font(.system(.callout,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(14).background(Color(nsColor:.textBackgroundColor),in:RoundedRectangle(cornerRadius:10))
                        Button("Copy Launch Command",systemImage:"doc.on.doc") { store.copyText(text,label:"Launch command") }.buttonStyle(SushiPrimaryButtonStyle())
                    case .failure(let error):
                        Feedback(text:error.localizedDescription,failure:true)
                    }
                    Text("Run the command in your terminal with the client installed. Sushi creates a dedicated configuration under ~/.sushi/<client> and starts the client. Copying does not run it.").font(.caption).foregroundStyle(.secondary)
                    if !store.online { Text("Start Sushi before running the command. The model list refreshes after connecting.").font(.caption).foregroundStyle(.secondary) }
                }.padding(12)
            }
            SushiCard("API endpoints") {
                VStack(alignment:.leading,spacing:14) {
                    LabeledContent("OpenAI compatible") { Text(openAIEndpoint).font(.caption.monospaced()).textSelection(.enabled) }
                    LabeledContent("Anthropic compatible") { Text(store.connectedEndpoint).font(.caption.monospaced()).textSelection(.enabled) }
                    HStack {
                        Button("Copy OpenAI URL") { store.copyEndpoint() }
                        Button("Copy Anthropic URL") { store.copyText(store.connectedEndpoint,label:"Anthropic endpoint") }
                        Spacer()
                        Button("Connection Settings") { store.page = .server; store.serverSection = "Connection" }
                    }
                    Text("The native launcher needs a server that trusts local connections. For strict API-key servers, configure these URLs and your key directly in the client. API keys are never included in copied commands.").font(.caption).foregroundStyle(.secondary)
                }.padding(12)
            }
        }
        .onAppear { if LaunchClient(rawValue:client) == nil { client = "pi" } }
        .onChange(of:store.models.map(\.id)) { _,ids in if !model.isEmpty && !ids.contains(model) { model = "" } }
    }
}
