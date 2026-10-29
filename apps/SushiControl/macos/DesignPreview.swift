import SwiftUI

@main struct DesignPreview {
    @MainActor static func main() throws {
        let suite = "org.sushi.design-preview."+UUID().uuidString
        let defaults = UserDefaults(suiteName:suite)!
        defer { defaults.removePersistentDomain(forName:suite) }
        let store = ServerStore(defaults:defaults,automaticallyRefresh:false)
        for connected in [false,true] {
        store.online = connected
        if connected {
            store.models = [ModelEntry(id:"Example model",loaded:true,state:"ready",architecture:"",quantization:"4-bit",contextTokens:131072,residentBytes:32*1073741824)]
            store.selected = "Example model"
            store.props = ["memory":["active_bytes":36.0*1073741824],"settings":["prefix_cache":["ram_enabled":true,"mem_bytes":8.0*1073741824,"disk_bytes":20.0*1073741824,"disk_used_bytes":6.0*1073741824,"disk_entries":24]]]
            store.metrics = ["gauges":["requests_running":1,"generation_tokens_live":2400.0,"prefill_tokens_live":72000.0],"counters":["prompt_tokens_total":10000.0,"prefix_cache_tokens_total":8400.0]]
            let now = Date()
            for index in 0...20 { store.history.record(time:now.addingTimeInterval(Double(index-20)*3),tokens:Double(index*120),memoryBytes:36*1073741824) }
            for index in 0...20 {
                store.prefillHistory.record(time:now.addingTimeInterval(Double(index-20)*3),tokens:Double(index*3600))
            }
        }
        for dark in [false,true] {
            NSApplication.shared.appearance = NSAppearance(named:dark ? .darkAqua : .aqua)
            let content = VStack(alignment:.leading,spacing:24) {
                HStack { Text("Status").font(.system(size:30,weight:.bold,design:.rounded)); Spacer(); }
                StatusDashboard(store:store)
            }.padding(28).frame(width:780).background(Color(nsColor:.windowBackgroundColor)).tint(.sushiAccent).environment(\.colorScheme,dark ? .dark : .light)
            let renderer = ImageRenderer(content:content)
            renderer.scale = 2
            guard let image = renderer.cgImage else { throw ServerFailure(message:"Could not render design preview") }
            let png = NSBitmapImageRep(cgImage:image).representation(using:.png,properties:[:])!
            try png.write(to:URL(fileURLWithPath:CommandLine.arguments[1]).appendingPathComponent((connected ? "connected-" : "")+(dark ? "design-dark.png" : "design-light.png")))
            if connected {
                let menu = MenuPerformancePanel(store:store).padding(16).frame(width:390).background(Color(nsColor:.windowBackgroundColor)).environment(\.colorScheme,dark ? .dark : .light)
                let menuRenderer = ImageRenderer(content:menu)
                menuRenderer.scale = 2
                guard let menuImage = menuRenderer.cgImage, let menuPNG = NSBitmapImageRep(cgImage:menuImage).representation(using:.png,properties:[:]) else { throw ServerFailure(message:"Could not render menu preview") }
                try menuPNG.write(to:URL(fileURLWithPath:CommandLine.arguments[1]).appendingPathComponent(dark ? "menu-dark.png" : "menu-light.png"))
            }
        }
        }
    }
}
