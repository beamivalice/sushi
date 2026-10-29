import SwiftUI

struct AboutPanel:View {
    @State private var selectedLicense = "LICENSE"
    private var version:String { Bundle.main.object(forInfoDictionaryKey:"CFBundleShortVersionString") as? String ?? "0.1" }
    private var license:String {
        guard let path = Bundle.main.path(forResource:selectedLicense,ofType:nil), let text = try? String(contentsOfFile:path,encoding:.utf8) else { return "License text is unavailable in this bundle." }
        return text
    }
    var body:some View {
        VStack(alignment:.leading,spacing:20) {
            HStack(spacing:18) {

                VStack(alignment:.leading,spacing:6) { Text("Sushi Control").font(.title.bold()).accessibilityAddTraits(.isHeader); Text("Version \(version) · Apple Silicon").foregroundStyle(.secondary); Text("Native controls for your local Sushi server.") }
                Spacer()
            }
            HStack {
                Link("Sushi on GitHub",destination:URL(string:"https://github.com/beamivalice/sushi")!)
                Spacer()
                Link("Documentation",destination:URL(string:"https://github.com/beamivalice/sushi/blob/main/docs/usage.md")!)
                Spacer()
                Link("Report an Issue",destination:URL(string:"https://github.com/beamivalice/sushi/issues")!)
            }
            Divider()
            Text("Licenses & acknowledgements").font(.headline)
            Picker("Document",selection:$selectedLicense) { Text("MIT License").tag("LICENSE"); Text("Apache 2.0").tag("LICENSE-APACHE-2.0"); Text("Third-party notices").tag("NOTICE") }.pickerStyle(.segmented)
            SushiScrollView { Text(license).font(.system(.caption,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(14) }.background(Color(nsColor:.textBackgroundColor),in:RoundedRectangle(cornerRadius:10))
        }.padding(24).frame(minWidth:620,minHeight:540)
    }
}
