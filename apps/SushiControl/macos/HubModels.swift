import Foundation
import Combine

struct HubModelFile:Identifiable {
    let name:String
    let bytes:Int64?
    var id:String { name }
}
struct HubModel:Identifiable {
    let id:String
    let downloads:Int?
    let likes:Int?
    let license:String?
    let modified:Date?
    let tags:[String]
    let files:[HubModelFile]
    let fileCount:Int
    var pageURL:URL { URL(string:"https://huggingface.co/"+id)! }
    static func validID(_ id:String)->Bool {
        let parts = id.split(separator:"/",omittingEmptySubsequences:false)
        return id.utf8.count <= 256 && parts.count == 2 && parts[0].lowercased() == "beamster" && !parts[1].isEmpty && parts[1] != "." && parts[1] != ".." && parts[1].utf8.allSatisfy { (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || [45,46,95].contains($0) }
    }
    init?(_ row:[String:Any]) {
        guard let id = row["id"] as? String, Self.validID(id), row["private"] as? Bool != true else { return nil }
        self.id = id
        func count(_ value:Any?)->Int? { ModelEntry.samplingNumber(value).flatMap { $0 < Double(Int.max) && $0.rounded() == $0 ? Int($0) : nil } }
        downloads = count(row["downloads"]); likes = count(row["likes"])
        let card = row["cardData"] as? [String:Any] ?? [:]
        let licenseText = card["license"] as? String ?? (card["license"] as? [String])?.prefix(3).joined(separator:", ")
        license = licenseText.flatMap { $0.utf8.count <= 256 ? $0 : nil }
        if let text = row["lastModified"] as? String, text.count <= 80 {
            let formatter = ISO8601DateFormatter(); formatter.formatOptions = [.withInternetDateTime,.withFractionalSeconds]
            modified = formatter.date(from:text) ?? ISO8601DateFormatter().date(from:text)
        } else { modified = nil }
        tags = Array((row["tags"] as? [String] ?? []).filter{$0.utf8.count <= 80}.prefix(20))
        let siblings = row["siblings"] as? [[String:Any]] ?? []
        fileCount = siblings.count
        var seen = Set<String>()
        files = siblings.prefix(100).compactMap { file in
            guard let name = file["rfilename"] as? String, name.utf8.count <= 512, !name.unicodeScalars.contains(where:{CharacterSet.controlCharacters.contains($0)}), seen.insert(name).inserted else { return nil }
            return HubModelFile(name:name,bytes:count(file["size"]).map(Int64.init))
        }
    }
    static func parseList(_ data:Data) throws -> [HubModel] {
        guard let rows = try JSONSerialization.jsonObject(with:data) as? [[String:Any]] else { throw ServerFailure(message:"The publisher returned an unexpected model list.") }
        var seen = Set<String>()
        return rows.prefix(40).compactMap { row in guard let model = HubModel(row), seen.insert(model.id).inserted else { return nil }; return model }
    }
    func matchingPack(_ catalog:[DownloadPack])->DownloadPack? {
        func normalized(_ text:String)->String { String(text.lowercased().filter{$0.isASCII && ($0.isLetter || $0.isNumber)}) }
        let repo = normalized(String(id.split(separator:"/")[1]))
        let matches = catalog.filter { pack in
            let parts = pack.id.split(separator:":",omittingEmptySubsequences:false)
            guard parts.count == 2 else { return false }
            if pack.id == "qwen3.8-flash-next:2.75bpw" {
                return id == "beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw"
            }
            return normalized(String(parts[0])+"sushi"+String(parts[1])) == repo
        }
        return matches.count == 1 ? matches[0] : nil
    }
}
final class HubClient:Sendable {
    private let session:URLSession
    init(session:URLSession? = nil) {
        if let session { self.session = session }
        else {
            let config = URLSessionConfiguration.ephemeral
            config.httpShouldSetCookies = false; config.httpCookieStorage = nil; config.urlCredentialStorage = nil; config.urlCache = nil
            config.timeoutIntervalForRequest = 15; config.timeoutIntervalForResource = 30
            self.session = URLSession(configuration:config,delegate:LocalRedirectPolicy(),delegateQueue:nil)
        }
    }
    deinit { session.invalidateAndCancel() }
    static func searchRequest(_ query:String) throws -> URLRequest {
        guard query.utf8.count <= 160, !query.contains("\0") else { throw ServerFailure(message:"Use a shorter model search.") }
        var url = URLComponents(string:"https://huggingface.co/api/models")!
        url.queryItems = [URLQueryItem(name:"author",value:"beamster"),URLQueryItem(name:"search",value:query),URLQueryItem(name:"limit",value:"40"),URLQueryItem(name:"full",value:"true"),URLQueryItem(name:"sort",value:"downloads"),URLQueryItem(name:"direction",value:"-1")]
        var request = URLRequest(url:url.url!); request.timeoutInterval = 15; request.setValue("application/json",forHTTPHeaderField:"Accept"); return request
    }
    static func detailRequest(_ id:String) throws -> URLRequest {
        guard HubModel.validID(id) else { throw ServerFailure(message:"Choose a public beamster model repository.") }
        var url = URLComponents(string:"https://huggingface.co/api/models/"+id)!
        url.queryItems = [URLQueryItem(name:"blobs",value:"true")]
        var request = URLRequest(url:url.url!); request.timeoutInterval = 15; request.setValue("application/json",forHTTPHeaderField:"Accept"); return request
    }
    private func fetch(_ request:URLRequest) async throws -> Data {
        let (bytes,response) = try await session.bytes(for:request)
        defer { bytes.task.cancel() }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { throw ServerFailure(message:"The public model service returned HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0). Try again later.") }
        let limit = 2*1024*1024
        guard response.expectedContentLength <= limit else { throw ServerFailure(message:"The model metadata exceeds 2 MiB.") }
        var data = Data()
        for try await byte in bytes {
            try Task.checkCancellation()
            guard data.count < limit else { throw ServerFailure(message:"The model metadata exceeds 2 MiB.") }
            data.append(byte)
        }
        return data
    }
    func search(_ query:String) async throws -> [HubModel] { try HubModel.parseList(await fetch(Self.searchRequest(query))) }
    func detail(_ id:String) async throws -> HubModel {
        let data = try await fetch(Self.detailRequest(id))
        guard let row = try JSONSerialization.jsonObject(with:data) as? [String:Any], let model = HubModel(row), model.id == id else { throw ServerFailure(message:"The model service returned a different repository.") }
        return model
    }
}
@MainActor final class HubBrowser:ObservableObject {
    @Published private(set) var models:[HubModel] = []
    @Published private(set) var selected:HubModel?
    @Published private(set) var loading = false
    @Published private(set) var error = ""
    private let client = HubClient()
    private var operation:Task<Void,Never>?
    func cancel() { operation?.cancel(); operation = nil; loading = false }
    func search(_ query:String) {
        cancel(); loading = true; error = ""; selected = nil; models = []
        operation = Task {
            do { let rows = try await client.search(query); try Task.checkCancellation(); models = rows }
            catch { if !Task.isCancelled { self.error = error.localizedDescription } }
            if !Task.isCancelled { loading = false; operation = nil }
        }
    }
    func select(_ model:HubModel) {
        cancel(); selected = model; loading = true; error = ""
        operation = Task {
            do { let detail = try await client.detail(model.id); try Task.checkCancellation(); selected = detail }
            catch { if !Task.isCancelled { self.error = error.localizedDescription } }
            if !Task.isCancelled { loading = false; operation = nil }
        }
    }
}
