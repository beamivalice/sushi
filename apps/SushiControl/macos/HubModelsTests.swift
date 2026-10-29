import Foundation

@main struct HubModelsTests {
 static func main() async throws {
  let request = try HubClient.searchRequest("Flash Next")
  precondition(request.url?.host == "huggingface.co" && request.url?.scheme == "https" && request.value(forHTTPHeaderField:"Authorization") == nil)
  precondition(URLComponents(url:request.url!,resolvingAgainstBaseURL:false)!.queryItems!.contains(URLQueryItem(name:"author",value:"beamster")))
  for id in ["other/model","beamster/../secret","beamster/..","https://example.com/model","beamster/a?token=x"] {
   do { _ = try HubClient.detailRequest(id); fatalError("Accepted unsafe repository") } catch {}
  }
  let data = Data(#"[{"id":"beamster/Qwen3.8-Flash-Next-Sushi-2.6bpw","downloads":12,"likes":3,"lastModified":"2026-10-20T12:00:00.000Z","cardData":{"license":"apache-2.0"},"siblings":[{"rfilename":"config.json","size":100}]},{"id":"other/model"}]"#.utf8)
  let rows = try HubModel.parseList(data)
  precondition(rows.count == 1 && rows[0].downloads == 12 && rows[0].license == "apache-2.0" && rows[0].files.first?.bytes == 100 && rows[0].modified != nil)
  let catalog = DownloadPack.parse("qwen3.8-flash-next:2.6bpw 43.2 GB no fits")
  precondition(rows[0].matchingPack(catalog)?.id == catalog.first?.id)
  precondition(rows[0].matchingPack(catalog+catalog) == nil)
  let stq = HubModel(["id":"beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw"])!
  let stqCatalog = DownloadPack.parse("qwen3.8-flash-next:2.75bpw 45.0 GB no fits")
  precondition(stq.matchingPack(stqCatalog)?.id == "qwen3.8-flash-next:2.75bpw")
  precondition(stq.matchingPack(catalog) == nil)
  let wrongFormat = HubModel(["id":"beamster/Qwen3.8-Flash-Next-Sushi-2.75bpw"])!
  precondition(wrongFormat.matchingPack(stqCatalog) == nil)
  let config = URLSessionConfiguration.ephemeral; config.protocolClasses = [HubFixtureProtocol.self]
  let client = HubClient(session:URLSession(configuration:config))
  HubFixtureProtocol.body = data
  let fetched = try await client.search("Flash Next")
  precondition(fetched.count == 1)
  let firstRow = (try JSONSerialization.jsonObject(with:data) as! [[String:Any]])[0]
  HubFixtureProtocol.body = try JSONSerialization.data(withJSONObject:firstRow)
  let fetchedDetail = try await client.detail(rows[0].id)
  precondition(fetchedDetail.fileCount == 1)
  do { _ = try await client.detail("beamster/different"); fatalError("Accepted mismatched repository metadata") } catch {}
  HubFixtureProtocol.status = 302
  do { _ = try await client.search("x"); fatalError("Accepted redirect response") } catch {}
  HubFixtureProtocol.status = 200; HubFixtureProtocol.body = Data(repeating:32,count:2*1024*1024+1)
  do { _ = try await client.search("x"); fatalError("Accepted oversized response") } catch {}
  HubFixtureProtocol.body = data; HubFixtureProtocol.delay = 2
  let pending = Task { try await client.search("cancel") }
  try await Task.sleep(nanoseconds:50_000_000); pending.cancel()
  do { _ = try await pending.value; fatalError("Ignored cancellation") }
  catch is CancellationError {}
  catch let error as URLError { precondition(error.code == .cancelled) }
  HubFixtureProtocol.delay = 0
  if CommandLine.arguments.contains("--live") {
   let live = HubClient(); let found = try await live.search("Qwen3.8-Flash-Next-Sushi")
   precondition(!found.isEmpty)
   let detail = try await live.detail(found[0].id)
   precondition(detail.id == found[0].id && detail.fileCount > 0)
   print("Public Hub smoke passed: publisher search, details and file metadata; no weights downloaded")
  }
  print("Hub browsing passed: fixed HTTPS publisher, safe IDs, metadata, catalogue matching, auth isolation, non-200 responses, cancellation and response cap")
 }
}
final class HubFixtureProtocol:URLProtocol {
 static var body = Data()
 static var status = 200
 static var delay = 0.0
 private let lock = NSLock()
 private var stopped = false
 override class func canInit(with request:URLRequest)->Bool { request.url?.host == "huggingface.co" }
 override class func canonicalRequest(for request:URLRequest)->URLRequest { request }
 override func startLoading() {
  precondition(request.value(forHTTPHeaderField:"Authorization") == nil)
  let body = Self.body, status = Self.status, delay = Self.delay
  let deliver = {
   self.lock.lock(); defer { self.lock.unlock() }
   guard !self.stopped else { return }
   let response = HTTPURLResponse(url:self.request.url!,statusCode:status,httpVersion:"HTTP/1.1",headerFields:["Content-Type":"application/json","Content-Length":String(body.count)])!
   self.client?.urlProtocol(self,didReceive:response,cacheStoragePolicy:.notAllowed)
   self.client?.urlProtocol(self,didLoad:body); self.client?.urlProtocolDidFinishLoading(self)
  }
  if delay > 0 { DispatchQueue.global().asyncAfter(deadline:.now()+delay,execute:deliver) } else { deliver() }
 }
 override func stopLoading() { lock.lock(); stopped = true; lock.unlock() }
}
