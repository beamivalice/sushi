import Foundation
import Darwin

enum ConverterSyntax {
    case legacy, expert
    init?(help:String) {
        guard help.contains("convert exl3") else { return nil }
        if help.contains("--expert-codebook") { self = .expert }
        else if help.contains("--codebook") { self = .legacy }
        else { return nil }
    }
}


struct DownloadPack:Identifiable {
    let id:String
    let size:String
    let downloaded:Bool
    let recommended:Bool
    let published:Bool
    let verdict:String
    static func parse(_ output:String) -> [DownloadPack] {
        guard let pattern = try? NSRegularExpression(pattern:#"^([a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+)\s+(\?|[0-9]+(?:\.[0-9]+)? [KMGT]B)\s+(yes|no)\s+(.+)$"#) else { return [] }
        var seen = Set<String>()
        return output.components(separatedBy:.newlines).compactMap { line in
            let text = line as NSString
            guard let match = pattern.firstMatch(in:line,range:NSRange(location:0,length:text.length)) else { return nil }
            let id = text.substring(with:match.range(at:1))
            guard seen.insert(id).inserted else { return nil }
            let verdict = text.substring(with:match.range(at:4))
            return DownloadPack(id:id,size:text.substring(with:match.range(at:2)),downloaded:text.substring(with:match.range(at:3)) == "yes",recommended:verdict.contains("<- recommended"),published:!verdict.contains("not yet published"),verdict:verdict.replacingOccurrences(of:"<- recommended",with:"").replacingOccurrences(of:"(not yet published)",with:"").trimmingCharacters(in:.whitespaces))
        }
    }
}

struct DownloadProgress {
    let index:Int
    let total:Int
    let file:String
    let fraction:Double?
    static func parse(_ output:String) -> DownloadProgress? {
        let text = output as NSString
        guard let header = try? NSRegularExpression(pattern:#"\[(\d+)/(\d+)\] pulling (.+) \(\d+ MB\)"#),
              let match = header.matches(in:output,range:NSRange(location:0,length:text.length)).last,
              let index = Int(text.substring(with:match.range(at:1))),
              let total = Int(text.substring(with:match.range(at:2))), index > 0, total >= index else { return nil }
        let suffix = text.substring(from:NSMaxRange(match.range)) as NSString
        let percent = try? NSRegularExpression(pattern:#"(?:^|[^0-9.])([0-9]+(?:\.[0-9]+)?)%"#)
        let last = percent?.matches(in:suffix as String,range:NSRange(location:0,length:suffix.length)).last
        let value = last.flatMap { Double(suffix.substring(with:$0.range(at:1))) }
        return DownloadProgress(index:index,total:total,file:text.substring(with:match.range(at:3)),fraction:value.flatMap { (0...100).contains($0) ? $0/100 : nil })
    }
}

struct ToolRecord:Identifiable, Codable {
    var id = UUID()
    let title:String
    let status:String
    let started:Date
    let finished:Date
    let logURL:URL
    let downloadModel:String?
}

struct QueuedDownload:Identifiable, Codable {
    var id = UUID()
    let binary:String
    let model:String
    let timeout:TimeInterval
}

private struct ToolArchive:Codable {
    let version:Int
    let queue:[QueuedDownload]
    let history:[ToolRecord]
    let active:QueuedDownload?
}

@MainActor final class ToolTask: ObservableObject {
    private let storageURL:URL?
    private var recoverableDownload:QueuedDownload?
    @Published private(set) var persistenceError:String?
    @Published private(set) var recoveryNotice:String?
    init(storageURL:URL? = nil) {
        self.storageURL = storageURL
        guard let storageURL, FileManager.default.fileExists(atPath:storageURL.path) else { return }
        do {
            let size = try FileManager.default.attributesOfItem(atPath:storageURL.path)[.size] as? NSNumber
            guard let size, size.intValue <= 1024*1024 else { throw ServerFailure(message:"Task archive exceeds its size limit.") }
            let archive = try JSONDecoder().decode(ToolArchive.self,from:Data(contentsOf:storageURL))
            guard archive.version == 1 else { throw ServerFailure(message:"Task archive version is unsupported.") }
            queue = Array(archive.queue.filter(Self.valid).prefix(20))
            history = Array(archive.history.filter { $0.logURL.isFileURL && $0.title.count <= 4096 }.prefix(20))
            if let active = archive.active, Self.valid(active) {
                queue.removeAll { $0.binary == active.binary && $0.model == active.model }
                queue.insert(active,at:0); queue = Array(queue.prefix(20))
                recoveryNotice = "A download was active when the app closed. Check that its CLI process has stopped before resuming. Restored downloads stay paused."
            }
        } catch { persistenceError = "Could not restore task history: \(error.localizedDescription)" }
    }
    private static func valid(_ item:QueuedDownload) -> Bool {
        item.binary.hasPrefix("/") && !item.binary.contains("\0") && item.binary.count <= 4096 && !item.model.isEmpty && !item.model.hasPrefix("-") && !item.model.contains("\0") && item.model.count <= 4096 && item.timeout.isFinite && item.timeout > 0
    }
    private func saveArchive() {
        guard let storageURL else { return }
        do {
            try FileManager.default.createDirectory(at:storageURL.deletingLastPathComponent(),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
            let data = try JSONEncoder().encode(ToolArchive(version:1,queue:queue,history:history,active:recoverableDownload))
            try data.write(to:storageURL,options:.atomic)
            try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:storageURL.path)
            persistenceError = nil
        } catch { persistenceError = "Could not save task history: \(error.localizedDescription)" }
    }
    @Published private(set) var queue:[QueuedDownload] = [] { didSet { saveArchive() } }
    @Published private(set) var queuePaused = true
    @Published private(set) var queueError:String?
    private var runningQueuedDownload = false
    func enqueueDownload(binary:String,model:String,timeout:TimeInterval) throws {
        let name = model.trimmingCharacters(in:.whitespacesAndNewlines)
        guard Self.valid(QueuedDownload(binary:binary,model:name,timeout:timeout)), FileManager.default.isExecutableFile(atPath:binary) else {
            throw ServerFailure(message:"Choose an installed CLI, a model name and a finite time limit.")
        }
        guard queue.count < 20 else { throw ServerFailure(message:"The queue holds up to 20 downloads.") }
        guard !queue.contains(where: { $0.binary == binary && $0.model == name }), !(running && downloadModel == name) else {
            throw ServerFailure(message:"This model is already downloading or queued.")
        }
        queue.append(QueuedDownload(binary:binary,model:name,timeout:timeout))
    }
    func removeQueued(_ id:UUID) { queue.removeAll { $0.id == id } }
    func moveQueued(_ id:UUID,offset:Int) {
        guard let index = queue.firstIndex(where:{$0.id == id}), queue.indices.contains(index+offset) else { return }
        queue.swapAt(index,index+offset)
    }
    func pauseQueue() { queuePaused = true }
    func clearQueue() { queue.removeAll(); queueError = nil; recoveryNotice = nil; queuePaused = true }
    func startQueue() throws {
        guard !running else { throw ServerFailure(message:"Wait for the current task to finish before starting the queue.") }
        queuePaused = false; queueError = nil; recoveryNotice = nil
        try runNextDownload()
    }
    private func runNextDownload() throws {
        guard !queuePaused else { return }
        guard let next = queue.first else { queuePaused = true; return }
        do {
            try run(binary:next.binary,arguments:["pull",next.model],title:"Download \(next.model)",timeout:next.timeout)
            runningQueuedDownload = true; queue.removeFirst()
        } catch {
            queuePaused = true; queueError = error.localizedDescription; throw error
        }
    }
    @Published private(set) var converterBinary = ""
    @Published private(set) var converterSyntax:ConverterSyntax?
    @Published private(set) var converterAvailable:Bool?
    @Published private(set) var catalog:[DownloadPack] = []
    @Published private(set) var catalogBinary = ""
    @Published private(set) var history:[ToolRecord] = [] { didSet { saveArchive() } }
    @Published private(set) var downloadProgress:DownloadProgress?
    private var downloadModel:String?
    func clearHistory() { history.removeAll() }
    @Published var running = false
    @Published var converting = false
    var exclusiveServerUse:Bool { modifyingRuntime || converting }
    @Published var modifyingRuntime = false
    @Published var title = ""
    @Published var output = ""
    @Published var status = ""
    @Published var started:Date?
    private var pid:pid_t = 0
    private var cancellation:Date?
    private var operation:Task<Void,Never>?
    @Published private(set) var logURL:URL?
    func run(binary:String,arguments:[String],title:String,timeout:TimeInterval,environment:[String:String]? = nil) throws {
        guard !running else { throw ServerFailure(message:"Wait for the current task to finish, or cancel it first.") }
        guard binary.hasPrefix("/"), FileManager.default.isExecutableFile(atPath:binary), timeout.isFinite, timeout > 0 else { throw ServerFailure(message:"Choose an installed Sushi executable in Server settings.") }
        let directory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/Sushi Control/Tasks")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let file = directory.appendingPathComponent(UUID().uuidString + ".log")
        let fd = Darwin.open(file.path,O_CREAT|O_EXCL|O_WRONLY,0o600)
        guard fd >= 0 else { throw ServerFailure(message:"Could not create a task log.") }
        defer { Darwin.close(fd) }
        var actions:posix_spawn_file_actions_t? = nil
        var attributes:posix_spawnattr_t? = nil
        posix_spawn_file_actions_init(&actions); posix_spawnattr_init(&attributes)
        defer { posix_spawn_file_actions_destroy(&actions); posix_spawnattr_destroy(&attributes) }
        posix_spawn_file_actions_adddup2(&actions,fd,STDOUT_FILENO)
        posix_spawn_file_actions_adddup2(&actions,fd,STDERR_FILENO)
        posix_spawn_file_actions_addopen(&actions,STDIN_FILENO,"/dev/null",O_RDONLY,0)
        posix_spawnattr_setflags(&attributes,Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT))
        posix_spawnattr_setpgroup(&attributes,0)
        let argv = ([binary]+arguments).map { strdup($0) } + [nil]
        let env = (environment ?? ProcessInfo.processInfo.environment).map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { argv.forEach { free($0) }; env.forEach { free($0) } }
        var child:pid_t = 0
        let code = argv.withUnsafeBufferPointer { arguments in env.withUnsafeBufferPointer { environment in
            posix_spawn(&child,binary,&actions,&attributes,arguments.baseAddress!,environment.baseAddress!)
        } }
        guard code == 0 else { throw ServerFailure(message:"Could not start Sushi: \(String(cString:strerror(code))).") }
        downloadModel = arguments.count == 2 && arguments.first == "pull" ? arguments[1] : nil
        recoverableDownload = downloadModel.map { QueuedDownload(binary:binary,model:$0,timeout:timeout) }
        saveArchive()
        downloadProgress = nil
        converting = arguments.first == "convert" && !arguments.contains("--help") && !arguments.contains("-h")
        modifyingRuntime = arguments.first == "update" && !arguments.contains("--check")
        if modifyingRuntime { converterBinary = ""; converterAvailable = nil; converterSyntax = nil }
        pid = child; self.title = title; running = true; output = ""; status = "Running"; started = Date(); cancellation = nil; logURL = file
        operation = Task {
            let deadline = Date().addingTimeInterval(timeout)
            var timedOut = false
            var outputExceeded = false
            while true {
                var value:Int32 = 0
                let result = waitpid(child,&value,WNOHANG)
                readOutput()
                if !outputExceeded, let size = try? FileManager.default.attributesOfItem(atPath:file.path)[.size] as? NSNumber, size.uint64Value > 16*1024*1024 {
                    outputExceeded = true; cancel(); kill(-child,SIGKILL)
                }
                if result == child {
                    kill(-child,SIGKILL)
                    if outputExceeded, let log = try? FileHandle(forWritingTo:file) {
                        try? log.truncate(atOffset:16*1024*1024); try? log.close(); readOutput()
                    }
                    let exitCode = (value >> 8) & 0xff
                    let signal = value & 0x7f
                    status = outputExceeded ? "Output limit exceeded" : timedOut ? "Timed out" : cancellation != nil ? "Cancelled" : signal != 0 ? "Stopped (signal \(signal))" : exitCode == 0 ? "Completed" : "Failed (exit \(exitCode))"
                    break
                }
                if result == -1 {
                    if errno == EINTR { continue }
                    kill(-child,SIGKILL)
                    status = "Could not read task status"; break
                }
                if Date() >= deadline && cancellation == nil { timedOut = true; cancel() }
                if let cancellation, Date().timeIntervalSince(cancellation) > 3 { kill(-child,SIGKILL) }
                try? await Task.sleep(nanoseconds:200_000_000)
            }
            if arguments == ["convert","--help"] {
                converterBinary = binary; converterSyntax = status == "Completed" ? ConverterSyntax(help:output) : nil; converterAvailable = converterSyntax != nil
            }
            if arguments == ["list"], status == "Completed" {
                catalog = DownloadPack.parse(output); catalogBinary = binary
            }
            recoverableDownload = nil
            history.insert(ToolRecord(title:self.title,status:status,started:started ?? Date(),finished:Date(),logURL:file,downloadModel:downloadModel),at:0)
            history = Array(history.prefix(20))
            downloadProgress = nil
            let continueQueue = runningQueuedDownload && status == "Completed"
            if runningQueuedDownload && status != "Completed" { queuePaused = true; queueError = "Queue paused after \(status.lowercased()). Resume the interrupted download from Recent tasks, or start the remaining queue." }
            runningQueuedDownload = false
            pid = 0; running = false; converting = false; modifyingRuntime = false; operation = nil
            if continueQueue { try? runNextDownload() }
        }
    }
    func cancel() {
        guard running && pid > 0 && cancellation == nil else { return }
        cancellation = Date(); status = "Cancelling…"; kill(-pid,SIGTERM)
    }
    private func readOutput() {
        guard let logURL, let handle = try? FileHandle(forReadingFrom:logURL) else { return }
        defer { try? handle.close() }
        let length = (try? handle.seekToEnd()) ?? 0
        try? handle.seek(toOffset:length > 64000 ? length-64000 : 0)
        output = String(decoding:(try? handle.readToEnd()) ?? Data(),as:UTF8.self).replacingOccurrences(of:"\r",with:"\n")
        if downloadModel != nil, let progress = DownloadProgress.parse(output) { downloadProgress = progress }
    }
}
