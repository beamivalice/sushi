import Foundation
import Darwin

struct CommandResult { var status:Int32; var text:String }
struct ServiceSnapshot {
    var known = false
    var registered = false
    var running = false
    var exitCode:Int?
    init() {}
    init(result:CommandResult) {
        known = result.status == 0 || (result.status == 113 && result.text.contains("Could not find service"))
        registered = result.status == 0
        guard registered else { return }
        let lines = result.text.components(separatedBy:"\n").map { $0.trimmingCharacters(in:.whitespaces) }
        running = lines.contains("state = running")
        exitCode = lines.first(where:{$0.hasPrefix("last exit code = ")}).flatMap { Int($0.dropFirst("last exit code = ".count)) }
    }
}
struct ManagedServer {
    var label = "org.sushi.SushiControl.server"
    var directory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Sushi Control")
    var externalPlist:URL? = nil
    var loginDirectory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/LaunchAgents")
    var loginPlistURL:URL { loginDirectory.appendingPathComponent(label+".plist") }
    var domain:String { "gui/\(getuid())" }
    var service:String { "\(domain)/\(label)" }
    var plistURL:URL { externalPlist ?? directory.appendingPathComponent("Server.plist") }
    var logURL:URL {
        if let externalPlist, let data = try? Data(contentsOf:externalPlist), let object = try? PropertyListSerialization.propertyList(from:data,format:nil) as? [String:Any], let path = object["StandardOutPath"] as? String, path.hasPrefix("/") { return URL(fileURLWithPath:path) }
        return directory.appendingPathComponent("server.log")
    }
    var errorLogURL:URL {
        if let externalPlist, let data = try? Data(contentsOf:externalPlist), let object = try? PropertyListSerialization.propertyList(from:data,format:nil) as? [String:Any], let path = object["StandardErrorPath"] as? String, path.hasPrefix("/") { return URL(fileURLWithPath:path) }
        return logURL
    }
    static func existingAgent(_ url:URL) throws -> ManagedServer {
        let object = try PropertyListSerialization.propertyList(from:Data(contentsOf:url),format:nil) as? [String:Any] ?? [:]
        guard let label = object["Label"] as? String, !label.isEmpty, label.count <= 255,
              label.unicodeScalars.allSatisfy({ CharacterSet.alphanumerics.union(CharacterSet(charactersIn:".-_")).contains($0) }),
              let args = object["ProgramArguments"] as? [String], let binary = args.first,
              binary.hasPrefix("/"), FileManager.default.isExecutableFile(atPath:binary) else {
            throw ServerFailure(message:"Choose a launch agent with a valid Label and executable ProgramArguments.")
        }
        return ManagedServer(label:label,externalPlist:url)
    }
    func command(_ arguments:[String],timeout:TimeInterval = 15,executable:String = "/bin/launchctl") async throws -> CommandResult {
        try await Task.detached {
            let file = FileManager.default.temporaryDirectory.appendingPathComponent("sushi-management-"+UUID().uuidString)
            guard FileManager.default.createFile(atPath:file.path,contents:nil,attributes:[.posixPermissions:0o600]) else { throw ServerFailure(message:"Could not capture server management output.") }
            defer { try? FileManager.default.removeItem(at:file) }
            let output = try FileHandle(forWritingTo:file)
            defer { try? output.close() }
            let process = Process()
            process.executableURL = URL(fileURLWithPath:executable); process.arguments = arguments
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = output; process.standardError = output
            try process.run()
            let deadline = Date().addingTimeInterval(timeout)
            while process.isRunning && Date() < deadline { try await Task.sleep(nanoseconds:50_000_000) }
            if process.isRunning {
                process.terminate()
                let grace = Date().addingTimeInterval(1)
                while process.isRunning && Date() < grace { try await Task.sleep(nanoseconds:20_000_000) }
                if process.isRunning { Darwin.kill(process.processIdentifier,SIGKILL) }
                let reapDeadline = Date().addingTimeInterval(2)
                while process.isRunning && Date() < reapDeadline { try await Task.sleep(nanoseconds:20_000_000) }
                throw ServerFailure(message:"Server management timed out. Check Diagnostics before trying again.")
            }
            let reader = try FileHandle(forReadingFrom:file)
            defer { try? reader.close() }
            let length = try reader.seekToEnd()
            try reader.seek(toOffset:length > 64000 ? length-64000 : 0)
            let data = try reader.readToEnd() ?? Data()
            return CommandResult(status:process.terminationStatus,text:String(decoding:data,as:UTF8.self))
        }.value
    }
    func checkedSnapshot() async throws -> ServiceSnapshot {
        let result = try await command(["print",service])
        let state = ServiceSnapshot(result:result)
        guard state.known else { throw ServerFailure(message:"Cannot determine the app-managed server’s state. " + String(result.text.suffix(1000))) }
        return state
    }
    func snapshot() async -> ServiceSnapshot {
        (try? await checkedSnapshot()) ?? ServiceSnapshot()
    }
    func registered() async -> Bool { await snapshot().registered }
    func install(binary:String,arguments:[String],startAtLogin:Bool = false,recoverAfterCrash:Bool = false) throws {
        guard externalPlist == nil else { throw ServerFailure(message:"An existing launch agent’s configuration cannot be overwritten by this app.") }
        guard binary.hasPrefix("/"), FileManager.default.isExecutableFile(atPath:binary) else { throw ServerFailure(message:"Choose an installed Sushi executable.") }
        let files = FileManager.default
        let hasLoginItem = (try? files.attributesOfItem(atPath:loginPlistURL.path)) != nil
        if hasLoginItem {
            guard (try? files.destinationOfSymbolicLink(atPath:loginPlistURL.path)) == plistURL.path else { throw ServerFailure(message:"The login launcher belongs to another configuration. It was left unchanged.") }
        }
        try files.createDirectory(at:directory,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        var object:[String:Any] = ["Label":label,"ProgramArguments":[binary]+arguments,"RunAtLoad":true,"KeepAlive":false,"ProcessType":"Interactive","StandardOutPath":logURL.path,"StandardErrorPath":logURL.path,"ExitTimeOut":30]
        if recoverAfterCrash { object["KeepAlive"] = ["SuccessfulExit":false]; object["ThrottleInterval"] = 30 }
        let data = try PropertyListSerialization.data(fromPropertyList:object,format:.xml,options:0)
        let previous = files.fileExists(atPath:plistURL.path) ? try Data(contentsOf:plistURL) : nil
        do {
            try data.write(to:plistURL,options:.atomic)
            try files.setAttributes([.posixPermissions:0o600],ofItemAtPath:plistURL.path)
            if startAtLogin && !hasLoginItem {
                try files.createDirectory(at:loginDirectory,withIntermediateDirectories:true)
                try files.createSymbolicLink(at:loginPlistURL,withDestinationURL:plistURL)
            } else if !startAtLogin && hasLoginItem { try files.removeItem(at:loginPlistURL) }
        } catch {
            if let previous { try previous.write(to:plistURL,options:.atomic); try files.setAttributes([.posixPermissions:0o600],ofItemAtPath:plistURL.path) }
            else { try? files.removeItem(at:plistURL) }
            throw error
        }
    }
    func validateExistingAgent() throws {
        if let externalPlist {
            let current = try Self.existingAgent(externalPlist)
            guard current.label == label else { throw ServerFailure(message:"The selected launch agent’s Label changed. Select it again before controlling it.") }
        }
    }
    func start() async throws {
        try validateExistingAgent()
        if try await checkedSnapshot().registered {
            let result = try await command(["kickstart",service])
            guard result.status == 0 else { throw ServerFailure(message:result.text) }
            return
        }
        for attempt in 0..<20 {
            let result = try await command(["bootstrap",domain,plistURL.path])
            if result.status == 0 { return }
            guard result.status == 5, attempt < 19 else { throw ServerFailure(message:result.text) }
            try await Task.sleep(nanoseconds:250_000_000)
        }
    }
    func stop() async throws {
        let result = try await command(["bootout",service],timeout:35)
        guard result.status == 0 else { throw ServerFailure(message:result.text) }
        let deadline = Date().addingTimeInterval(35)
        while try await checkedSnapshot().registered {
            guard Date() < deadline else { throw ServerFailure(message:"The server is still stopping. Check its log before restarting.") }
            try await Task.sleep(nanoseconds:100_000_000)
        }
    }
}
