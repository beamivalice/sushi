import Foundation

@main struct ManagedServerTests {
    static func main() async throws {
        precondition(!ServiceSnapshot().known)
        let absent = ServiceSnapshot(result:CommandResult(status:113,text:"Could not find service in domain"))
        precondition(absent.known && !absent.registered)
        let denied = ServiceSnapshot(result:CommandResult(status:1,text:"Operation not permitted"))
        precondition(!denied.known && !denied.registered)
        let active = ServiceSnapshot(result:CommandResult(status:0,text:"state = running\nlast exit code = 0"))
        precondition(active.known && active.registered && active.running)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:directory) }
        let loginDirectory = directory.appendingPathComponent("LoginAgents")
        let startupServer = ManagedServer(label:"org.sushi.startup.test."+UUID().uuidString.lowercased(),directory:directory.appendingPathComponent("startup"),loginDirectory:loginDirectory)
        try startupServer.install(binary:"/bin/sleep",arguments:["60"],startAtLogin:true,recoverAfterCrash:true)
        let startupPlist = try PropertyListSerialization.propertyList(from:Data(contentsOf:startupServer.plistURL),format:nil) as! [String:Any]
        precondition((startupPlist["KeepAlive"] as? [String:Bool])?["SuccessfulExit"] == false)
        precondition(startupPlist["ThrottleInterval"] as? Int == 30)
        let loginTarget = try FileManager.default.destinationOfSymbolicLink(atPath:startupServer.loginPlistURL.path); precondition(loginTarget == startupServer.plistURL.path)
        try startupServer.install(binary:"/bin/sleep",arguments:["60"],startAtLogin:false,recoverAfterCrash:false)
        precondition(!FileManager.default.fileExists(atPath:startupServer.loginPlistURL.path))
        try FileManager.default.createDirectory(at:loginDirectory,withIntermediateDirectories:true)
        try Data("foreign launcher".utf8).write(to:startupServer.loginPlistURL)
        do { try startupServer.install(binary:"/bin/sleep",arguments:["60"],startAtLogin:true); fatalError("Replaced a foreign login agent") }
        catch let error as ServerFailure { precondition(error.message.contains("another")) }
        let foreignData = try Data(contentsOf:startupServer.loginPlistURL); precondition(foreignData == Data("foreign launcher".utf8))
        try FileManager.default.removeItem(at:startupServer.loginPlistURL)
        let originalConfig = try Data(contentsOf:startupServer.plistURL)
        var failedLoginServer = startupServer
        failedLoginServer.loginDirectory = directory.appendingPathComponent("not-a-directory")
        try Data("file".utf8).write(to:failedLoginServer.loginDirectory)
        do { try failedLoginServer.install(binary:"/bin/sleep",arguments:["30"],startAtLogin:true); fatalError("Accepted invalid login directory") } catch {}
        let rolledBackConfig = try Data(contentsOf:startupServer.plistURL); precondition(originalConfig == rolledBackConfig)
        if CommandLine.arguments.contains("--policy-only") { print("Startup policy passed: opt-in link, crash policy, removal and foreign-file preservation"); return }
        let crashMarker = directory.appendingPathComponent("crashed"), recoveredMarker = directory.appendingPathComponent("recovered")
        let recoveryScript = "if [ ! -e \"$1\" ]; then : > \"$1\"; exit 7; fi; : > \"$2\"; exec /bin/sleep 60"
        try startupServer.install(binary:"/bin/sh",arguments:["-c",recoveryScript,"recovery-fixture",crashMarker.path,recoveredMarker.path],startAtLogin:true,recoverAfterCrash:true)
        do {
            let loginLoad = try await startupServer.command(["bootstrap",startupServer.domain,startupServer.loginPlistURL.path])
            guard loginLoad.status == 0 else { throw ServerFailure(message:loginLoad.text) }
            let deadline = Date().addingTimeInterval(45)
            while true {
                let state = try await startupServer.checkedSnapshot()
                if FileManager.default.fileExists(atPath:recoveredMarker.path) && state.running { break }
                guard Date() < deadline else { throw ServerFailure(message:"Crash recovery did not restart the test job") }
                try await Task.sleep(nanoseconds:200_000_000)
            }
            try await startupServer.stop()
            let stoppedStartup = await startupServer.snapshot(); precondition(!stoppedStartup.registered)
        } catch { if await startupServer.registered() { try? await startupServer.stop() }; throw error }
        let server = ManagedServer(label:"org.sushi.control.test.\(UUID().uuidString.lowercased())",directory:directory)
        let pidFile = directory.appendingPathComponent("timeout.pid")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let script = "import os,signal,time; open("+String(reflecting:pidFile.path)+",'w').write(str(os.getpid())); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
        let started = Date()
        do { _ = try await server.command(["-c",script],timeout:0.5,executable:"/usr/bin/python3"); throw ServerFailure(message:"Timeout was not enforced") }
        catch { guard error.localizedDescription.contains("timed out") else { throw error } }
        precondition(Date().timeIntervalSince(started) < 6)
        let child = Int32(try String(contentsOf:pidFile,encoding:.utf8))!
        precondition(kill(child,0) == -1 && errno == ESRCH)
        let binary = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/bin/sleep"
        var args = ["60"]
        var client:Connection? = nil
        if binary != "/bin/sleep" {
            let port = CommandLine.arguments[2]
            let models = directory.appendingPathComponent("models")
            try FileManager.default.createDirectory(at:models,withIntermediateDirectories:true)
            args = ["serve","--host","127.0.0.1","--port",port,"--model-dir",models.path,"--metrics"]
            client = Connection(endpoint:"http://127.0.0.1:\(port)")
        }
        try server.install(binary:binary,arguments:args)
        let bytes = try Data(contentsOf:server.plistURL)
        let plist = try PropertyListSerialization.propertyList(from:bytes,format:nil) as! [String:Any]
        precondition(plist["ProgramArguments"] as? [String] == [binary]+args)
        precondition(plist["KeepAlive"] as? Bool == false)
        let attributes = try FileManager.default.attributesOfItem(atPath:server.plistURL.path)
        precondition(attributes[.posixPermissions] as? Int == 0o600)
        let before = await server.registered()
        precondition(!before)
        do {
            try await server.start()
            guard await server.registered() else { throw ServerFailure(message:"Test service did not register") }
            if let client { try await ready(client) }
            let reopened = ManagedServer(label:server.label,directory:directory)
            guard await reopened.registered() else { throw ServerFailure(message:"Reopened controller lost service") }
            try await reopened.stop()
            guard !(await server.registered()) else { throw ServerFailure(message:"Test service did not stop") }
            try await reopened.start()
            if let client { try await ready(client) }
            try await reopened.stop()
        } catch {
            if await server.registered() { try? await server.stop() }
            throw error
        }
        let external = try ManagedServer.existingAgent(server.plistURL)
        let preserved = try Data(contentsOf:server.plistURL)
        do {
            do { try external.install(binary:"/bin/sleep",arguments:["10"]); throw ServerFailure(message:"External configuration overwrite was accepted") }
            catch { guard error.localizedDescription.contains("cannot be overwritten") else { throw error } }
            try await external.start()
            let reopenedExternal = try ManagedServer.existingAgent(server.plistURL)
            guard await reopenedExternal.registered() else { throw ServerFailure(message:"External service was not registered") }
            try await reopenedExternal.stop()
            try await reopenedExternal.start()
            try await reopenedExternal.stop()
            let afterExternal = try Data(contentsOf:server.plistURL)
            precondition(preserved == afterExternal)
        } catch {
            if await external.registered() { try? await external.stop() }
            throw error
        }
        try server.install(binary:"/usr/bin/false",arguments:[])
        try await server.start()
        let failedDeadline = Date().addingTimeInterval(5)
        var failed = await server.snapshot()
        while failed.exitCode == nil && Date() < failedDeadline { try await Task.sleep(nanoseconds:100_000_000); failed = await server.snapshot() }
        precondition(failed.registered && !failed.running && failed.exitCode == 1)
        try await server.stop()
        print("Managed service lifecycle passed: start, reopen, stop, restart, cleanup")
    }
    static func ready(_ client:Connection) async throws {
        let deadline = Date().addingTimeInterval(25)
        while Date() < deadline {
            if let models = try? await client.fetch("v1/models") {
                precondition(ModelEntry.parse(models).isEmpty)
                return
            }
            try await Task.sleep(nanoseconds:200_000_000)
        }
        throw ServerFailure(message:"Managed stock server did not become ready")
    }
}
