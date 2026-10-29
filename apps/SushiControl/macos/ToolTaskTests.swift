import Foundation
import Darwin

@main struct ToolTaskTests {
    @MainActor static func main() async throws {
        let stateDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let stateFile = stateDirectory.appendingPathComponent("tasks.json")
        defer { try? FileManager.default.removeItem(at:stateDirectory) }
        let saved = ToolTask(storageURL:stateFile)
        try saved.enqueueDownload(binary:"/bin/echo",model:"restored",timeout:5)
        let permissions = try FileManager.default.attributesOfItem(atPath:stateFile.path)[.posixPermissions] as! NSNumber
        precondition(permissions.intValue & 0o777 == 0o600)
        var archive = try JSONSerialization.jsonObject(with:Data(contentsOf:stateFile)) as! [String:Any]
        archive["active"] = (archive["queue"] as! [[String:Any]])[0]
        try JSONSerialization.data(withJSONObject:archive).write(to:stateFile)
        let interrupted = ToolTask(storageURL:stateFile)
        precondition(interrupted.queue.count == 1 && interrupted.recoveryNotice != nil && interrupted.queuePaused && !interrupted.running)
        let reopened = ToolTask(storageURL:stateFile)
        precondition(reopened.queue.map(\.model) == ["restored"] && reopened.queuePaused && !reopened.running)
        try reopened.startQueue(); try await finish(reopened)
        let completed = ToolTask(storageURL:stateFile)
        precondition(completed.queue.isEmpty && completed.history.first?.downloadModel == "restored")
        completed.clearHistory()
        precondition(ToolTask(storageURL:stateFile).history.isEmpty)
        try Data("invalid JSON".utf8).write(to:stateFile)
        let corrupt = ToolTask(storageURL:stateFile)
        precondition(corrupt.queue.isEmpty && corrupt.persistenceError != nil)
        try Data(repeating:32,count:1024*1024+1).write(to:stateFile)
        precondition(ToolTask(storageURL:stateFile).persistenceError != nil)
        let queued = ToolTask()
        try queued.enqueueDownload(binary:"/bin/echo",model:"first",timeout:5)
        try queued.enqueueDownload(binary:"/bin/echo",model:"second",timeout:5)
        precondition(!queued.running && queued.queue.count == 2)
        do { try queued.enqueueDownload(binary:"/bin/echo",model:"first",timeout:5); fatalError("Duplicate accepted") } catch {}
        queued.moveQueued(queued.queue[1].id,offset:-1)
        precondition(queued.queue.map(\.model) == ["second","first"])
        queued.moveQueued(queued.queue[0].id,offset:1)
        try queued.startQueue()
        let queueDeadline = Date().addingTimeInterval(5)
        while queued.running || !queued.queue.isEmpty {
            precondition(Date() < queueDeadline)
            try await Task.sleep(nanoseconds:50_000_000)
        }
        precondition(queued.history.map(\.downloadModel) == ["second","first"])
        try queued.enqueueDownload(binary:"/usr/bin/false",model:"failure",timeout:5)
        try queued.enqueueDownload(binary:"/bin/echo",model:"pending",timeout:5)
        try queued.startQueue(); try await finish(queued)
        precondition(queued.queuePaused && queued.queue.count == 1 && queued.queue[0].model == "pending")
        queued.removeQueued(queued.queue[0].id)
        precondition(queued.queue.isEmpty)
        try queued.enqueueDownload(binary:"/bin/sleep",model:"60",timeout:5)
        try queued.enqueueDownload(binary:"/bin/echo",model:"after-cancel",timeout:5)
        try queued.startQueue()
        queued.cancel(); try await finish(queued)
        precondition(queued.queuePaused && queued.queue.count == 1)
        queued.removeQueued(queued.queue[0].id)
        try queued.enqueueDownload(binary:"/bin/echo",model:"pause-first",timeout:5)
        try queued.enqueueDownload(binary:"/bin/echo",model:"pause-next",timeout:5)
        try queued.startQueue(); queued.pauseQueue(); try await finish(queued)
        precondition(queued.queue.count == 1 && queued.queue[0].model == "pause-next")
        queued.clearQueue()
        precondition(queued.queue.isEmpty && queued.queueError == nil && queued.queuePaused)
        for index in 0..<20 { try queued.enqueueDownload(binary:"/bin/echo",model:"bounded-\(index)",timeout:5) }
        do { try queued.enqueueDownload(binary:"/bin/echo",model:"overflow",timeout:5); fatalError("Queue exceeded bound") } catch {}
        precondition(queued.queue.count == 20)
        queued.clearQueue()
        let catalog = DownloadPack.parse("PACK SIZE DOWNLOADED THIS MAC\nqwen3.8-flash-next:2.6bpw  43.2 GB  yes  fits  <- recommended\nglm-5.3-flash:2bpw ? no may stream (--ssd-budget-gb) (not yet published)\nmalformed row\n")
        precondition(catalog.count == 2 && catalog[0].downloaded && catalog[0].recommended && catalog[0].size == "43.2 GB")
        precondition(!catalog[1].published && catalog[1].size == "?")
        precondition(DownloadPack.parse("note: could not reach huggingface.co; showing local models only").isEmpty)
        let progress = DownloadProgress.parse("[1/3] pulling config.json (0 MB)\r### 100.0%\n[2/3] pulling model.safetensors (120 MB)\r#### 42.5%")!
        precondition(progress.file == "model.safetensors" && progress.index == 2 && progress.total == 3 && progress.fraction == 0.425)
        precondition(DownloadProgress.parse("[1/2] pulling config.json (0 MB)\r100.0%\n[2/2] pulling model.safetensors (12 MB)")?.fraction == nil)
        precondition(DownloadProgress.parse("[1/1] pulling weights (1 MB)\r150.0%")?.fraction == nil)
        precondition(DownloadProgress.parse("unknown CLI output") == nil)
        let task = ToolTask()
        try task.run(binary:"/bin/echo",arguments:["model with spaces"],title:"Output",timeout:5)
        try await finish(task)
        precondition(task.output.contains("model with spaces") && task.status == "Completed")
        precondition(task.history.count == 1 && task.history[0].status == "Completed")
        precondition(task.history[0].downloadModel == nil)
        let descriptorFile = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data("private app descriptor".utf8).write(to:descriptorFile)
        let descriptor = Darwin.open(descriptorFile.path,O_RDONLY)
        precondition(descriptor > STDERR_FILENO)
        defer { Darwin.close(descriptor); try? FileManager.default.removeItem(at:descriptorFile) }
        try task.run(binary:"/bin/sh",arguments:["-c","test ! -e /dev/fd/\(descriptor)"],title:"Descriptor isolation",timeout:5)
        try await finish(task)
        precondition(task.status == "Completed", "Child inherited an unrelated app descriptor")
        let childFile = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:childFile) }
        try task.run(binary:"/bin/sh",arguments:["-c","sleep 60 & echo $! > \"$1\"; wait","test",childFile.path],title:"Cancel",timeout:10)
        let deadline = Date().addingTimeInterval(3)
        while !FileManager.default.fileExists(atPath:childFile.path) && Date() < deadline { try await Task.sleep(nanoseconds:50_000_000) }
        let child = pid_t(try String(contentsOf:childFile,encoding:.utf8).trimmingCharacters(in:.whitespacesAndNewlines))!
        task.cancel(); try await finish(task)
        let cleanup = Date().addingTimeInterval(3)
        while kill(child,0) == 0 && Date() < cleanup { try await Task.sleep(nanoseconds:50_000_000) }
        precondition(kill(child,0) == -1 && errno == ESRCH)
        precondition(task.status == "Cancelled")
        try task.run(binary:"/bin/sleep",arguments:["60"],title:"Timeout",timeout:0.2)
        try await finish(task)
        precondition(task.status == "Timed out")
        try task.run(binary:"/bin/sh",arguments:["-c","dd if=/dev/zero bs=1048576 count=17 2>/dev/null"],title:"Output limit",timeout:5)
        try await finish(task)
        precondition(task.status == "Output limit exceeded")
        let capturedLog = task.logURL!
        let capturedSize = try FileManager.default.attributesOfItem(atPath:capturedLog.path)[.size] as! NSNumber
        precondition(capturedSize.uint64Value <= 16*1024*1024 && task.output.utf8.count <= 64000)
        try FileManager.default.removeItem(at:capturedLog)
        do {
            try task.run(binary:"/bin/echo",arguments:[],title:"Invalid deadline",timeout:.infinity)
            fatalError("Unbounded deadline accepted")
        } catch { precondition(!task.running) }
        if CommandLine.arguments.count > 1 {
            try task.run(binary:CommandLine.arguments[1],arguments:["--version"],title:"Stock version",timeout:10)
            try await finish(task)
            precondition(task.status == "Completed" && task.output.contains("sushi "))
            try task.run(binary:CommandLine.arguments[1],arguments:["pull","__invalid_model_for_test__"],title:"Invalid download",timeout:10)
            try await finish(task)
            precondition(task.status.hasPrefix("Failed"))
        }
        for index in 0..<22 {
            try task.run(binary:"/bin/echo",arguments:[String(index)],title:"History \(index)",timeout:5)
            try await finish(task)
        }
        precondition(task.history.count == 20 && task.history.first?.title == "History 21" && task.history.last?.title == "History 2")
        task.clearHistory()
        precondition(task.history.isEmpty)
        print("Tool task tests passed: output, child cleanup, deadline, stock CLI")
    }
    @MainActor static func finish(_ task:ToolTask) async throws {
        let deadline = Date().addingTimeInterval(10)
        while task.running {
            if Date() > deadline { task.cancel(); throw ServerFailure(message:"Task test exceeded deadline") }
            try await Task.sleep(nanoseconds:100_000_000)
        }
    }
}
