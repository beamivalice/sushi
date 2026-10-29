import Foundation

struct LocalFileIdentity:Equatable {
    let device:UInt64
    let inode:UInt64
    init(_ url:URL) throws {
        let attributes = try FileManager.default.attributesOfItem(atPath:url.path)
        guard let device = attributes[.systemNumber] as? NSNumber, let inode = attributes[.systemFileNumber] as? NSNumber else { throw ServerFailure(message:"Cannot identify the selected file safely.") }
        self.device = device.uint64Value; self.inode = inode.uint64Value
    }
}
struct LocalModelFile {
    let url:URL
    let architecture:String
    let bytes:Int64
    let fileCount:Int
    let partial:Bool
    private let identity:LocalFileIdentity
    static func inspect(_ input:URL) throws -> LocalModelFile {
        guard input.isFileURL, input.path.hasPrefix("/") else { throw ServerFailure(message:"Choose a local model folder or GGUF file.") }
        let values = try input.resourceValues(forKeys:[.isSymbolicLinkKey,.isDirectoryKey,.isRegularFileKey,.fileSizeKey])
        guard values.isSymbolicLink != true else { throw ServerFailure(message:"Choose the actual model folder or file rather than a symbolic link.") }
        let url = input.standardizedFileURL.resolvingSymlinksInPath()
        let home = FileManager.default.homeDirectoryForCurrentUser
        let protected = [URL(fileURLWithPath:"/"),home]+["Library","Documents","Desktop","Downloads","Applications",".sushi",".sushi/models",".cache/huggingface/hub"].map { home.appendingPathComponent($0) }
        guard !protected.contains(where:{$0.resolvingSymlinksInPath().standardizedFileURL.path == url.path}) else { throw ServerFailure(message:"Select an individual model, not a system, home or model-library folder.") }
        let identity = try LocalFileIdentity(url)
        if values.isRegularFile == true {
            guard url.pathExtension.lowercased() == "gguf" else { throw ServerFailure(message:"Choose a GGUF file or a model folder containing config.json and weights.") }
            let file = try FileHandle(forReadingFrom:url); defer { try? file.close() }
            guard try file.read(upToCount:4) == Data("GGUF".utf8) else { throw ServerFailure(message:"The selected file does not have a GGUF header.") }
            return LocalModelFile(url:url,architecture:"GGUF",bytes:Int64(values.fileSize ?? 0),fileCount:1,partial:false,identity:identity)
        }
        guard values.isDirectory == true else { throw ServerFailure(message:"Choose a regular model file or folder.") }
        let config = url.appendingPathComponent("config.json")
        let size = try config.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? Int.max
        guard size <= 1024*1024,
              let object = try JSONSerialization.jsonObject(with:Data(contentsOf:config)) as? [String:Any],
              let architecture = object["model_type"] as? String, !architecture.isEmpty,
              try FileManager.default.contentsOfDirectory(at:url,includingPropertiesForKeys:nil).contains(where:{["safetensors","gguf"].contains($0.pathExtension.lowercased())}) else {
            throw ServerFailure(message:"The selected folder needs config.json with a model type and model weights.")
        }
        var bytes:Int64 = 0, count = 0, partial = false
        let deadline = Date().addingTimeInterval(2)
        let keys:Set<URLResourceKey> = [.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey]
        guard let iterator = FileManager.default.enumerator(at:url,includingPropertiesForKeys:Array(keys),options:[],errorHandler:{ _,_ in partial = true; return true }) else { throw ServerFailure(message:"Cannot inspect this model folder.") }
        for case let file as URL in iterator {
            if count >= 20000 || Date() > deadline { partial = true; break }
            guard let value = try? file.resourceValues(forKeys:keys) else { partial = true; continue }
            if value.isSymbolicLink == true { iterator.skipDescendants(); partial = true; continue }
            if value.isRegularFile == true { count += 1; bytes += Int64(value.fileSize ?? 0) }
        }
        return LocalModelFile(url:url,architecture:architecture,bytes:bytes,fileCount:count,partial:partial,identity:identity)
    }
    func verifyIdentity() throws {
        guard try LocalFileIdentity(url) == identity else { throw ServerFailure(message:"The selected model path changed. Choose it again before moving files.") }
    }
    func moveToTrash() throws -> ModelTrashReceipt {
        try verifyIdentity()
        var destination:NSURL?
        try FileManager.default.trashItem(at:url,resultingItemURL:&destination)
        guard let trashedURL = destination as URL? else { throw ServerFailure(message:"The model moved to Trash, but its restore location was not returned. Restore it in Finder.") }
        return ModelTrashReceipt(originalURL:url,trashedURL:trashedURL,identity:try LocalFileIdentity(trashedURL))
    }
}
struct ModelTrashReceipt {
    let originalURL:URL
    let trashedURL:URL
    private let identity:LocalFileIdentity
    fileprivate init(originalURL:URL,trashedURL:URL,identity:LocalFileIdentity) { self.originalURL = originalURL; self.trashedURL = trashedURL; self.identity = identity }
    func restore() throws {
        guard !FileManager.default.fileExists(atPath:originalURL.path) else { throw ServerFailure(message:"The original model path is occupied. Restore the trashed item using Finder.") }
        guard try LocalFileIdentity(trashedURL) == identity else { throw ServerFailure(message:"The trashed item changed. Restore it using Finder.") }
        try FileManager.default.moveItem(at:trashedURL,to:originalURL)
    }
}
