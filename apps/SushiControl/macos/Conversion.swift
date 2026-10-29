import Foundation

struct ConversionPlan:Codable,Equatable {
    var source = ""
    var output = ""
    var expertBits = 2.5
    var calibrationTokens = 65536
    var batch = 16
    var resume = false
    var codebook = "mcg"
    var window = 15
    var trunkBits = 0
    var trunkGroup = 0
    var calibrationFile = ""
    var factorPrecision = 64
    private static func within(_ path:String,root:String) -> Bool {
        func identity(_ path:String) -> String? {
            guard let values = try? FileManager.default.attributesOfItem(atPath:URL(fileURLWithPath:path).resolvingSymlinksInPath().path), let device = values[.systemNumber] as? NSNumber, let inode = values[.systemFileNumber] as? NSNumber else { return nil }
            return device.stringValue+":"+inode.stringValue
        }
        guard let rootID = identity(root) else { return false }
        var cursor = URL(fileURLWithPath:path)
        while true {
            if identity(cursor.path) == rootID { return true }
            if cursor.path == "/" { return false }
            cursor.deleteLastPathComponent()
        }
    }
    func arguments(syntax:ConverterSyntax = .legacy) throws -> [String] {
        guard source.hasPrefix("/"), output.hasPrefix("/"), source.utf8.count <= 4096, output.utf8.count <= 4096, !source.contains("\0"), !output.contains("\0") else { throw ServerFailure(message:"Choose absolute source and output paths.") }
        let input = URL(fileURLWithPath:source).resolvingSymlinksInPath().standardizedFileURL.path
        let destination = URL(fileURLWithPath:output).resolvingSymlinksInPath().standardizedFileURL.path
        guard input != "/", destination != "/", input != destination, !destination.hasPrefix(input+"/"), !input.hasPrefix(destination+"/") else { throw ServerFailure(message:"Source and output must be separate folders, outside one another.") }
        guard FileManager.default.fileExists(atPath:input+"/config.json") else { throw ServerFailure(message:"Choose a source checkpoint folder containing config.json.") }
        let sourceInfo = try ConversionSourceInfo.read(input)
        guard !sourceInfo.isSushiPack else { throw ServerFailure(message:"This is already a Sushi pack. Choose the original source checkpoint instead.") }
        if Self.within(destination,root:input) || Self.within(input,root:destination) { throw ServerFailure(message:"Source and output resolve to overlapping folders.") }
        if FileManager.default.fileExists(atPath:destination) {
            let directory = (try? URL(fileURLWithPath:destination).resourceValues(forKeys:[.isDirectoryKey]).isDirectory) == true
            guard directory else { throw ServerFailure(message:"The output path must name a folder.") }
            if !resume, !(try FileManager.default.contentsOfDirectory(atPath:destination)).isEmpty { throw ServerFailure(message:"The output folder contains files. Choose a new folder, or explicitly resume an interrupted conversion.") }
        } else if resume { throw ServerFailure(message:"Choose the existing interrupted output folder to resume.") }
        guard expertBits.isFinite, (1.5...8).contains(expertBits), abs(expertBits*8-(expertBits*8).rounded()) < 0.000001, (2048...Int(UInt32.max)).contains(calibrationTokens), (1...64).contains(batch) else { throw ServerFailure(message:"Use 1.5–8 expert bits in steps of 0.125, at least 2,048 calibration tokens and batch size 1–64.") }
        var result = ["convert","exl3","--model",input,"--out",destination,"--experts-k",String(expertBits),"--calib-tokens",String(calibrationTokens),"--batch",String(batch)]
        guard ["mcg","mul1"].contains(codebook), (12...16).contains(window), [0,2,3,4,5,6,8].contains(trunkBits), [0,32,64,128].contains(trunkGroup), [32,64].contains(factorPrecision) else { throw ServerFailure(message:"Choose supported codebook, search window, trunk and factor settings.") }
        result += [syntax == .expert ? "--expert-codebook" : "--codebook",syntax == .expert ? "exl3-"+codebook : codebook,"--window",String(window),"--factor-precision",String(factorPrecision)]
        if trunkBits > 0 { result += ["--trunk-bits",String(trunkBits)] }
        if trunkGroup > 0 { result += ["--trunk-group",String(trunkGroup)] }
        if !calibrationFile.isEmpty {
            guard calibrationFile.hasPrefix("/"), !calibrationFile.contains("\0"), FileManager.default.isReadableFile(atPath:calibrationFile), (try? URL(fileURLWithPath:calibrationFile).resourceValues(forKeys:[.isRegularFileKey]).isRegularFile) == true else { throw ServerFailure(message:"Choose a readable calibration text file.") }
            result += ["--calib",calibrationFile]
        }
        if resume { result.append("--resume") }
        return result
    }
}

struct ConversionSourceInfo {
    let architecture:String
    let precision:String?
    let isSushiPack:Bool
    static func read(_ path:String) throws -> ConversionSourceInfo {
        guard path.hasPrefix("/"), !path.contains("\0") else { throw ServerFailure(message:"Choose an absolute source checkpoint folder.") }
        let url = URL(fileURLWithPath:path).appendingPathComponent("config.json").resolvingSymlinksInPath()
        let attributes = try url.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey])
        guard attributes.isRegularFile == true, let size = attributes.fileSize, size <= 4*1024*1024 else { throw ServerFailure(message:"The source config.json must be a regular file smaller than 4 MiB.") }
        guard let root = try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any] else { throw ServerFailure(message:"The source config.json must contain a model configuration object.") }
        let text = root["text_config"] as? [String:Any] ?? [:]
        guard let architecture = root["model_type"] as? String ?? text["model_type"] as? String, !architecture.isEmpty, architecture.utf8.count <= 128 else { throw ServerFailure(message:"The source configuration must identify its model type.") }
        let expert = root["expert_quant"] as? [String:Any] ?? text["expert_quant"] as? [String:Any] ?? [:]
        let quant = root["quantization_config"] as? [String:Any] ?? [:]
        let isPack = ["exl3","stq"].contains((expert["format"] as? String)?.lowercased() ?? "") || ["exl3","stq"].contains((quant["quant_method"] as? String)?.lowercased() ?? "")
        let precision = root["dtype"] as? String ?? root["torch_dtype"] as? String ?? text["dtype"] as? String ?? text["torch_dtype"] as? String
        return ConversionSourceInfo(architecture:architecture,precision:precision.flatMap{$0.utf8.count <= 128 ? $0 : nil},isSushiPack:isPack)
    }
}
