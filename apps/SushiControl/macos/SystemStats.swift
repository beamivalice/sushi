import Foundation
import Darwin
import IOKit
import CoreFoundation

struct SystemSnapshot {
    var physicalBytes = Double(ProcessInfo.processInfo.physicalMemory)
    var wiredBytes:Double?
    var compressedBytes:Double?
    var freeBytes:Double?
    var gpuName:String?
    var gpuCoreCount:Int?
    var gpuUsagePercent:Double?
    var cpuUsagePercent:Double?
    var cpuTicks:[UInt32]?
    var uptime = ProcessInfo.processInfo.systemUptime
    var thermal = "Unknown"
    var thermalWarning = false
    static let processorName:String = {
        var length = 0
        guard sysctlbyname("machdep.cpu.brand_string",nil,&length,nil,0) == 0, length > 0, length < 1024 else { return "Apple Silicon" }
        var bytes = [CChar](repeating:0,count:length)
        let result = bytes.withUnsafeMutableBufferPointer { sysctlbyname("machdep.cpu.brand_string",$0.baseAddress,&length,nil,0) }
        guard result == 0 else { return "Apple Silicon" }
        return String(decoding:bytes.prefix { $0 != 0 }.map { UInt8(bitPattern:$0) },as:UTF8.self)
    }()
    static func cpuPercent(previous:[UInt32]?,current:[UInt32]) -> Double? {
        guard let previous, previous.count == 4, current.count == 4 else { return nil }
        let delta = zip(current,previous).map { UInt64($0 &- $1) }
        let total = delta.reduce(0,+)
        guard total > 0 else { return nil }
        return Double(total-delta[2])/Double(total)*100
    }
    static func memoryBytes(_ memory:vm_statistics64_data_t,pageSize:Double) -> (free:Double,wired:Double,compressed:Double) {
        (Double(memory.free_count)*pageSize,Double(memory.wire_count)*pageSize,Double(memory.compressor_page_count)*pageSize)
    }
    static func gpuInfo(_ properties:[String:Any]) -> (name:String?,cores:Int?,percent:Double?) {
        func number(_ value:Any?) -> Double? {
            guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(), value.doubleValue.isFinite else { return nil }
            return value.doubleValue
        }
        let name = (properties["model"] as? String)?.trimmingCharacters(in:.whitespacesAndNewlines)
        let coreNumber = number(properties["gpu-core-count"])
        let cores = coreNumber.flatMap { $0 > 0 && $0 <= 1024 && $0.rounded() == $0 ? Int($0) : nil }
        let performance = properties["PerformanceStatistics"] as? [String:Any]
        let percent = number(performance?["Device Utilization %"]).flatMap { (0...100).contains($0) ? $0 : nil }
        return (name?.isEmpty == false ? name : nil,cores,percent)
    }
    static func capture() -> SystemSnapshot {
        var result = SystemSnapshot()
        let gpu = IOServiceGetMatchingService(kIOMainPortDefault,IOServiceMatching("AGXAccelerator"))
        if gpu != 0 {
            defer { IOObjectRelease(gpu) }
            var properties:Unmanaged<CFMutableDictionary>?
            if IORegistryEntryCreateCFProperties(gpu,&properties,kCFAllocatorDefault,0) == KERN_SUCCESS, let properties {
                let info = gpuInfo(properties.takeRetainedValue() as NSDictionary as? [String:Any] ?? [:])
                result.gpuName = info.name; result.gpuCoreCount = info.cores; result.gpuUsagePercent = info.percent
            }
        }
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: result.thermal = "Nominal"
        case .fair: result.thermal = "Warm"
        case .serious: result.thermal = "Hot"; result.thermalWarning = true
        case .critical: result.thermal = "Critical"; result.thermalWarning = true
        @unknown default: break
        }
        let host = mach_host_self()
        defer { mach_port_deallocate(mach_task_self_,host) }
        var memory = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let memoryStatus = withUnsafeMutablePointer(to:&memory) { pointer in
            pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) { host_statistics64(host,HOST_VM_INFO64,$0,&count) }
        }
        let pageSize = Double(sysconf(_SC_PAGESIZE))
        if memoryStatus == KERN_SUCCESS && pageSize > 0 {
            let bytes = memoryBytes(memory,pageSize:pageSize)
            result.wiredBytes = bytes.wired; result.compressedBytes = bytes.compressed; result.freeBytes = bytes.free
        }
        var cpu = host_cpu_load_info_data_t()
        var cpuCount = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
        let cpuStatus = withUnsafeMutablePointer(to:&cpu) { pointer in
            pointer.withMemoryRebound(to:integer_t.self,capacity:Int(cpuCount)) { host_statistics(host,HOST_CPU_LOAD_INFO,$0,&cpuCount) }
        }
        if cpuStatus == KERN_SUCCESS { result.cpuTicks = [cpu.cpu_ticks.0,cpu.cpu_ticks.1,cpu.cpu_ticks.2,cpu.cpu_ticks.3] }
        return result
    }
}
struct SystemMonitor {
    private var previousTicks:[UInt32]?
    private var sampledAt:TimeInterval?
    private var latest = SystemSnapshot.capture()
    mutating func sample() -> SystemSnapshot {
        let now = ProcessInfo.processInfo.systemUptime
        if let sampledAt, now-sampledAt < 0.5 { return latest }
        var snapshot = SystemSnapshot.capture()
        snapshot.cpuUsagePercent = SystemSnapshot.cpuPercent(previous:previousTicks,current:snapshot.cpuTicks ?? [])
        previousTicks = snapshot.cpuTicks; sampledAt = now; latest = snapshot
        return snapshot
    }
}
