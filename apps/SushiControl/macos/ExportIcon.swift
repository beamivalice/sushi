import AppKit

@main struct ExportIcon {
    static func main() throws {
        let folder = URL(fileURLWithPath:CommandLine.arguments[1])
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        for size in [16,32,128,256,512] {
            for scale in [1,2] {
                let pixels = size * scale
                let bitmap = NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:pixels,pixelsHigh:pixels,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:0,bitsPerPixel:0)!
                NSGraphicsContext.saveGraphicsState()
                NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep:bitmap)
                sushiImage(size:CGFloat(pixels),appIcon:true).draw(in:NSRect(x:0,y:0,width:pixels,height:pixels))
                NSGraphicsContext.restoreGraphicsState()
                let suffix = scale == 2 ? "@2x" : ""
                try bitmap.representation(using:.png,properties:[:])!.write(to:folder.appendingPathComponent("icon_\(size)x\(size)\(suffix).png"))
            }
        }
    }
}
