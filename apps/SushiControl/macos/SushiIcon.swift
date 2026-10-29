import AppKit

func sushiImage(size:CGFloat, appIcon:Bool = false) -> NSImage {
    let imageSize = NSSize(width:size,height:size)
    let bitmap = NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:max(1,Int((size*2).rounded(.up))),pixelsHigh:max(1,Int((size*2).rounded(.up))),bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:0,bitsPerPixel:0)!
    bitmap.size = imageSize
    let context = NSGraphicsContext(bitmapImageRep:bitmap)!.cgContext
    context.saveGState()
    context.scaleBy(x:size/32,y:size/32)
    context.translateBy(x:0,y:32)
    context.scaleBy(x:1,y:-1)
    context.setShouldAntialias(true)

    if appIcon {
        let tile = CGPath(roundedRect:CGRect(x:0.5,y:0.5,width:31,height:31),cornerWidth:7,cornerHeight:7,transform:nil)
        context.setFillColor(CGColor(gray:0.975,alpha:1))
        context.addPath(tile)
        context.fillPath()

        let nori = CGColor(gray:0.055,alpha:1)
        let rice = CGColor(gray:0.975,alpha:1)
        context.setFillColor(nori)
        context.fillEllipse(in:CGRect(x:5,y:5,width:22,height:22))
        context.setFillColor(rice)
        context.fillEllipse(in:CGRect(x:8,y:8,width:16,height:16))
        context.setFillColor(nori)
        context.fillEllipse(in:CGRect(x:12,y:12,width:8,height:8))
        context.setFillColor(rice)
        context.fillEllipse(in:CGRect(x:14,y:14,width:2,height:2))
        context.fillEllipse(in:CGRect(x:17,y:14,width:2,height:2))
        context.fillEllipse(in:CGRect(x:15.5,y:17,width:2,height:2))

        context.setStrokeColor(CGColor(gray:0.82,alpha:1))
        context.setLineWidth(0.5)
        context.addPath(tile)
        context.strokePath()
    } else {
        context.setFillColor(CGColor(gray:0,alpha:1))
        context.fillEllipse(in:CGRect(x:5.5,y:5.5,width:21,height:21))
        context.setBlendMode(.clear)
        context.fillEllipse(in:CGRect(x:9,y:9,width:14,height:14))
        context.setBlendMode(.normal)
        context.setFillColor(CGColor(gray:0,alpha:1))
        context.fillEllipse(in:CGRect(x:12,y:12,width:8,height:8))
        context.setBlendMode(.clear)
        context.fillEllipse(in:CGRect(x:14,y:14,width:1.7,height:1.7))
        context.fillEllipse(in:CGRect(x:17,y:14,width:1.7,height:1.7))
        context.fillEllipse(in:CGRect(x:15.15,y:17,width:1.7,height:1.7))
        context.restoreGState()
        let image = NSImage(size:imageSize)
        image.addRepresentation(bitmap)
        image.isTemplate = true
        image.accessibilityDescription = "Sushi roll"
        return image
    }

    context.restoreGState()
    let image = NSImage(size:imageSize)
    image.addRepresentation(bitmap)
    image.isTemplate = false
    image.accessibilityDescription = "Sushi Control"
    return image
}
