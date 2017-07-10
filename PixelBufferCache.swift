import Foundation
import AVFoundation

class PixelBufferCache {
    private static var pools: [String: CVPixelBufferPool] = [:]
    
    class func pixelBufferAttributes(_ size: CGSize) -> [String: Any] {
        return [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: size.width,
            kCVPixelBufferHeightKey as String: size.height,
            kCVPixelFormatOpenGLESCompatibility as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
    }
    
    private class func pixelBufferPool(_ size: CGSize) -> CVPixelBufferPool? {
        var pixelBufferPool: CVPixelBufferPool?
        
        let poolAttributes: [String: Int] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 2]
        let pixelBufferAttributes: [String: Any] = PixelBufferCache.pixelBufferAttributes(size)
        
        let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes as CFDictionary?, pixelBufferAttributes as CFDictionary?, &pixelBufferPool)
        
        guard status == kCVReturnSuccess else { return nil }
        
        return pixelBufferPool
    }
    
    private class func getPool(_ size: CGSize) -> CVPixelBufferPool? {
        guard size != CGSize.zero else {return nil}
        
        let key = "\(size.width)_\(size.height)"
        
        if let pool = pools[key] {
            return pool
        }
        
        let pool = pixelBufferPool(size)
        pools[key] = pool
        
        return pool
    }
    
    class func createPixelBuffer(_ size: CGSize) -> CVPixelBuffer? {
        guard let pixelBufferPool = getPool(size) else { return nil }
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pixelBuffer)
        
        guard status == kCVReturnSuccess else { return nil }
        
        return pixelBuffer
    }
}
