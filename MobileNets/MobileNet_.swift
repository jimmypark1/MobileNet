import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders
import Forge


class MobileNet: NeuralNetwork {
    typealias Prediction = (label: String, output: CIImage)

   // let model: Model
    var outputImage: [MPSImage] = []
    

    
    let lanczos: MPSImageLanczosScale
    let relu: MPSCNNNeuronReLU
    
    let conv1: MPSCNNConvolution
    let pool1: MPSCNNPoolingMax
    let conv2: MPSCNNConvolution
    let pool2: MPSCNNPoolingMax
    let conv3: MPSCNNConvolution

    
    let fc4: MPSCNNFullyConnected
    let fc5: MPSCNNFullyConnected
    let fc6: MPSCNNFullyConnected
    
    let scaledImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 48, height: 48, featureChannels: 3)
    let conv1ImgDesc  = MPSImageDescriptor(channelFormat: .float16, width: 48, height: 48, featureChannels: 64)
    let pool1ImgDesc  = MPSImageDescriptor(channelFormat: .float16, width: 24, height: 24, featureChannels: 64)
    
    let conv2ImgDesc  = MPSImageDescriptor(channelFormat: .float16, width: 24, height: 24, featureChannels: 64)
    let pool2ImgDesc  = MPSImageDescriptor(channelFormat: .float16, width: 12, height:  12, featureChannels: 64)
   
    let conv3ImgDesc  = MPSImageDescriptor(channelFormat: .float16, width: 12, height: 12, featureChannels: 64)

    let fc4ImgDesc    = MPSImageDescriptor(channelFormat: .float16, width:  1, height:  1, featureChannels: 100)
    let fc5ImgDesc    = MPSImageDescriptor(channelFormat: .float16, width:  1, height:  1, featureChannels: 400)
 
    let fc6ImgDesc    = MPSImageDescriptor(channelFormat: .float16, width:  1, height:  1, featureChannels: 2304)
   
    
    public init(device: MTLDevice, inflightBuffers: Int)
    {
  
        let sigmoid = MPSCNNNeuronSigmoid(device: device)
        
        
        for _ in 0..<inflightBuffers {
            outputImage.append(MPSImage(device: device, imageDescriptor: fc6ImgDesc))
        }

        lanczos = MPSImageLanczosScale(device: device)
        relu = MPSCNNNeuronReLU(device: device, a: 0)
        
        weightsLoader = { name, count in ParameterLoaderBundle(name: name, count: count, suffix: "_W", ext: "bin") }
        biasLoader = { name, count in ParameterLoaderBundle(name: name, count: count, suffix: "_b", ext: "bin") }

        
        conv1 = convolution(device: device, kernel: (5, 5), inChannels: 3, outChannels: 64, activation: relu, name: "conv1")
        pool1 = maxPooling(device: device, kernel: (3, 3), stride: (2, 2))
        conv2 = convolution(device: device, kernel: (5, 5), inChannels: 64, outChannels: 64, activation: relu, name: "conv2")
        pool2 = maxPooling(device: device, kernel: (3, 3), stride: (2, 2))
        
        conv3 = convolution(device: device, kernel: (3, 3), inChannels: 64, outChannels: 64, activation: relu, name: "conv3")
        
        fc4 = dense(device: device, shape: (12, 12), inChannels: 64, fanOut: 100, activation: sigmoid, name: "fc4")
        fc5 = dense(device: device, fanIn: 100, fanOut: 400, activation: sigmoid, name: "fc5")
        fc6 = dense(device: device, fanIn: 400, fanOut: 2304, activation: sigmoid, name: "fc6")

    
    
    }
    

    public func encode(commandBuffer: MTLCommandBuffer, texture inputTexture: MTLTexture, inflightIndex: Int) {
        //  resizeLayer.setCropRect(x: 0, y: 60, width: 360, height: 360)

        MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: [
        scaledImgDesc,  conv1ImgDesc, pool1ImgDesc,
        conv2ImgDesc, pool2ImgDesc, conv3ImgDesc,fc4ImgDesc, fc5ImgDesc,fc6ImgDesc])
    
    
        let scaledImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: scaledImgDesc)
        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputTexture, destinationTexture: scaledImg.texture)
    
        let conv1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1ImgDesc)
        conv1.applyPadding(type: .same, sourceImage: scaledImg, destinationImage: conv1Img)
        conv1.encode(commandBuffer: commandBuffer, sourceImage: scaledImg, destinationImage: conv1Img)
    
        let pool1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool1ImgDesc)
        pool1.encode(commandBuffer: commandBuffer, sourceImage: conv1Img, destinationImage: pool1Img)
    
        let conv2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2ImgDesc)
        conv2.applyPadding(type: .same, sourceImage: pool1Img, destinationImage: conv2Img)
        conv2.encode(commandBuffer: commandBuffer, sourceImage: pool1Img, destinationImage: conv2Img)
    
        let pool2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool2ImgDesc)
        pool2.encode(commandBuffer: commandBuffer, sourceImage: conv2Img, destinationImage: pool2Img)
   
        
        let conv3Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3ImgDesc)
        conv3.applyPadding(type: .same, sourceImage: pool1Img, destinationImage: conv3Img)
        conv3.encode(commandBuffer: commandBuffer, sourceImage: pool2Img, destinationImage: conv3Img)

        
        let fc4Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc4ImgDesc)
        fc4.encode(commandBuffer: commandBuffer, sourceImage: conv3Img, destinationImage: fc4Img)
   
        let fc5Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc5ImgDesc)
        fc5.encode(commandBuffer: commandBuffer, sourceImage: fc4Img, destinationImage: fc5Img)

        let fc6Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc6ImgDesc)
        fc6.encode(commandBuffer: commandBuffer, sourceImage: fc5Img, destinationImage: fc6Img)

    // Finally, apply the softmax function to the output of the last layer.
    // The output image is not an MPSTemporaryImage but a regular MSPImage.
  }


    
public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {
    
    /*
    let probabilities = model.outputImage(inflightIndex: inflightIndex).toFloatArray()
    //print(probabilities)
        
    var bytes = [UInt8](repeating : 0, count : 2304*4)

    for i in 0...2303 {
        
        bytes[4*i] =  UInt8(255*probabilities[i])
        bytes[4*i+1] = UInt8(255*probabilities[i])
        bytes[4*i+2] =  UInt8(255*probabilities[i])
        bytes[4*i+3] = 100
       
        if probabilities[i] > 0.85
        {
            bytes[4*i] =  0
            bytes[4*i+1] = 0
            bytes[4*i+2] =  0
            
            bytes[4*i+3] =  0
        }
        else
        {
            bytes[4*i] =  0
            bytes[4*i+1] = 255
            bytes[4*i+2] =  0
            bytes[4*i+3] = 255
            
        }
 
        
        
       
    }
    let colorSpace=CGColorSpaceCreateDeviceRGB()

    let inputData1 = NSData(bytes: bytes, length: 2304*4 )
    let output_Image = CIImage(bitmapData: inputData1 as Data, bytesPerRow: 4*48, size: CGSize(width:48,height:48), format: kCIFormatBGRA8, colorSpace:colorSpace)
  
    let (maxIndex, maxValue) = probabilities.argmax()
  
    
    let fm = FileManager.default
    let docsurl = try! fm.url(for:.documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
    let myurl = docsurl.appendingPathComponent("output11.jpg")
    //let myurl2 = docsurl.appendingPathComponent("input.jpg")
    
   // try! CIContext().writeJPEGRepresentation(of:output_Image ,to:myurl ,colorSpace: colorSpace as! CGColorSpace )
    //   try! CIContext().writeJPEGRepresentation(of:input ,to:myurl2 ,colorSpace: colorSpace as! CGColorSpace )
    */
    
    var result = NeuralNetworkResult<Prediction>()
    //result.debugTexture = grayscale.image?.texture
 /*
    result.ciImage = output_Image;
   // print(grayscale)
    result.predictions.append((label: "\(maxIndex)", output: output_Image))
  //  print(inputData1)
   // result.predictionsImg.append(output_Image)
    */
    return result
 
   
  }
}
