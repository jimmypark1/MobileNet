import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders
import Forge

/**
  The neural network from the paper "MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications" https://arxiv.org/abs/1704.04861v1

  **NOTE:** This is currently using random parameters; the network *hasn't been 
  trained on anything yet*, so the predictions don't make any sense at all! 
  I just wanted to see how fast/slow this network architecture is on iPhone.
*/
class MobileNet: NeuralNetwork {
    typealias Prediction = (label: String, output: CIImage)

    
 // let classes: Int

    let model: Model
 
   // let resizeLayer: Resize
   // let grayscale: Tensor
    

    public init(device: MTLDevice,
                widthMultiplier: Float = 1,
                resolutionMultiplier: Float = 1,
                shallow: Bool = false,
                classes: Int = 1000,
                inflightBuffers: Int) {

        
        
       // let resolution = Int(300 * resolutionMultiplier)
        let relu = MPSCNNNeuronReLU(device: device, a: 0)
    
        let sigmoid = MPSCNNNeuronSigmoid(device: device)
        
        let input = Input()

        
        /// Simple SegNet
        let x = input
            --> Resize(width: 48, height: 48)
            --> Convolution(kernel: (5, 5), channels: 64, stride: (1, 1),padding:true, activation: relu, useBias:true, name: "conv1")
            
            --> MaxPooling(kernel: (3, 3), stride: (2, 2),padding:true,edgeMode:.clamp)
            
            --> Convolution(kernel: (5, 5), channels: 64, stride: (1, 1),padding:true,activation: relu, useBias:true, name: "conv2")
            
            --> MaxPooling(kernel: (2, 2), stride: (2, 2),padding:true,edgeMode:.clamp)
         
            --> Convolution(kernel: (3, 3), channels: 64, stride: (1, 1),padding:true,activation:relu, useBias:true, name: "conv3")
            --> Dense(neurons: 100,activation: sigmoid, useBias:true, name: "fc4")
            --> Dense(neurons: 400,activation: sigmoid, useBias:true, name: "fc5")
            --> Dense(neurons: 48*48, activation: sigmoid,useBias:true, name: "fc6")
    
        model = Model(input: input, output: x)

    
        let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
        
            name, count, type in ParameterLoaderBundle(name: name,
                                                   count: count,
                                                   suffix: type == .weights ? "_W" : "_b",
                                                   ext: "bin")

        }

        if success {
            print(model.summary())
        }
    }
    

  public func encode(commandBuffer: MTLCommandBuffer, texture inputTexture: MTLTexture, inflightIndex: Int) {

    
    model.encode(commandBuffer: commandBuffer, texture: inputTexture, inflightIndex: inflightIndex)
    
    let fm = FileManager.default
    let docsurl = try! fm.url(for:.documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
    let myurl = docsurl.appendingPathComponent("input_mtl_texture.jpg")
    //  let myurl2 = docsurl.appendingPathComponent("input.jpg")
    let colorSpace=CGColorSpaceCreateDeviceRGB()
    let input = CIImage(mtlTexture:(inputTexture)) as! CIImage
    
   // try! CIContext().writeJPEGRepresentation(of:input ,to:myurl ,colorSpace: colorSpace as! CGColorSpace )

  }


    
public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {
    let probabilities = model.outputImage(inflightIndex: inflightIndex).toFloatArray()
  //  print(probabilities)
    var result = NeuralNetworkResult<Prediction>()
    
    var bytes = [UInt8](repeating : 0, count : 2304*4)

    for i in 0...2303 {
        
        bytes[4*i] =  UInt8(255*probabilities[i])
        bytes[4*i+1] = UInt8(255*probabilities[i])
        bytes[4*i+2] =  UInt8(255*probabilities[i])
        bytes[4*i+3] = 150
       
        if probabilities[i] > 0.7
        {
            bytes[4*i] =  255
            bytes[4*i+1] = 255
            bytes[4*i+2] =  255
            
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

    
    //result.debugTexture = grayscale.image?.texture
    
    result.ciImage = output_Image;
   // print(grayscale)
    result.predictions.append((label: "\(maxIndex)", output: output_Image))
  //  print(inputData1)
   // result.predictionsImg.append(output_Image)
    
    return result
 
   
  }
}
