/*
  Copyright (c) 2016-2017 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

import UIKit
import Metal
import MetalPerformanceShaders
import CoreMedia
import Forge
import GLKit

let MaxBuffersInFlight = 3   // use triple buffering

/*
  The neural network from Google's MobileNets paper.
  
	The paper says MobileNet-224 with alpha=1.0 has 4.2M parameters. 
  We have 4,216,072, so that seems to be correct.
*/

class CameraViewController: UIViewController {

  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var predictionLabel: UILabel!
  @IBOutlet weak var extraLabel: UILabel!
  @IBOutlet weak var timeLabel: UILabel!
  @IBOutlet weak var debugImageView: UIImageView!
    @IBOutlet weak var preview: GLKView!
    
  var videoCapture: VideoCapture!
  var device: MTLDevice!
  var commandQueue: MTLCommandQueue!
  var runner: Runner!
  var network: MobileNet!
    
    var context:CIContext!
 
  var startupGroup = DispatchGroup()
    
  var output_image : CIImage!
    var input_texture : MTLTexture!

  override func viewDidLoad() {
    super.viewDidLoad()
    
   // predictionLabel.text = ""
   // extraLabel.text = ""
    timeLabel.text = ""

    device = MTLCreateSystemDefaultDevice()
    if device == nil {
      print("Error: this device does not support Metal")
      return
    }

    commandQueue = device.makeCommandQueue()

    // NOTE: At this point you'd disable the UI and show a spinner.

    videoCapture = VideoCapture(device: device)
  
    videoCapture.openSession()
    self.resizePreviewLayer()
    videoCapture.delegate = self
    context = videoCapture.context
   
    videoCapture.preview = preview
    
    

    // Initialize the neural network.
    startupGroup.enter()
    createNeuralNetwork {
      self.startupGroup.leave()
    }

    // Once the NN is set up, we can start capturing live video.
    startupGroup.notify(queue: .main) {
      // NOTE: At this point you'd remove the spinner and enable the UI.

      self.videoCapture.start()
    }
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  // MARK: - UI stuff

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  func resizePreviewLayer() {
   // videoCapture.previewLayer?.frame = videoPreview.bounds
  //  videoCapture.preview?.frame = preview.bounds
   // print(preview.bounds)

 
}

  // MARK: - Neural network

  func createNeuralNetwork(completion: @escaping () -> Void) {
    // Make sure the current device supports MetalPerformanceShaders.
    guard MPSSupportsMTLDevice(device) else {
      print("Error: this device does not support Metal Performance Shaders")
      return
    }

    runner = Runner(commandQueue: commandQueue, inflightBuffers: MaxBuffersInFlight)

    // Because it may take a few seconds to load the network's parameters,
    // perform the construction of the neural network in the background.
    DispatchQueue.global().async {

      timeIt("Setting up neural network") {
        /*
        self.network = MobileNet(device: self.device,
                                 widthMultiplier: 1,
                                 resolutionMultiplier: 1,
                                 shallow: false,
                                 inflightBuffers: MaxBuffersInFlight)
 */
        self.network = MobileNet(device: self.device, inflightBuffers: MaxBuffersInFlight)
        
      }
      DispatchQueue.main.async(execute: completion)
    }
  }
    /*
    public func predict<NeuralNetworkType: NeuralNetwork>(
        network: NeuralNetworkType,
        texture inputTexture: MTLTexture,
        queue: DispatchQueue,
        completion: @escaping (NeuralNetworkResult<NeuralNetworkType.PredictionType>) -> Void)
*/
    func predict2(texture: MTLTexture, queue: DispatchQueue,
                  completion: @escaping (CIImage) -> Void){
        
        //queue.async { completion(result)}
    }
    
  func predict(texture: MTLTexture){
    // Since we want to run in "realtime", every call to predict() results in
    // a UI update on the main thread. It would be a waste to make the neural
    // network do work and then immediately throw those results away, so the 
    // network should not be called more often than the UI thread can handle.
    // It is up to VideoCapture to throttle how often the neural network runs.

    runner.predict(network: network, texture: texture, queue: .main) { result in
        /*
        let colorSpace=CGColorSpaceCreateDeviceRGB()
        
        let fm = FileManager.default
        let docsurl = try! fm.url(for:.documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let myurl = docsurl.appendingPathComponent("input_raw.jpg")
        //let myurl2 = docsurl.appendingPathComponent("input.jpg")
        let input = CIImage(mtlTexture:texture )
      //  try! CIContext().writeJPEGRepresentation(of:input! ,to:myurl ,colorSpace: colorSpace as! CGColorSpace )
        */
        self.process(predictions: result.predictions, texture: texture)
        
        
        
        self.timeLabel.text = String(format: "Elapsed %.5f seconds (%.2f FPS)", result.elapsed, 1/result.elapsed)
      
    }
  }

  private func show(predictions: [MobileNet.Prediction]) {
    output_image = predictions[0].output
    
   // output_image =
  }
    private func process(predictions: [MobileNet.Prediction], texture:MTLTexture) {
        output_image = predictions[0].output
      //  input_texture = texture
       let input = CIImage(mtlTexture:texture )!
        
        
        let scale = Double(1280)/48//Double(preview.bounds.height*2)/48
        //let scale = Double(preview.bounds.height*2)/48
       // print(scale)
        
        //print(preview.bounds.width)
        //print(preview.bounds.height)
        let filter0 = CIFilter(name: "CILanczosScaleTransform")!
        filter0.setValue(output_image, forKey: "inputImage")
        filter0.setValue(scale, forKey: "inputScale")
       // filter0.setValue(Double(1600)/Double(1200), forKey: "inputAspectRatio")
       // filter0.setValue(Double(preview.bounds.height)/Double(preview.bounds.width), forKey: "inputAspectRatio")
        filter0.setValue(Double(720)/Double(1280), forKey: "inputAspectRatio")
        //filter0.setValue(1, forKey: "inputAspectRatio")
        let outputImage = filter0.value(forKey: "outputImage") as! CIImage

        //print(preview.bounds.width)
        //print(preview.bounds.width)
        
        let filter = CIFilter(name: "CIBlendWithMask")!
        filter.setValue(input, forKey: kCIInputBackgroundImageKey)
        filter.setValue(outputImage, forKey: kCIInputImageKey)
        filter.setValue(outputImage, forKey: kCIInputMaskImageKey)
        
        let image = filter.value(forKey: kCIOutputImageKey) as! CIImage
        

        var rect:CGRect = preview.bounds
        rect.origin.x = 0
        rect.origin.y = 0
        rect.size.width = preview.bounds.width*2
        rect.size.height = preview.bounds.height*2
        
        let colorSpace=CGColorSpaceCreateDeviceRGB()
        
        
        let fm = FileManager.default
        let docsurl = try! fm.url(for:.documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let myurl = docsurl.appendingPathComponent("output3.jpg")
        let myurl2 = docsurl.appendingPathComponent("input.jpg")
        
       // try! CIContext().writeJPEGRepresentation(of:image ,to:myurl ,colorSpace: colorSpace as! CGColorSpace )
      //  try! CIContext().writeJPEGRepresentation(of:input ,to:myurl2 ,colorSpace: colorSpace as! CGColorSpace )
      //  preview.bindDrawable()
        /*
        self.context.draw(input,
                          in: CGRect(x: 0, y: 0,
                                     width: rect.size.width,
                                     height: rect.size.height),
                          from: image.extent)
 */
        self.context.draw(image, in: rect, from: image.extent)
/*
        */
      //  self.debugImageView.image = UIImage.image(texture: predictions[0].input!)
        
        preview.display()
        
    }


}

extension CameraViewController: VideoCaptureDelegate {
   
    func test(didCaptureVideoTexture texture: MTLTexture?){
        
    }

 
    func videoCaptureRev(_ capture: VideoCapture, didCaptureVideoTexture2 texture: MTLTexture?, timestamp: CMTime) {
        let colorSpace=CGColorSpaceCreateDeviceRGB()
        
        let fm = FileManager.default
       // let or = simulatedOrientation.asCGImagePropertyOrientation()
       // let ci = CIImage(cvPixelBuffer:pb).applyingOrientation(or)

        
        let docsurl = try! fm.url(for:.documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let myurl = docsurl.appendingPathComponent("input_raw0.jpg")
        //let myurl2 = docsurl.appendingPathComponent("input.jpg")
        let input = CIImage(mtlTexture:texture! )
       // try! CIContext().writeJPEGRepresentation(of:input! ,to:myurl ,colorSpace: colorSpace as! CGColorSpace )
        
        
        
        if let texture = texture {
            predict(texture: texture)
           
          
        }
      //  return output_image
    }

  func videoCapture(_ capture: VideoCapture, didCapturePhotoTexture texture: MTLTexture?, previewImage: UIImage?) {
    // not implemented
  }
}
