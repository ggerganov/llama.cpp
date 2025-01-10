import XCTest
@testable import omnivlm

final class LlavaTests: XCTestCase {
    func testOmniVlm() {
        omnivlm_init("/Users/liute/Downloads/model-q4_0.gguf",
                     "/Users/liute/Downloads/projector-q4_0.gguf",
                     "vlm-81-instruct")

        let startTime = Date()

        if let cString = omnivlm_inference("describe the image", "/Users/liute/Downloads/cat.png") {
            let res = String(cString: cString)
            print("res: \(res)")
            
            let endTime = Date()
            let inferenceTime = endTime.timeIntervalSince(startTime)
            print("Inference time: \(inferenceTime) seconds")

            XCTAssertFalse(res.isEmpty, "res should not be null")
        } else {
            XCTFail("failed")
        }
        
        omnivlm_free()
    }
}
