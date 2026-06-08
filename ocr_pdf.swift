import Foundation
import PDFKit
import Vision

// Force running in UI/Main thread if needed, or standard script
autoreleasepool {
    let arguments = CommandLine.arguments
    guard arguments.count >= 3 else {
        print("Usage: swift ocr_pdf.swift <input_pdf_path> <output_txt_path>")
        exit(1)
    }

    let pdfPath = arguments[1]
    let txtPath = arguments[2]

    let pdfURL = URL(fileURLWithPath: pdfPath)
    guard let document = PDFDocument(url: pdfURL) else {
        print("Error: Could not open PDF document at \(pdfPath)")
        exit(1)
    }

    var fullText = ""
    let pageCount = document.pageCount

    for i in 0..<pageCount {
        guard let page = document.page(at: i) else { continue }
        
        // Get the page bounds
        let bounds = page.bounds(for: .cropBox)
        
        // Render PDF page to NSImage / CGImage
        // We use a high scale (e.g. 2.0 or 3.0) for good OCR quality (approx 200-300 DPI)
        let scale: CGFloat = 3.0
        let size = CGSize(width: bounds.width * scale, height: bounds.height * scale)
        
        let image = NSImage(size: size)
        image.lockFocus()
        if let context = NSGraphicsContext.current?.cgContext {
            // Fill background with white
            context.setFillColor(CGColor.white)
            context.fill(CGRect(origin: .zero, size: size))
            
            // Draw PDF page scaled
            context.scaleBy(x: scale, y: scale)
            // PDFKit draws relative to the bounds origin
            context.translateBy(x: -bounds.origin.x, y: -bounds.origin.y)
            page.draw(with: .cropBox, to: context)
        }
        image.unlockFocus()
        
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            print("Failed to render page \(i) to CGImage")
            continue
        }
        
        // Setup Vision OCR Request
        var pageText = ""
        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        let request = VNRecognizeTextRequest { request, error in
            if let error = error {
                print("OCR Error on page \(i): \(error)")
                return
            }
            guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
            
            for observation in observations {
                guard let topCandidate = observation.topCandidates(1).first else { continue }
                pageText += topCandidate.string + "\n"
            }
        }
        
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        
        do {
            try requestHandler.perform([request])
        } catch {
            print("Failed to perform OCR request on page \(i): \(error)")
        }
        
        fullText += "--- Page \(i + 1) ---\n" + pageText + "\n"
        print("  OCR: Processed page \(i + 1)/\(pageCount)")
    }

    do {
        try fullText.write(toFile: txtPath, atomically: true, encoding: .utf8)
        print("Successfully saved OCR text to \(txtPath)")
    } catch {
        print("Failed to write text to \(txtPath): \(error)")
        exit(1)
    }
}
