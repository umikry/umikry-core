#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>

#include "UmikryFaceDetector.hpp"
#include "UmikryFaceTransformator.hpp"

using namespace std;
using namespace cv;

static void show_usage(string name) {
    cerr << "Usage: " << name << " <option(s)>\n"
         << "Options:\n"
         << "\t-h,--help\t\tShow this help message\n"
         << "\t-s,--source \t\tSpecify the source image path\n"
         << "\t-m,--models \t\tPath to the models folder\n"
         << "\t-d,--destination \tSpecify the destination path" << endl;
}

int main(int argc, const char** argv) {
    if (argc < 3) {
        show_usage(argv[0]);
        return 1;
    }

    string source;
    string destination;
    string model_path;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        } else if ((arg == "-s") || (arg == "--source")) {
            if (i + 1 < argc) {
                source = argv[++i];
            } else {
                cerr << "--source option requires an argument." << endl;
                return 1;
            }  
        } else if ((arg == "-d") || (arg == "--destination")) {
            if (i + 1 < argc) {
                destination = argv[++i];
            } else {
                cerr << "--destination option requires an argument." << endl;
                return 1;
            }  
        } else if ((arg == "-m") || (arg == "--models")) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                cerr << "--models option requires an argument." << endl;
                return 1;
            }  
        } 
    }

    if (!source.empty() && !destination.empty() && !model_path.empty()) {
        Mat image = imread(source);

        if (!image.data) {
          cout << "The image '" << source << "' is not readable." << endl;
          return 1;
        }

        UmikryFaceDetector umikryFaceDetector = UmikryFaceDetector(DetectionMethod::CAFFE, model_path);
        
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        vector<Rect> faces = umikryFaceDetector.detect(image);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        cout << "Detect " << faces.size() << " faces in " 
             << elapsed_seconds.count()
             << " s" << endl;
        cout << "Use transformation method: blur" << endl;
        
        UmikryFaceTransformator umikryFaceTransformator = UmikryFaceTransformator(TransformationMethod::BLUR);
        
        start = std::chrono::steady_clock::now();
        umikryFaceTransformator.transform(image, faces);
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;
        
        cout << "Transform " << faces.size() << " faces in " 
             << elapsed_seconds.count()
             << " s" << endl;

        for (Rect face : faces) {
            rectangle(image, face, Scalar(0, 0, 255), 2);
        }

        imwrite(destination, image);
        return 0;
    } else {
        cerr << "Please specify a source image and a destination." << endl;
        return 1;
    }
}