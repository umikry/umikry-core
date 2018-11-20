#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "UmikryFaceDetector.hpp"

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
        vector<Rect> faces = umikryFaceDetector.detect(image);

        cout << "Found " << faces.size() << " faces." << endl;

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