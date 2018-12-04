/*
Copyright (c) 2018, umikry.com
License AGPL-3.0

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License, version 3,
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License, version 3,
along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>

#include "umikrycore/FaceDetector.hpp"
#include "umikrycore/FaceTransformator.hpp"

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
    string model_path = ".";

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
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
        auto image = imread(source);
        if (!image.data) {
          cout << "The image '" << source << "' is not readable." << endl;
          return 1;
        }

        cout << image.dims << " " << image.size() << " " << image.rows << " " << image.cols << " " << image.channels() << endl; 

        umikry::FaceDetector umikryFaceDetector(umikry::DetectionMethod::CAFFE, model_path);
        
        auto start = chrono::steady_clock::now();
		auto faces = umikryFaceDetector.detect(image);
        auto end = chrono::steady_clock::now();
        auto elapsed_seconds = end - start;

        cout << "Detect " << faces.size() << " faces in " 
             << elapsed_seconds.count()
             << " s" << endl;
        cout << "Use transformation method: blur" << endl;
        
        umikry::FaceTransformator umikryFaceTransformator(umikry::TransformationMethod::BLUR);
        
        start = chrono::steady_clock::now();
        umikryFaceTransformator.transform(image, faces);
        end = chrono::steady_clock::now();
        elapsed_seconds = end - start;
        
        cout << "Transform " << faces.size() << " faces in " 
             << elapsed_seconds.count()
             << " s" << endl;

        for (const Rect face : faces) {
            rectangle(image, face, Scalar(0, 0, 255), 2);
        }

        imwrite(destination, image);
        return 0;
    } else {
        cerr << "Please specify a source image and a destination." << endl;
        return 1;
    }
}
