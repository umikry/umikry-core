#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

static void show_usage(string name) {
    cerr << "Usage: " << name << " <option(s)>\n"
         << "Options:\n"
         << "\t-h,--help\t\tShow this help message\n"
         << "\t-s,--source \t\tSpecify the source image path\n"
         << "\t-d,--destination \tSpecify the destination path" << endl;
}

int main(int argc, const char** argv) {
    if (argc < 3) {
        show_usage(argv[0]);
        return 1;
    }

    string source;
    string destination;

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
        } 
    }

    if (!source.empty() && !destination.empty()) {
    	Mat image;
    	image = imread(source);

    	if (!image.data) {
		  cout << "The image '" << source << "' is not readable." << endl;
		  return 1;
		}

		// TODO: Implement umikry core components 
		cout << "umikry-core (C++) is not implement yet. " 
			 << "For now it just copies the source to destination." << endl;

		imwrite(destination, image);
		return 0;
    } else {
    	cerr << "Please specify a source image and a destination." << endl;
        return 1;
    }
}