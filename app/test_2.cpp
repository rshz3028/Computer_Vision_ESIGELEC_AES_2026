#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 0 = default camera. Try 1,2... if you have multiple cams.
    cv::VideoCapture cap(0, cv::CAP_ANY);

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera.\n";
        std::cerr << "Try changing index (0->1) or backend (e.g. CAP_DSHOW).\n";
        return 1;
    }

    // Optional: set resolution (may or may not be respected)
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame;                 // grab a new frame
        if (frame.empty()) break;     // end or error

        cv::imshow("Camera", frame);

        int key = cv::waitKey(1) & 0xFF;  // 1ms delay
        if (key == 27 || key == 'q') {    // ESC or q
            break;
        } else if (key == 's') {
            cv::imwrite("frame.png", frame);
            std::cout << "Saved frame.png\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
