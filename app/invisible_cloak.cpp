#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>

using Clock = std::chrono::steady_clock;

static inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

static void drawTextOutlined(cv::Mat& img, const std::string& text, cv::Point org,
                             double scale, int thickness,
                             const cv::Scalar& fg, const cv::Scalar& outline) {
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, outline, thickness + 2, cv::LINE_AA);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv::LINE_AA);
}

static cv::Mat makeCloakMask(const cv::Mat& frameBgr,
                             const cv::Mat& bgBgr,
                             int darkThresh,     // 0..255 (lower = stricter black)
                             int diffThresh) {   // 0..255 (how different from background)
    CV_Assert(frameBgr.size() == bgBgr.size());
    CV_Assert(frameBgr.type() == CV_8UC3 && bgBgr.type() == CV_8UC3);

    // Convert to grayscale for simple "dark" test + "difference" test
    cv::Mat g, bgG;
    cv::cvtColor(frameBgr, g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgBgr, bgG, cv::COLOR_BGR2GRAY);

    // 1) Dark pixels in current frame
    cv::Mat darkMask;
    cv::threshold(g, darkMask, darkThresh, 255, cv::THRESH_BINARY_INV); // g < darkThresh => 255

    // 2) Pixels that changed vs background (so we only remove NEW dark areas)
    cv::Mat diff, diffMask;
    cv::absdiff(g, bgG, diff);
    cv::threshold(diff, diffMask, diffThresh, 255, cv::THRESH_BINARY); // diff > diffThresh => 255

    // 3) Cloak mask = dark AND changed
    cv::Mat cloakMask;
    cv::bitwise_and(darkMask, diffMask, cloakMask);

    // Clean up mask (reduce holes + speckles)
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(cloakMask, cloakMask, cv::MORPH_OPEN, k, cv::Point(-1, -1), 1);
    cv::morphologyEx(cloakMask, cloakMask, cv::MORPH_CLOSE, k, cv::Point(-1, -1), 2);

    // Optional: soften edges slightly for nicer composite
    cv::GaussianBlur(cloakMask, cloakMask, cv::Size(7, 7), 1.5);
    cv::threshold(cloakMask, cloakMask, 80, 255, cv::THRESH_BINARY);

    return cloakMask; // 0/255 mask
}

int main() {
    const int W = 640, H = 480;
    const bool mirror = true;

    int darkThresh = 55;  // start here: higher -> more things count as "black"
    int diffThresh = 25;  // higher -> needs stronger change vs background

    cv::VideoCapture cap(0, cv::CAP_DSHOW); // try CAP_ANY/CAP_MSMF if needed
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera.\n";
        return 1;
    }

    cv::namedWindow("CloakReplace", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Mask", cv::WINDOW_AUTOSIZE);

    // Capture initial background
    cv::Mat bg;
    {
        cv::Mat f;
        cap >> f;
        if (f.empty()) { std::cerr << "ERROR: Empty frame.\n"; return 1; }
        cv::resize(f, f, cv::Size(W, H));
        if (mirror) cv::flip(f, f, 1);
        bg = f.clone();
    }

    std::cout << "Controls:\n"
              << "  b = recapture background (empty room)\n"
              << "  [ / ] = decrease/increase dark threshold\n"
              << "  - / = = decrease/increase diff threshold\n"
              << "  q or Esc = quit\n\n";

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::resize(frame, frame, cv::Size(W, H));
        if (mirror) cv::flip(frame, frame, 1);

        cv::Mat cloakMask = makeCloakMask(frame, bg, darkThresh, diffThresh);

        // Composite: start from current frame, replace masked pixels with background
        cv::Mat out = frame.clone();
        bg.copyTo(out, cloakMask);

        // UI text
        drawTextOutlined(out, "b=bg  [ ] darkThresh  -/= diffThresh  q=quit",
                         cv::Point(10, H - 20), 0.6, 2,
                         cv::Scalar(255,255,255), cv::Scalar(0,0,0));

        drawTextOutlined(out,
                         "darkThresh=" + std::to_string(darkThresh) +
                         "  diffThresh=" + std::to_string(diffThresh),
                         cv::Point(10, 30), 0.8, 2,
                         cv::Scalar(255,255,255), cv::Scalar(0,0,0));

        cv::imshow("CloakReplace", out);
        cv::imshow("Mask", cloakMask);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;

        if (key == 'b') {
            bg = frame.clone();
            std::cout << "Background recaptured.\n";
        }

        if (key == '[') { darkThresh = clampi(darkThresh - 2, 0, 255); }
        if (key == ']') { darkThresh = clampi(darkThresh + 2, 0, 255); }

        if (key == '-') { diffThresh = clampi(diffThresh - 2, 0, 255); }
        if (key == '=') { diffThresh = clampi(diffThresh + 2, 0, 255); }
    }

    return 0;
}
