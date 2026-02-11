// test_5.cpp
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>    // std::lround
#include <iostream>
#include <string>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

using Clock = std::chrono::steady_clock;

static void beepAlarm() {
#ifdef _WIN32
    Beep(1200, 120);
    Beep(900, 120);
#endif
}

static void drawTextOutlined(cv::Mat& img, const std::string& text, cv::Point org,
                             double scale, int thickness,
                             const cv::Scalar& fg, const cv::Scalar& outline) {
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, outline, thickness + 2, cv::LINE_AA);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv::LINE_AA);
}

static cv::Mat preprocessForDiff(const cv::Mat& bgr, int targetW, int targetH) {
    cv::Mat smallImg, grayImg, blurImg;
    cv::resize(bgr, smallImg, cv::Size(targetW, targetH), 0, 0, cv::INTER_AREA);
    cv::cvtColor(smallImg, grayImg, cv::COLOR_BGR2GRAY);
    // blur reduces sensor noise / tiny flicker
    cv::GaussianBlur(grayImg, blurImg, cv::Size(7, 7), 1.2);
    return blurImg; // CV_8UC1
}

// compute SAD on preprocessed images (optionally masked)
static double sadScore(const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask /*0/255 or empty*/) {
    cv::Mat diff;
    cv::absdiff(a, b, diff); // 8-bit

    if (!mask.empty()) {
        cv::Mat masked;
        // keep diff only where mask != 0
        cv::bitwise_and(diff, diff, masked, mask);
        return cv::sum(masked)[0];
    } else {
        return cv::sum(diff)[0];
    }
}

int main() {
    const int W = 640, H = 480;
    const bool mirror = false; // alarm cams usually not mirrored

    // Downscale for faster + more stable metric
    const int DW = 160, DH = 120;

    // Thresholding strategy:
    // We compare SAD per pixel (normalized) instead of raw SAD
    // Typical starting values:
    // - still room: 0.5 .. 2.0
    // - subtle motion: 3 .. 8
    // - person enters: 10+ (depends on lighting)
    double presenceThreshold = 6.0; // normalized SAD per pixel (tweak)

    // Debounce / cooldown (avoid rapid flicker alarms)
    const double minAlarmHoldSec = 1.0;   // once alarmed, stay alarm for at least this long
    const double minClearHoldSec = 1.0;   // once cleared, stay clear for at least this long

    cv::VideoCapture cap(0, cv::CAP_DSHOW); // try CAP_ANY / CAP_MSMF on Windows if needed
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera.\n";
        return 1;
    }

    cv::namedWindow("Alarm", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "ERROR: Empty frame.\n";
        return 1;
    }
    cv::resize(frame, frame, cv::Size(W, H));

    // Background reference
    cv::Mat bgBgr = frame.clone();
    cv::Mat bgPre = preprocessForDiff(bgBgr, DW, DH);

    // Optional ROI mask (select stable region; ignore windows/TV)
    cv::Rect roi;
    cv::Mat maskSmall; // 0/255 in DWxDH

    bool alarmState = false;
    auto lastStateChange = Clock::now();

    std::cout << "Controls:\n"
              << "  b = recapture background\n"
              << "  r = select ROI (ignore outside)\n"
              << "  c = clear ROI (use full frame)\n"
              << "  +/- = adjust threshold\n"
              << "  q / Esc = quit\n\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::resize(frame, frame, cv::Size(W, H));

        if (mirror) cv::flip(frame, frame, 1);

        cv::Mat pre = preprocessForDiff(frame, DW, DH);

        // Compute normalized SAD (per pixel), optionally in ROI
        double denomPixels = (double)(DW * DH);

        if (!maskSmall.empty()) {
            int masked = cv::countNonZero(maskSmall);
            denomPixels = (double)std::max(1, masked);
        }

        double score = sadScore(bgPre, pre, maskSmall) / denomPixels;

        bool detected = (score >= presenceThreshold);

        // Debounce logic
        double sinceChange = std::chrono::duration<double>(Clock::now() - lastStateChange).count();

        if (!alarmState) {
            // currently clear -> can go alarmed
            if (detected && sinceChange >= minClearHoldSec) {
                alarmState = true;
                lastStateChange = Clock::now();
                beepAlarm();
            }
        } else {
            // currently alarmed -> can go clear
            if (!detected && sinceChange >= minAlarmHoldSec) {
                alarmState = false;
                lastStateChange = Clock::now();
            }
        }

        // Visualize diff heatmap (debug)
        cv::Mat diff;
        cv::absdiff(bgPre, pre, diff); // 8UC1
        cv::Mat diffVis;
        cv::resize(diff, diffVis, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
        cv::applyColorMap(diffVis, diffVis, cv::COLORMAP_JET);

        cv::Mat ui = frame.clone();

        // Overlay small diff preview on the right
        cv::Mat preview;
        cv::resize(diffVis, preview, cv::Size(W / 3, H / 3));
        preview.copyTo(ui(cv::Rect(W - preview.cols - 10, 10, preview.cols, preview.rows)));

        // ROI box on main view
        if (roi.area() > 0) {
            cv::rectangle(ui, roi, cv::Scalar(0, 255, 255), 2);
        }

        // Text
        drawTextOutlined(ui, "SAD/pixel: " + std::to_string(score),
                         cv::Point(10, 30), 0.8, 2,
                         cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0));

        drawTextOutlined(ui, "Threshold: " + std::to_string(presenceThreshold),
                         cv::Point(10, 60), 0.8, 2,
                         cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0));

        if (alarmState) {
            // red tint overlay
            cv::Mat tint(ui.size(), ui.type(), cv::Scalar(0, 0, 255));
            cv::addWeighted(ui, 0.65, tint, 0.35, 0.0, ui);

            drawTextOutlined(ui, "ALARM: Presence detected!",
                             cv::Point(10, 110), 1.0, 3,
                             cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0));
        } else {
            drawTextOutlined(ui, "Status: clear",
                             cv::Point(10, 110), 1.0, 2,
                             cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0));
        }

        drawTextOutlined(ui, "b=background  r=ROI  c=clearROI  +/- threshold  q=quit",
                         cv::Point(10, H - 20), 0.6, 2,
                         cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0));

        cv::imshow("Alarm", ui);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;

        if (key == 'b') {
            bgBgr = frame.clone();
            bgPre = preprocessForDiff(bgBgr, DW, DH);
            std::cout << "Background recaptured.\n";
        }

        if (key == '+' || key == '=') {
            presenceThreshold += 0.5;
            std::cout << "Threshold = " << presenceThreshold << "\n";
        } else if (key == '-' || key == '_') {
            presenceThreshold = std::max(0.5, presenceThreshold - 0.5);
            std::cout << "Threshold = " << presenceThreshold << "\n";
        }

        if (key == 'r') {
            // Let user select ROI on the live frame (main resolution)
            cv::Rect selected = cv::selectROI("Alarm", ui, false, false);
            if (selected.area() > 0) {
                roi = selected;

                // create mask in DWxDH: mask = 255 inside ROI, else 0
                // Convert ROI coords from WxH to DWxDH
                cv::Rect roiSmall(
                    (int)std::lround((double)roi.x * DW / W),
                    (int)std::lround((double)roi.y * DH / H),
                    (int)std::lround((double)roi.width * DW / W),
                    (int)std::lround((double)roi.height * DH / H)
                );
                roiSmall &= cv::Rect(0, 0, DW, DH);

                maskSmall = cv::Mat::zeros(DH, DW, CV_8UC1);
                cv::rectangle(maskSmall, roiSmall, cv::Scalar(255), -1);
                std::cout << "ROI set.\n";
            }
        }

        if (key == 'c') {
            roi = cv::Rect();
            maskSmall.release();
            std::cout << "ROI cleared (using full frame).\n";
        }
    }

    return 0;
}
