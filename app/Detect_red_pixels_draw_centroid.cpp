#include <opencv2/opencv.hpp>
#include <iostream>

static cv::Point2f centroidFromContour(const std::vector<cv::Point>& c) {
    cv::Moments m = cv::moments(c);
    if (std::abs(m.m00) < 1e-6) return {-1.f, -1.f};
    return cv::Point2f(static_cast<float>(m.m10 / m.m00),
                       static_cast<float>(m.m01 / m.m00));
}

int main() {
    cv::VideoCapture cap(0, cv::CAP_DSHOW); // try CAP_ANY or CAP_MSMF if needed
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera.\n";
        return 1;
    }

    cv::Mat frame, hsv, mask1, mask2, mask, cleaned;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert to HSV (better for color thresholding)
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // Red wraps around HSV hue (near 0 and near 180)
        // These ranges are a good starting point; you may tune S/V thresholds for lighting.
        cv::inRange(hsv, cv::Scalar(0, 120, 70),  cv::Scalar(10, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255), mask2);
        mask = mask1 | mask2;

        // Noise cleanup
        cv::GaussianBlur(mask, mask, cv::Size(7, 7), 0);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(mask, cleaned, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Pick the largest red region by area
        int bestIdx = -1;
        double bestArea = 0.0;
        for (int i = 0; i < (int)contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > bestArea) {
                bestArea = area;
                bestIdx = i;
            }
        }

        // Draw result
        cv::Mat out = frame.clone();

        // A minimum area filter helps ignore tiny red noise
        const double minArea = 300.0;

        if (bestIdx >= 0 && bestArea >= minArea) {
            cv::drawContours(out, contours, bestIdx, cv::Scalar(0, 255, 255), 2);

            cv::Point2f c = centroidFromContour(contours[bestIdx]);
            if (c.x >= 0 && c.y >= 0) {
                cv::circle(out, c, 8, cv::Scalar(0, 255, 0), -1); // green centroid dot
                cv::putText(out,
                            "centroid: (" + std::to_string((int)c.x) + "," + std::to_string((int)c.y) + ")",
                            cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
                cv::putText(out,
                            "centroid: (" + std::to_string((int)c.x) + "," + std::to_string((int)c.y) + ")",
                            cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        } else {
            cv::putText(out, "No red target", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(out, "No red target", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        cv::imshow("Camera", out);
        cv::imshow("Red Mask", cleaned);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
    }

    return 0;
}
