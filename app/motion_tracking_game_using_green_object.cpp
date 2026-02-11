#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

// ---------- helpers ----------
static inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

static std::string nowTimestamp() {
    std::time_t t = ::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    tm = *std::localtime(&t);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static std::string sanitizeNickname(std::string s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) {
        return c == '\n' || c == '\r' || c == ',' || c == '\t';
    }), s.end());

    auto is_space = [](unsigned char c) { return std::isspace(c); };
    while (!s.empty() && is_space((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && is_space((unsigned char)s.back())) s.pop_back();

    if (s.empty()) s = "player";
    if (s.size() > 20) s.resize(20);
    return s;
}

static std::string resourcesDirPath() {
#ifdef RESOURCES_DIR
    return std::string(RESOURCES_DIR);
#else
    return std::string("resources");
#endif
}

static std::string scoreFilePath() {
    fs::path p = fs::path(resourcesDirPath()) / "scores.csv";
    return p.string();
}

static void ensureResourcesDirExists() {
    std::error_code ec;
    fs::create_directories(fs::path(resourcesDirPath()), ec);
}

struct ScoreEntry {
    std::string name;
    int score = 0;
    std::string time;
};

static void appendScore(const std::string& name, int score) {
    ensureResourcesDirExists();
    std::ofstream out(scoreFilePath(), std::ios::app);
    if (!out) {
        std::cerr << "WARN: Could not write scores file: " << scoreFilePath() << "\n";
        return;
    }
    out << name << "," << score << "," << nowTimestamp() << "\n";
}

static std::vector<ScoreEntry> loadScoresSorted() {
    std::vector<ScoreEntry> v;
    std::ifstream in(scoreFilePath());
    if (!in) return v;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string name, scoreStr, timeStr;

        if (!std::getline(ss, name, ',')) continue;
        if (!std::getline(ss, scoreStr, ',')) continue;
        std::getline(ss, timeStr);

        try {
            int sc = std::stoi(scoreStr);
            v.push_back({name, sc, timeStr});
        } catch (...) {
            // skip malformed
        }
    }

    std::sort(v.begin(), v.end(), [](const ScoreEntry& a, const ScoreEntry& b) {
        return a.score > b.score;
    });
    return v;
}

static void beepPop() {
#ifdef _WIN32
    Beep(1200, 60);
#endif
}

static void drawTextOutlined(cv::Mat& img, const std::string& text, cv::Point org,
                             double scale, int thickness,
                             const cv::Scalar& fg, const cv::Scalar& outline) {
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, outline, thickness + 2, cv::LINE_AA);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv::LINE_AA);
}

// ---------- blob detection ----------
static cv::Mat broadGreenMask(const cv::Mat& bgr) {
    cv::Mat hsv, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // Broad green band
    cv::inRange(hsv, cv::Scalar(35, 50, 40), cv::Scalar(90, 255, 255), mask);

    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k, cv::Point(-1, -1), 1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k, cv::Point(-1, -1), 1);
    return mask;
}

struct BlobInfo {
    cv::Rect bbox;
    int area = 0;
    cv::Point centroid{-1, -1};
    cv::Mat mask; // 0/255 for that blob
};

static bool largestBlob(const cv::Mat& binMask, BlobInfo& out) {
    CV_Assert(binMask.type() == CV_8UC1);
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(binMask, labels, stats, centroids, 8, CV_32S);
    if (n <= 1) return false;

    int best = -1, bestArea = 0;
    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > bestArea) { bestArea = area; best = i; }
    }
    if (best < 0) return false;

    int x = stats.at<int>(best, cv::CC_STAT_LEFT);
    int y = stats.at<int>(best, cv::CC_STAT_TOP);
    int w = stats.at<int>(best, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(best, cv::CC_STAT_HEIGHT);

    out.bbox = cv::Rect(x, y, w, h);
    out.area = bestArea;
    out.centroid = cv::Point(
        (int)std::lround(centroids.at<double>(best, 0)),
        (int)std::lround(centroids.at<double>(best, 1))
    );

    out.mask = cv::Mat::zeros(binMask.size(), CV_8UC1);
    out.mask.setTo(255, labels == best);
    return true;
}

static cv::Mat calibratedHueMask(const cv::Mat& bgr, int hCenter, int hWin, int sMin, int vMin) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    cv::Mat H = ch[0], S = ch[1], V = ch[2];

    int lo = hCenter - hWin;
    int hi = hCenter + hWin;

    cv::Mat hm;
    if (lo < 0) hm = (H >= (lo + 180)) | (H <= hi);
    else if (hi > 179) hm = (H >= lo) | (H <= (hi - 180));
    else hm = (H >= lo) & (H <= hi);

    cv::Mat mask = hm & (S >= sMin) & (V >= vMin);
    mask.convertTo(mask, CV_8U, 255);

    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k, cv::Point(-1, -1), 1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k, cv::Point(-1, -1), 1);
    return mask;
}

// ---------- balloons ----------
static cv::Scalar brightColor(std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(140, 255);
    while (true) {
        int b = dist(rng), g = dist(rng), r = dist(rng);
        if (b + g + r >= 600) return cv::Scalar(b, g, r);
    }
}

struct Balloon {
    float x{}, y{}, vx{}, vy{};
    int r{};
    cv::Scalar color;
};

static Balloon randomBalloon(int W, int H, std::mt19937& rng) {
    std::uniform_int_distribution<int> rdist(10, 26);
    int r = rdist(rng);

    std::uniform_int_distribution<int> xdist(r, W - r);
    std::uniform_int_distribution<int> ydist(r, H - r);

    std::uniform_real_distribution<float> angdist(0.f, 2.f * (float)CV_PI);
    std::uniform_real_distribution<float> spdist(2.0f, 5.0f);

    float ang = angdist(rng);
    float sp = spdist(rng);

    Balloon b;
    b.r = r;
    b.x = (float)xdist(rng);
    b.y = (float)ydist(rng);
    b.vx = std::cos(ang) * sp;
    b.vy = std::sin(ang) * sp;
    b.color = brightColor(rng);
    return b;
}

static bool outOfBounds(const Balloon& b, int W, int H) {
    return (b.x + b.r < 0) || (b.x - b.r > W) || (b.y + b.r < 0) || (b.y - b.r > H);
}

static inline float dist2(float ax, float ay, float bx, float by) {
    float dx = ax - bx, dy = ay - by;
    return dx * dx + dy * dy;
}

// ---------- main ----------
int main() {
    std::cout << "Enter nickname: ";
    std::string nickname;
    std::getline(std::cin, nickname);
    nickname = sanitizeNickname(nickname);

    const int W = 640;
    const int H = 480;
    const bool mirror = true;

    cv::VideoCapture cap(0, cv::CAP_DSHOW); // try CAP_MSMF / CAP_ANY
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera.\n";
        return 1;
    }

    cv::namedWindow("Game", cv::WINDOW_AUTOSIZE);

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    // ----- 5s calibration -----
    const double calibSeconds = 5.0;
    std::vector<int> hueSamples;
    hueSamples.reserve(20000);

    auto tCalib0 = Clock::now();

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::resize(frame, frame, cv::Size(W, H));
        if (mirror) cv::flip(frame, frame, 1);

        cv::Mat m = broadGreenMask(frame);
        BlobInfo blob;

        cv::Mat vis = frame.clone();
        double elapsed = std::chrono::duration<double>(Clock::now() - tCalib0).count();
        double remaining = calibSeconds - elapsed;

        drawTextOutlined(
            vis,
            "CALIBRATING " + std::to_string((int)std::ceil(std::max(0.0, remaining))),
            cv::Point(10, 35),
            1.0, 2,
            cv::Scalar(255, 255, 255),
            cv::Scalar(0, 0, 0)
        );

        if (largestBlob(m, blob) && blob.area > 900) {
            cv::rectangle(vis, blob.bbox, cv::Scalar(0, 255, 0), 2);
            cv::circle(vis, blob.centroid, 8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            cv::Mat hsv;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

            int step = 2;
            for (int y = blob.bbox.y; y < blob.bbox.y + blob.bbox.height; y += step) {
                const uchar* bm = blob.mask.ptr<uchar>(y);
                const cv::Vec3b* hv = hsv.ptr<cv::Vec3b>(y);
                for (int x = blob.bbox.x; x < blob.bbox.x + blob.bbox.width; x += step) {
                    if (bm[x] == 0) continue;
                    int Hh = hv[x][0];
                    int Ss = hv[x][1];
                    int Vv = hv[x][2];
                    if (Ss > 70 && Vv > 60) hueSamples.push_back(Hh);
                }
            }
        }

        cv::imshow("Game", vis);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') return 0;

        if (elapsed >= calibSeconds) break;
    }

    int hCenter = 60;
    int hWin = 14;
    if (!hueSamples.empty()) {
        std::nth_element(hueSamples.begin(), hueSamples.begin() + (hueSamples.size() / 2), hueSamples.end());
        hCenter = hueSamples[hueSamples.size() / 2];
        hWin = 14;
    } else {
        hCenter = 60;
        hWin = 20;
    }

    // ----- game -----
    const double gameSeconds = 60.0;
    auto tGame0 = Clock::now();

    int score = 0;
    std::vector<Balloon> balloons;
    const int maxBalloons = 18;
    const float spawnProb = 0.25f;

    const cv::Scalar cursorColor(0, 255, 0);
    const int cursorR = 8;
    const int brushR = 6;

    cv::Mat paint(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
    bool hasPrev = false;
    cv::Point prevPt;

    bool hasSmooth = false;
    cv::Point2f smoothPt;
    const float alpha = 0.75f;

    cv::Scalar bgColor(230, 230, 230);

    while (true) {
        double elapsed = std::chrono::duration<double>(Clock::now() - tGame0).count();
        double remaining = gameSeconds - elapsed;
        if (remaining <= 0) break;

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::resize(frame, frame, cv::Size(W, H));
        if (mirror) cv::flip(frame, frame, 1);

        if ((int)balloons.size() < maxBalloons && u01(rng) < spawnProb)
            balloons.push_back(randomBalloon(W, H, rng));

        for (auto& b : balloons) { b.x += b.vx; b.y += b.vy; }
        balloons.erase(std::remove_if(balloons.begin(), balloons.end(),
                        [&](const Balloon& b) { return outOfBounds(b, W, H); }),
                       balloons.end());

        cv::Mat mask = calibratedHueMask(frame, hCenter, hWin, 60, 40);
        BlobInfo blob;
        bool hasC = false;
        cv::Point c;

        if (largestBlob(mask, blob) && blob.area > 600) {
            hasC = true;
            c = blob.centroid;
        }

        cv::Mat ui(H, W, CV_8UC3, bgColor);

        if (hasC) {
            int cx = c.x, cy = c.y;

            if (!hasSmooth) { smoothPt = cv::Point2f((float)cx, (float)cy); hasSmooth = true; }
            else {
                smoothPt.x = alpha * smoothPt.x + (1.f - alpha) * cx;
                smoothPt.y = alpha * smoothPt.y + (1.f - alpha) * cy;
            }

            int sx = clampi((int)std::lround(smoothPt.x), 0, W - 1);
            int sy = clampi((int)std::lround(smoothPt.y), 0, H - 1);
            cv::Point spt(sx, sy);

            int hit = -1;
            for (int i = 0; i < (int)balloons.size(); ++i) {
                const auto& b = balloons[i];
                float rr = (float)(b.r + cursorR);
                if (dist2((float)sx, (float)sy, b.x, b.y) <= rr * rr) { hit = i; break; }
            }

            if (hit >= 0) {
                beepPop();
                cv::Scalar poppedColor = balloons[hit].color;
                balloons.erase(balloons.begin() + hit);
                score++;
                bgColor = poppedColor;

                paint.setTo(cv::Scalar(0, 0, 0));
                hasPrev = false;
            } else {
                if (!hasPrev) cv::circle(paint, spt, brushR, cursorColor, -1, cv::LINE_AA);
                else cv::line(paint, prevPt, spt, cursorColor, brushR * 2, cv::LINE_AA);
                prevPt = spt;
                hasPrev = true;
            }

            cv::circle(ui, spt, cursorR, cursorColor, -1, cv::LINE_AA);
            cv::circle(ui, spt, cursorR, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        } else {
            hasPrev = false;
            hasSmooth = false;
        }

        for (const auto& b : balloons) {
            cv::Point p((int)std::lround(b.x), (int)std::lround(b.y));
            cv::circle(ui, p, b.r, b.color, -1, cv::LINE_AA);
            cv::circle(ui, p, b.r, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }

        drawTextOutlined(ui, "Player: " + nickname, cv::Point(10, 30), 0.8, 2,
                         cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        drawTextOutlined(ui, "Score: " + std::to_string(score), cv::Point(10, 60), 0.9, 2,
                         cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        drawTextOutlined(ui, "Time: " + std::to_string((int)std::max(0.0, remaining)), cv::Point(10, 90), 0.9, 2,
                         cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        cv::Mat out;
        cv::addWeighted(ui, 0.85, paint, 0.95, 0.0, out);

        cv::imshow("Game", out);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
    }

    appendScore(nickname, score);
    auto scores = loadScoresSorted();

    cv::Mat final(H, W, CV_8UC3, bgColor);

    drawTextOutlined(final, "TIME UP!", cv::Point(20, 60), 1.3, 3,
                     cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    drawTextOutlined(final, "SCORE: " + std::to_string(score), cv::Point(20, 110), 1.0, 2,
                     cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    drawTextOutlined(final, "SCOREBOARD (Top 10)", cv::Point(20, 170), 0.9, 2,
                     cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    int y = 210;
    for (int i = 0; i < (int)scores.size() && i < 10; ++i) {
        std::ostringstream line;
        line << (i + 1) << ". " << scores[i].name << " - " << scores[i].score;
        drawTextOutlined(final, line.str(), cv::Point(20, y), 0.8, 2,
                         cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        y += 32;
    }

    drawTextOutlined(final, "Press any key to exit", cv::Point(20, H - 20), 0.7, 2,
                     cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    cv::imshow("Game", final);
    cv::waitKey(0);
    return 0;
}
