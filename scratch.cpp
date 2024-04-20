#include <opencv2/opencv.hpp>

void captureLiveFaces() {
    int counter = 0;
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Failed to open camera" << std::endl;
        return;
    }

    // Set video width and height
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    while (true) {
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture frame" << std::endl;
            break;
        }

        // Mirror the frame
        cv::flip(frame, frame, 1);

        // Display the frame
        cv::imshow("Live Feed", frame);

        // Check for keypress to break the loop
        if (cv::waitKey(1) == 'l') {
            break;
        }
    }

    // Release the camera and destroy OpenCV windows
    cap.release();
    cv::destroyAllWindows();
}

int main() {
    captureLiveFaces();
    return 0;
}
