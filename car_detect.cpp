#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <random>

struct Detection {
    int class_id;
    std::string className;
    float confidence;
    cv::Scalar color;
    cv::Rect box;
};

class Inference {
public:
    Inference(const std::string& modelPath, const cv::Size& inputSize, const std::string& classesFile, bool useGPU);

    std::vector<Detection> runInference(const cv::Mat& input);

    void setModelConfidenceThreshold(float confidenceThreshold) {
        modelConfidenceThreshold = confidenceThreshold;
    }

    void setModelNMSThreshold(float nmsThreshold) {
        modelNMSThreshold = nmsThreshold;
    }

private:
    cv::dnn::Net net;
    cv::Size modelShape;
    std::vector<std::string> classes;
    float modelConfidenceThreshold;
    float modelScoreThreshold;
    float modelNMSThreshold;
    bool letterBoxForSquare;

    cv::Mat formatToSquare(const cv::Mat& source);
};

Inference::Inference(const std::string& modelPath, const cv::Size& inputSize, const std::string& classesFile, bool useGPU) {
    net = cv::dnn::readNetFromONNX(modelPath);
    modelShape = inputSize;
    modelConfidenceThreshold = 0.5;
    modelScoreThreshold = 0.5;
    modelNMSThreshold = 0.4;
    letterBoxForSquare = true;

    if (useGPU) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }
}

cv::Mat Inference::formatToSquare(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<Detection> Inference::runInference(const cv::Mat& input) {
    cv::Mat modelInput = input;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput);

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    try {
        net.forward(outputs, net.getUnconnectedOutLayersNames());
    }
    catch (cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        return {};
    }

    if (outputs.empty()) {
        std::cerr << "No outputs from network!" << std::endl;
        return {};
    }

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];
    bool yolov8 = false;

    if (dimensions > rows) {
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];
        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }
    float* data = (float*)outputs[0].data;

    float x_factor = modelInput.cols / modelShape.width;
    float y_factor = modelInput.rows / modelShape.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        if (yolov8) {
            float* classes_scores = data + 4;
            cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > modelScoreThreshold) {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else {
            float confidence = data[4];

            if (confidence >= modelConfidenceThreshold) {
                float* classes_scores = data + 5;
                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > modelScoreThreshold) {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    std::vector<Detection> detections;
    for (unsigned long i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];
        detections.push_back(result);
    }

    return detections;
}

int main() {
    std::string modelPath = "C:/yolo8/license_detect.onnx";
    cv::Size inputSize(640, 640);
    std::string classesFile = "C:/yolo8/classes.txt";
    bool useGPU = true;

    Inference inf(modelPath, inputSize, classesFile, useGPU);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    cv::namedWindow("Detection", cv::WINDOW_AUTOSIZE);
    int confidenceThreshold = 50;
    int nmsThreshold = 40;
    cv::createTrackbar("Confidence Threshold", "Detection", &confidenceThreshold, 100);
    cv::createTrackbar("NMS Threshold", "Detection", &nmsThreshold, 100);

    std::ifstream classesFileInput(classesFile);
    std::vector<std::string> classNames;
    std::string line;
    while (std::getline(classesFileInput, line)) {
        classNames.push_back(line);
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Received empty frame" << std::endl;
            break;
        }

        inf.setModelConfidenceThreshold(confidenceThreshold / 100.0);
        inf.setModelNMSThreshold(nmsThreshold / 100.0);

        std::vector<Detection> detections = inf.runInference(frame);

        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, det.color, 2);
            std::string label = det.className + ": " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame, cv::Rect(cv::Point(det.box.x, det.box.y - labelSize.height),
                cv::Size(labelSize.width, labelSize.height + baseLine)),
                det.color, cv::FILLED);
            cv::putText(frame, label, cv::Point(det.box.x, det.box.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        cv::imshow("Detection", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
