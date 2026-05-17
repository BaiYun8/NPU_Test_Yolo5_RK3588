#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include "SafeQueue.h"
#include "thread_poll.h"
#include "yolov5s.h"

const int MAX_CONCURRENT_FRAMES = 10;

struct FrameData {
    cv::Mat frame;
    int index;
};

SafeQueue<FrameData> g_readQueue(100);
SafeQueue<FrameData> g_writeQueue(100);
std::atomic<bool> g_readFinish(false);
std::atomic<bool> g_processFinish(false);

void readThreadFunc(cv::VideoCapture &cap)
{
    int idx = 0;
    while (true)
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            std::cerr << "[ReadThread] read failed or EOF.\n";
            break;
        }

        FrameData data{frame.clone(), idx++};
        g_readQueue.enqueue(data);
        std::cout << "read queue size: " << g_readQueue.size() << std::endl;
    }

    g_readFinish = true;
    std::cerr << "[ReadThread] finished.\n";
}

void aggregatorThreadFunc(ThreadPoll &npu_pool)
{
    int nextWriteIndex = 0;
    std::map<int, std::future<ProcessResult>> tasks_inflight;

    while (true)
    {
        FrameData inputFD;
        if (!g_readQueue.empty() && tasks_inflight.size() < MAX_CONCURRENT_FRAMES)
        {
            if (g_readQueue.dequeue(inputFD)) {
                auto fut = npu_pool.submit_task_async(inputFD.index, inputFD.frame);
                tasks_inflight[inputFD.index] = std::move(fut);

                std::cout << "inflight tasks: " << tasks_inflight.size()
                          << ", submitted frame: " << inputFD.index << std::endl;
            }
        }

        auto it = tasks_inflight.find(nextWriteIndex);
        while (it != tasks_inflight.end())
        {
            auto status = it->second.wait_for(std::chrono::milliseconds(0));
            if (status != std::future_status::ready)
            {
                break;
            }

            ProcessResult result;
            try {
                result = it->second.get();
            } catch (const std::exception& e) {
                result.success = false;
                result.error_msg = e.what();
                result.processed_img = cv::Mat();
                std::cerr << "task " << nextWriteIndex << " exception: " << e.what() << std::endl;
            }

            FrameData outputFD;
            outputFD.index = nextWriteIndex;
            if (!result.processed_img.empty()) {
                outputFD.frame = result.processed_img.clone();
            }
            g_writeQueue.enqueue(outputFD);

            tasks_inflight.erase(it);
            std::cout << "processed frame: " << nextWriteIndex
                      << ", remaining tasks: " << tasks_inflight.size() << std::endl;
            nextWriteIndex++;

            it = tasks_inflight.find(nextWriteIndex);
        }

        if (g_readFinish && g_readQueue.empty() && tasks_inflight.empty())
        {
            std::cout << "process thread finished" << std::endl;
            break;
        }

        if (tasks_inflight.size() >= MAX_CONCURRENT_FRAMES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    g_processFinish = true;
    g_writeQueue.stop();
    std::cerr << "[AggregatorThread] finished.\n";
}

void writeThreadFunc(cv::VideoWriter &writer)
{
    while (true)
    {
        if (g_processFinish && g_writeQueue.empty())
        {
            break;
        }

        FrameData outputFD;
        if (!g_writeQueue.dequeue(outputFD)) {
            continue;
        }

        if (!outputFD.frame.empty())
        {
            writer.write(outputFD.frame);
        }
        std::cout << "write queue size: " << g_writeQueue.size() << std::endl;
    }
    std::cerr << "[WriteThread] finished.\n";
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::string inPath  = "../video.mp4";
    std::string outPath = "../output.avi";

    cv::VideoCapture cap(inPath);
    if (!cap.isOpened())
    {
        std::cerr << "Fail to open input video: " << inPath << "\n";
        return -1;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps < 1.0) {
        fps = 25.0;
    }

    int fourcc = cv::VideoWriter::fourcc('H','2','6','4');

    cv::VideoWriter writer(outPath, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened())
    {
        std::cerr << "Fail to create output video: " << outPath << "\n";
        return -1;
    }

#ifdef ENABLE_RKNN_PROFILING
    int npu_thread_count = 1;
#else
    int npu_thread_count = 3;
#endif
    ThreadPoll npu_pool("../model/yolov5s.rknn", npu_thread_count);

    std::thread tRead(readThreadFunc, std::ref(cap));
    std::thread tAggregator(aggregatorThreadFunc, std::ref(npu_pool));
    std::thread tWrite(writeThreadFunc, std::ref(writer));

    tRead.join();
    tAggregator.join();
    tWrite.join();

    g_readQueue.stop();
    g_writeQueue.stop();

    writer.release();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "total processing time: " << elapsed_ms.count() << " ms\n";
    std::cerr << "[Main] All done.\n";
    return 0;
}
