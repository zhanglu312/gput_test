#include <iostream>
 
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
 
 
int main(int argc, char**argv) {
    std::cout << "OpenCV version=" << std::hex << CV_VERSION << std::dec << std::endl;



    cv::Mat frame, gray;
    cv::UMat uframe, uFrameGray;
    cv::cuda::GpuMat image_gpu, image_gpu_gray;
    cv::VideoCapture capture("test.mp4");

    bool useNone = (argc == 1);
    std::cout << "Use None=" << useNone << std::endl;

    bool useOpenCL = (argc == 2);
    std::cout << "Use OpenCL=" << useOpenCL << std::endl;
    
std::vector<cv::ocl::PlatformInfo> platforms;
 cv::ocl::getPlatfomsInfo(platforms);
//Access to Platform
const cv::ocl::PlatformInfo* platform = &platforms[0];
//Platform Name
std::cout << "Platform Name : " << platform->name().c_str() << std::endl;
//Access Device within Platform
cv::ocl::Device current_device;
platform->getDevice(current_device, 0);
std::cout<<"Device:"<<current_device.name()<<std::endl;
current_device.set(0);
cv::ocl::setUseOpenCL(useOpenCL);
    
 
    bool useCuda = (argc == 3);
    std::cout << "Use CUDA=" << useCuda << std::endl;
 
    cv::Ptr<cv::CascadeClassifier> cascade = cv::makePtr<cv::CascadeClassifier>("lbpcascade_frontalface_improved.xml");
    cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu = cv::cuda::CascadeClassifier::create("lbpcascade_frontalface_improved.xml");
 
    double time = 0.0;
    int nb = 0;
    if(capture.isOpened()) {
        for(;;) {
            capture >> frame;
            if(frame.empty() || nb >= 1000) {
                break;
            }
	    printf("frame Ind: %d\n", nb);
 
            std::vector<cv::Rect> faces;
            double t = 0.0;
            if(useOpenCL) {
                t = (double) cv::getTickCount();
                frame.copyTo(uframe);
                cv::cvtColor(uframe, uFrameGray, CV_BGR2GRAY);
                cascade->detectMultiScale(uFrameGray, faces);
                t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
            }

	    if (useCuda) 
	    {
                t = (double) cv::getTickCount();
                image_gpu.upload(frame);
                cv::cuda::cvtColor(image_gpu, image_gpu_gray, CV_BGR2GRAY);
                cv::cuda::GpuMat objbuf;
                cascade_gpu->detectMultiScale(image_gpu_gray, objbuf);
                cascade_gpu->convert(objbuf, faces);
                t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
            }
 
	    if (useNone) 
            {
		t = (double) cv::getTickCount();
		cv::cvtColor(frame, gray, CV_BGR2GRAY);
                cascade->detectMultiScale(gray, faces);
                t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
	    }

            time += t;
            nb++;
 
            /*for(std::vector<cv::Rect>::const_iterator it = faces.begin(); it != faces.end(); ++it) {
                cv::rectangle(frame, *it, cv::Scalar(0,0,255));
            }
            std::stringstream ss;
            ss << "FPS=" << (nb / time);
            cv::putText(frame, ss.str(), cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255));
 
            cv::imshow("Frame", frame);
            char c = cv::waitKey(30);
            if(c == 27) {
                break;
            }*/
        }
    }
 
    std::cout << "Mean time=" << (time / nb) << " s" << " ; Mean FPS=" << (nb / time) << " ; nb=" << nb << std::endl;
    system("pause");
    return 0;
}
