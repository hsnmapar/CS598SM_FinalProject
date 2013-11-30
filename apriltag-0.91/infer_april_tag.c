#include "apriltag.h"
#include "image_u8.h"
#include "tag36h11.h"
#include "tag36h10.h"
#include "zarray.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using std::cout;
using std::endl;


// input filepath and result directory
int main(int argc, char** argv) {
   if(argc != 3) {
	  cout<<" Usage: image_name, result_directory" << endl;
	  return -1;
   }

   Mat image;
   image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

   if(!image.data) {
	  cout << "Could not open or find the image" << endl;
	  return -2;
   }

   namedWindow("Display window, CV_WINDOW_AUTOSIZE");
   imshow("Display window", image);

   april_tag_family_t *tf = tag36h11_create();
   april_tag_detector_t *td = april_tag_detector_create(tf);

   unsigned char* im_char = (unsigned char*) image.data;
   if(!image.isContinuous()) {
	  cout << "not continuous!" << endl;
	  return -1;
   }
   image_u8_t *im = image_u8_create_from_rgb3(image.cols, image.rows, im_char, 0);

   zarray_t *detections = april_tag_detector_detect(td, im);
   for (int i = 0; i < zarray_size(detections); i++) {
	  april_tag_detection_t *det;
	  zarray_get(detections, i, &det);

	  printf("detection %3d: id %4d, hamming %d, goodness %f\n", i, det->id, det->hamming, det->goodness);
	  april_tag_detection_destroy(det);
   }
   waitKey(0);
   return 0;

}
