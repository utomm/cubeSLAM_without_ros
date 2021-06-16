/*
 * line_detection interface
 * Copyright Shichao Yang,2018, Carnegie Mellon University
 * Email: shichaoy@andrew.cmu.edu
 *
 */

#include <line_lbd/line_descriptor.hpp>

// #include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <ctime>
#include <line_lbd/line_lbd_allclass.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	std::string image_path;
	std::string save_folder = "./line_lbd/data";

	/* get parameters from comand line */
	FileStorage fs2("/home/hu/CLionProjects/cubeSLAM_without_ros/line_lbd/test_config.yaml", FileStorage::READ);
	fs2["image_path"] >> image_path;
	save_folder = (std::string)fs2["save_folder"];
	bool use_LSD_algorithm = (int)fs2["use_LSD_algorithm"] != 0;
	bool save_to_imgs = (int)fs2["save_to_imgs"] != 0;
	bool save_to_txts = (int)fs2["save_to_txts"] != 0;
	fs2.release();

	cv::Mat raw_img = imread(image_path, 1);
	if (raw_img.data == NULL)
	{
		std::cout << "Error, image could not be loaded. Please, check its path \n" << image_path << std::endl;
		return -1;
	}

	int numOfOctave_ = 1;
	float Octave_ratio = 2.0;

	auto* line_lbd_ptr = new line_lbd_detect(numOfOctave_, Octave_ratio);
	line_lbd_ptr->use_LSD = use_LSD_algorithm;
	line_lbd_ptr->line_length_thres = 15;  // remove short edges


	// using my line detector class, could select LSD or edline.
	cv::Mat out_edges;
	std::vector<KeyLine> keylines_raw, keylines_out;
	line_lbd_ptr->detect_raw_lines(raw_img, keylines_raw);
	line_lbd_ptr->filter_lines(keylines_raw, keylines_out);  // remove short lines

	// show image
	if (raw_img.channels() == 1)
		cvtColor(raw_img, raw_img, COLOR_GRAY2BGR);
	cv::Mat raw_img_cp;
	drawKeylines(raw_img, keylines_out, raw_img_cp, cv::Scalar(0, 150, 0), 2); // B G R
	imshow("Line detector", raw_img_cp);
	waitKey();

	if (save_to_imgs)
	{
		std::string img_save_name = save_folder + "saved_edges.jpg";
		cv::imwrite(img_save_name, raw_img_cp);
	}

	if (save_to_txts)
	{
		std::string txt_save_name = save_folder + "saved_edges.txt";
		ofstream resultsFile;
		resultsFile.open(txt_save_name);
		for (auto& j : keylines_out)
		{
			resultsFile << j.startPointX << "\t" << j.startPointY << "\t"
						<< j.endPointX << "\t" << j.endPointY << endl;
		}
		resultsFile.close();
	}

}
