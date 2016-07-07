/*
 * MonoVisualOdometry.h
 *
 *  Created on: Jun 19, 2016
 *      Author: francisco
 */

#ifndef MONOVISUALODOMETRY_H_
#define MONOVISUALODOMETRY_H_
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

class MonoVisualOdometry {
private :
	Mat cur_frame,cur_frame_c, prev_frame,prev_frame_c, cur_frame_kp, prev_frame_kp;
	vector<KeyPoint> keypoints_1, keypoints_2, good_keypoints_1, good_keypoints_2;
	vector<vector<DMatch> > matches;
	vector<DMatch> good_matches;
	vector<Point2f> point1, point2;
	Mat descriptors_1, descriptors_2,  img_matches;
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<flann::IndexParams> indexParams;
	Ptr<flann::SearchParams> searchParams;
	Ptr<DescriptorMatcher> matcher;
	// relative scale
	double scale;
	//double f = (double)(8.941981e+02 + 8.927151e+02)/2;
	//Point2f pp((float)6.601406e+02, (float)2.611004e+02);
	double f ; // focal length in pixels as in K intrinsic matrix
	Point2f pp; //principal point in pixel
	//global rotation and translation
	Mat R_, t_,Rprev_,tprev_;
	//local rotation and transalation from prev to cur
	Mat E, R, t;
public:
	MonoVisualOdometry(Mat &pcur_frame_c){
		pcur_frame_c.copyTo(cur_frame_c);
		cvtColor(cur_frame_c, cur_frame, CV_BGR2GRAY);

		detector = ORB::create();
		extractor = ORB::create();
		indexParams = makePtr<flann::LshIndexParams> (6, 12, 1);
		searchParams = makePtr<flann::SearchParams>(50);
		matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

		detector->detect(cur_frame, keypoints_2);
		extractor->compute(cur_frame, keypoints_2, descriptors_2);

		R_ = (Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
		t_ = (Mat_<double>(3, 1) << 0., 0., 0.);

        // relative scale
        scale = 1.0;

		//f = (double)(8.941981e+02 + 8.927151e+02)/2;
		//pp((float)6.601406e+02, (float)2.611004e+02);
		f = (double)(7.188560000000e+02 + 7.188560000000e+02)/2;
		pp=Point2f((float)6.071928000000e+02, (float)1.852157000000e+02);
	}
	virtual ~MonoVisualOdometry(){}
	// function performs ratiotest
	// to determine the best keypoint matches
	// between consecutive poses
	void ratioTest(vector<vector<DMatch> > &matches, vector<DMatch> &good_matches) {
		for (vector<vector<DMatch> >::iterator it = matches.begin(); it!=matches.end(); it++) {
			if (it->size()>1 ) {
				if ((*it)[0].distance/(*it)[1].distance > 0.6f) {
					it->clear();
				}
			} else {
				it->clear();
			}
			if (!it->empty()) good_matches.push_back((*it)[0]);
		}
	}
	void step(Mat& pcurFrame_c){
		cur_frame.copyTo(prev_frame);
		cur_frame_c.copyTo(prev_frame_c);
		pcurFrame_c.copyTo(cur_frame_c);
		cvtColor(cur_frame_c, cur_frame, CV_BGR2GRAY);
		keypoints_1 = keypoints_2;
		descriptors_2.copyTo(descriptors_1);
		point1 = point2;

		detector->detect(cur_frame, keypoints_2);
		extractor->compute(cur_frame, keypoints_2, descriptors_2);
		matches.clear();
		good_matches.clear();

		try {
			matcher->knnMatch(descriptors_1, descriptors_2, matches, 2);
			ratioTest(matches, good_matches);
		} catch(Exception &e) {
			//cerr << "knnMatch error"<<endl;;
		}

		// TODO track features using Lucas Kanade
		// If no. of features falls below threshold
		// then recompute features and use knnMatch
		// to select good features
		// Repeat.


		// Retrieve 2D points from good_matches
		// Compute Essential Matrix, R & T
		good_keypoints_1.clear();
		good_keypoints_2.clear();
		point1.clear();
		point2.clear();
		for ( size_t m = 0; m < good_matches.size(); m++) {
			int i1 = good_matches[m].queryIdx;
			int i2 = good_matches[m].trainIdx;
			CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints_1.size()));
            CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints_2.size()));
            good_keypoints_1.push_back(keypoints_1[i1]);
            good_keypoints_2.push_back(keypoints_2[i2]);
		}
		KeyPoint::convert(good_keypoints_1, point1, vector<int>());
		KeyPoint::convert(good_keypoints_2, point2, vector<int>());


		if (point1.size() >5 && point2.size() > 5) {
			E = findEssentialMat(point2, point1, f, pp, RANSAC, 0.999, 1.0);
			recoverPose(E, point2, point1, R, t, f, pp);
			R_.copyTo(Rprev_);
			t_.copyTo(tprev_);
			t_ = t_ + (R_*(scale*t));
  			R_ = R*R_;
		}
	}
	void drawMatches(){
		cv::drawMatches( prev_frame_c, keypoints_1, cur_frame_c, keypoints_2,
               	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  		resize(img_matches, img_matches, Size(), 0.5, 0.5);
  		imshow("matches", img_matches);

	}

	const Mat& getCurFrameC() const {
		return cur_frame_c;
	}

	const Mat& getCurFrameKp() const {
		return cur_frame_kp;
	}

	const Mat& getDescriptors1() const {
		return descriptors_1;
	}

	const Mat& getDescriptors2() const {
		return descriptors_2;
	}

	const Ptr<FeatureDetector>& getDetector() const {
		return detector;
	}

	void setDetector(const Ptr<FeatureDetector>& detector) {
		this->detector = detector;
	}

	const Ptr<DescriptorExtractor>& getExtractor() const {
		return extractor;
	}

	void setExtractor(const Ptr<DescriptorExtractor>& extractor) {
		this->extractor = extractor;
	}

	double getF() const {
		return f;
	}

	void setF(double f) {
		this->f = f;
	}

	const vector<KeyPoint>& getGoodKeypoints1() const {
		return good_keypoints_1;
	}

	const vector<KeyPoint>& getGoodKeypoints2() const {
		return good_keypoints_2;
	}

	const vector<DMatch>& getGoodMatches() const {
		return good_matches;
	}

	const Mat& getImgMatches() const {
		return img_matches;
	}

	const vector<KeyPoint>& getKeypoints1() const {
		return keypoints_1;
	}

	const vector<KeyPoint>& getKeypoints2() const {
		return keypoints_2;
	}

	const Ptr<DescriptorMatcher>& getMatcher() const {
		return matcher;
	}

	void setMatcher(const Ptr<DescriptorMatcher>& matcher) {
		this->matcher = matcher;
	}

	const vector<vector<DMatch> >& getMatches() const {
		return matches;
	}

	const vector<Point2f>& getPoint1() const {
		return point1;
	}

	const vector<Point2f>& getPoint2() const {
		return point2;
	}

	const Point2f& getPp() const {
		return pp;
	}

	void setPp(const Point2f& pp) {
		this->pp = pp;
	}

	const Mat& getPrevFrameC() const {
		return prev_frame_c;
	}

	const Mat& getPrevFrameKp() const {
		return prev_frame_kp;
	}

	const Mat& getR() const {
		return R_;
	}

	void setR(const Mat& r) {
		R_ = r;
	}
	const Mat& getRl() const {
		return R;
	}

	const Mat& getRprev() const {
		return Rprev_;
	}

	void setRprev(const Mat& rprev) {
		Rprev_ = rprev;
	}

	double getScale() const {
		return scale;
	}

	void setScale(double scale) {
		this->scale = scale;
	}

	const Mat& getT() const {
		return t_;
	}
	const Mat& getTl() const {
		return t;
	}

	void setT(const Mat& t) {
		t_ = t;
	}

	const Mat& getTprev() const {
		return tprev_;
	}

	void setTprev(const Mat& tprev) {
		tprev_ = tprev;
	}
};

#endif /* MONOVISUALODOMETRY_H_ */
