# Calibrator

In order to run a sample, 
1. Extract the given data files to "/data/kinexon/calibration_challenge/"
2. Run 
```
 python3 calibration.py
```
Refer to Algorithm_overview.png for an overview.  The folder results has some output images, which are generated.



TODO further:
1. Refactor code
2. The last part of calibrating the camera2 with respect to camera1 is not implemented fully yet.  This requires more time- and as the suggestion was to only develop an outline this should be done further.
3. Improve calibration with more views and data points
4. I did not go deep into some deep learning based calibration methods as this would be more complicated.  In longer term, a literature survey into this would be beneficial.


