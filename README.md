# Defect_Detection_System
Python-based defect detection system utilizing custom trained neural networks.

Files are for inspecting an image of a product, determining the location of the product to inspect, and manipulation of the image to prepare it to be fed through a pre-trained neural network model for defect detection.

The process begins with Find_Ends.py. This looks at images taken from a camera that is automatically triggered. Since the camera is off at an angle, and a straight-on view is desired, this does some initial image manipulation to warp the perspective of the image to make it easier to work with later. 

Next, the image is taken by Monitor_Ends.py for additional manipulation. This uses a second, pre-trained neural network to determine if the image is okay to use. This NN model uses an anomaly detection method, so a low score (low chance of an anomaly) is desired. If the image passes this check, some measurements are taken for the final step in the process.

Finally, Ends_NN_2Class.py uses the manipulated image so far to do the final defect detection. Using the measurements from the earlier steps, the image is split into 40 sub-images. These sub-images are then fed through the defect detection neural network. Results are then displayed on a monitor with additional python applications.
