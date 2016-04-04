# hapticDServer

HapticsDServer hosts a local http server for image processing and classification of haptics data. The server looks at the POST request's field parameter to determine appropriate response. 

##dotReward
The dotReward modlue relies on openCV to threshold an image and then determine the location of two colored markers. Markers are centered with k-means clustering. Once the location of the markers are determined we threshold in the region of the markers for a range of blue HSV. With an image mask a contour around the blue (zipper edge) is determined. The contour represents the zipper which we then fit a line to. The vertical offset from the zipper contor to the center of previously determined two dots will return the error for each action and be used for reinforcement learning. 

##deepNets
Uses two trained tensorflow graph of a deep neural net to classify the direction of motion and region on contact of a contour in a pinch grasp.

