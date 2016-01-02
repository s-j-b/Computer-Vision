#3D Scanner

Names: Simon J. Bloch and Gibson Cook

Class: Computer Vision - Spring 2015 - Swarthmore College

Start Date: 5/1/2015

End Date: 5/15/2015

###Welcome to our 3-D Point Cloud Scanner!

####In this file, you will find:

A. Instructions for running our software

B. A description for all relevant contents of this directory.

##A. Instructions

    1. Create a projector-camera setup as described in our lab write-up. You must have a significant horizontal translation between the two. 

    2. Connect the setup to your computer.

    3. Prepare your calibration board with both a printed chessboard and room for a projected chessboard. 

    4. Run getpix.py:

        a. Take calibration pictures of your two-chessboard setup by pressing enter

       	b. Make sure you slightly change the plane of the board to get substantially diverse data. 

	    c. Keep your printed coordinates in the upper right hand corner, to keep a consistent coordinate space.

    5. justPaper and justProj should now have identical images in them. Mask out the relevant chessboard in each image to make the directory names honest.

    6. Run textpix.py. It will tell you which images to delete. Delete this images from BOTH FOLDERS

    7. Run testbox.py, and move your object-turntable setup so it sits within the viewable box.

    8. Run scan3D.py. Click the black screen and press enter. 

    9. Run PointCloudApp.py with the name of your new point cloud as the only argument.

    10. Enjoy the beauty of 3D. Ignore the unbeauty of noisy data.

##B. File descriptions

    1. Programs:

        a. Main program: 
	
#####scan3D.py - Calls other programs and creates a 3D point cloud

        b. Secondary programs:
	
#####calibrate.py - Calibrates the projector/camera setup using images in justPaper and justProj

#####scanner.py - Runs one scan of the object

#####PointCloudApp.py - Shows you a point cloud

	c. Tertiary programs:

#####getpix.py - gets calibration images

#####textpix.py - tests calibration images

#####textbox.py - tests the framing of your object in the camera’s view

    2. Other:

#####README.txt - dis
	
#####chessboard.png - the projected chessboard image
	
#####justPaper - contains masked images of the paper calibration chessboard

#####justProj - contains masked images of the projected calibration chessboard

#####finalTest.npz - a really great sample point cloud
		
#####twoCups.npz - a flawed sample point cloud

#####extras - a directory containing:

######findrotation.py - unfinished rotation code

######projcalib.py - the model for our calibration program
		
######readcsv.py - a program we used to parse cdv files of point data

		





