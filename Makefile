CFLAGS = -I/usr/local/cuda/include -I/usr/include/hdf5/serial -I/opt/caffe/include -I/opt/caffe/build/include -I/usr/local/include/opencv -I/usr/local/include
LIBS = -L/usr/local/lib -L/opt/caffe/build/lib -lpthread -lcaffe -lglog -lboost_system -lboost_thread -lboost_filesystem -lboost_regex -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_cudawarping -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dpm -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_plot -lopencv_dnn -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_imgproc -lopencv_cudaarithm -lopencv_core -lopencv_cudev

HELPER = goturn/helper
GOTURN = goturn


goturn:
	g++ $(CFLAGS) goturn.cpp -o goturn $(LIBS)
	
test:
	g++ $(CFLAGS) test.cpp -o test $(LIBS)

tracker: main.o helper.o bounding_box.o image_proc.o goturn.o detector.o
	g++ $(CFLAGS) main.o helper.o bounding_box.o image_proc.o goturn.o detector.o -o tracker $(LIBS)

main.o: main.cpp
	g++ -c $(CFLAGS) main.cpp $(LIBS)

helper.o: $(HELPER)/helper.cpp $(HELPER)/helper.h goturn/native/trax.h
	g++ -c $(CFLAGS) $(HELPER)/helper.cpp $(LIBS)

bounding_box.o: $(HELPER)/bounding_box.cpp $(HELPER)/bounding_box.h
	g++ -c $(CFLAGS) $(HELPER)/bounding_box.cpp $(LIBS)

image_proc.o: $(HELPER)/image_proc.cpp $(HELPER)/image_proc.h
	g++ -c $(CFLAGS) $(HELPER)/image_proc.cpp $(LIBS)

goturn.o: $(GOTURN)/goturn.cpp $(GOTURN)/goturn.h
	g++ -c $(CFLAGS) $(GOTURN)/goturn.cpp bounding_box.o image_proc.o $(LIBS)

detector.o: $(GOTURN)/detector.cpp $(GOTURN)/detector.h
	g++ -c $(CFLAGS) $(GOTURN)/detector.cpp $(LIBS)