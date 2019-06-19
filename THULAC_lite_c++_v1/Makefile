dst_dir=.
include_dir=include
src_dir=src
bin_dir=.
test_dir=test
thulac=g++ -O3 -march=native -I $(include_dir)

# all: $(bin_dir)/thulac_test $(bin_dir)/train_c $(bin_dir)/thulac
all: $(bin_dir)/thulac $(bin_dir)/train_c $(bin_dir)/thulac_test

$(bin_dir)/thulac: $(src_dir)/thulac.cc $(include_dir)/*.h
	$(thulac) $(src_dir)/thulac.cc -o $(bin_dir)/thulac

$(bin_dir)/train_c: $(src_dir)/train_c.cc $(include_dir)/*.h
	$(thulac) -o $(bin_dir)/train_c $(src_dir)/train_c.cc

$(bin_dir)/thulac_test: $(test_dir)/test_case.cpp $(include_dir)/*.h
	$(thulac) -o $(bin_dir)/thulac_test $(test_dir)/test_case.cpp

# $(bin_dir)/thulac_test: $(src_dir)/thulac_test.cc $(include_dir)/*.h
# 	$(thulac) -o $(bin_dir)/thulac_test $(src_dir)/thulac_test.cc	

clean:
	rm -f $(bin_dir)/thulac 
	rm -f $(bin_dir)/train_c 
	rm -f $(bin_dir)/thulac_test

pack:
	tar -czvf THULAC_lite_c++_v1.tar.gz src Makefile doc README.md
