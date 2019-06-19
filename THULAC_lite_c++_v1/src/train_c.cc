
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <getopt.h>
#include "cb_tagging_learner.h"
using namespace thulac;



int main (int argc,char **argv) {
    int iteration=15;
    Character separator='/';
    int seg_only=false;
    int bigram_threshold=1;
    static struct option long_options[] =
		{
			{"help",     no_argument,       0, 'h'},
            {"separator",     required_argument,       0, 's'},
			{"iteration",  required_argument,       0, 'i'},
            {"bigram_threshold",  required_argument,       0, 'b'},
            
			{0, 0, 0, 0}
		};
    int c;
    int option_index = 0;
    while ( (c = getopt_long(argc, argv, "b:s:i:h",long_options,&option_index)) != -1) {
        switch (c) {
            case 0:
                break;
            case 'b':
                bigram_threshold=atoi(optarg);
                break;
            case 's':
                separator=optarg[0];
                break;
            case 'i' : 
                iteration=atoi(optarg);
                break;
            case 'h' :
            case '?' : 
            default : 

                fprintf(stderr,"");
                return 1;
        }
    }
    if(!(optind+1<argc)){
        fprintf(stderr,"need two auguments for training file and prefix for model files\n");
        return 1;
    }
    
    std::string training_filename(argv[optind]);
    std::string model_filename_prefix(argv[optind+1]);
    
    //std::cout<<separator<<"\n";
    TaggingLearner* tl=new TaggingLearner(iteration,separator,seg_only);
    tl->bigram_threshold=bigram_threshold;
    tl->train(training_filename.c_str(),
            (model_filename_prefix+"_model.bin").c_str(),
            (model_filename_prefix+"_dat.bin").c_str(),
            (model_filename_prefix+"_label.txt").c_str());
    

    delete tl;
}

