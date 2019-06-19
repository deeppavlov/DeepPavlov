#include "preprocess.h"
#include "thulac_base.h"
#include "cb_tagging_decoder.h"
#include "postprocess.h"
#include "timeword.h"
#include "negword.h"
#include "punctuation.h"
#include "filter.h"
using namespace thulac;
using std::cout;
using std::endl;
void showhelp(){
    std::cerr<<"Command line usage:"<<std::endl;
    std::cerr<<"./thulac [-t2s] [-seg_only] [-filter] [-deli delimeter] [-user userword.txt] [-model_dir dir]"<<std::endl;
    std::cerr<<"or"<<std::endl;
    std::cerr<<"./thulac [-t2s] [-seg_only] [-filter] [-deli delimeter] [-user userword.txt] <inputfile >outputfile"<<std::endl;
    std::cerr<<"\t-t2s\t\t\ttransfer traditional Chinese text to Simplifed Chinese text"<<std::endl;
    std::cerr<<"\t-seg_only\t\tsegment text without Part-of-Speech"<<std::endl;
    std::cerr<<"\t-filter\t\t\tuse filter to remove the words that have no much sense, like \"could\""<<std::endl;
    std::cerr<<"\t-deli delimeter\t\tagsign delimeter between words and POS tags. Default is _"<<std::endl;
    std::cerr<<"\t-user userword.txt\tUse the words in the userword.txt as a dictionary and the words will labled as \"uw\""<<std::endl;
    std::cerr<<"\t-model_dir dir\t\tdir is the directory that containts all the model file. Default is \"models/\""<<std::endl;
}

int main (int argc,char **argv) {

    char* user_specified_dict_name=NULL;
    char* model_path_char = NULL;


    Character separator = '_';

    bool useT2S = false;
    bool seg_only = false;
    bool useFilter = false;
    bool use_second = false;
    int max_length = 50000;

    int c = 1;
    while(c < argc){
        std::string arg = argv[c];
        if(arg == "-t2s"){
            useT2S = true;
        }else if(arg == "-user"){
            user_specified_dict_name = argv[++c];
        }else if(arg == "-deli"){
            separator = argv[++c][0];
        }else if(arg == "-seg_only"){
            seg_only = true;
        }else if(arg == "-filter"){
            useFilter = true;
        }else if(arg == "-model_dir"){
            model_path_char = argv[++c];
        }else{
            showhelp();
            return 1;
        }
        c ++;
    }

    /*
    while ( (c = getopt(argc, argv, "d:s:tgh")) != -1) {
        switch (c) {
            case 'd' :
                user_specified_dict_name=optarg;
                break;
            case 's':
                separator=optarg[0];
                break;
            case 't':
                useT2S = true;
                break;
            case 'g':
                seg_only = true;
                break;
            case 'h' :
            case '?' : 
            default : 
                showhelp();
                return 1;
        }
    }
    */
    std::string prefix;
    if(model_path_char != NULL){
        prefix = model_path_char;
        if(*prefix.rbegin() != '/'){
            prefix += "/";
        }
    }else{
        prefix = "models/";
    }

    TaggingDecoder* cws_decoder=new TaggingDecoder();

    if(seg_only){
        cws_decoder->threshold=0;
    }else{
        cws_decoder->threshold=15000;
    }
    permm::Model* cws_model = new permm::Model((prefix+"cws_model.bin").c_str());
    DAT* cws_dat = new DAT((prefix+"cws_dat.bin").c_str());
    char** cws_label_info = new char*[cws_model->l_size];
    int** cws_pocs_to_tags = new int*[16];

    get_label_info((prefix+"cws_label.txt").c_str(), cws_label_info, cws_pocs_to_tags);
    cws_decoder->init(cws_model, cws_dat, cws_label_info, cws_pocs_to_tags);
    cws_decoder->set_label_trans();

    TaggingDecoder* tagging_decoder = NULL;
    permm::Model* tagging_model = NULL;
    DAT* tagging_dat = NULL;
    char** tagging_label_info = NULL;
    int** tagging_pocs_to_tags = NULL;
    if(!seg_only){
        tagging_decoder = new TaggingDecoder();
        tagging_decoder->separator = separator;
        if(use_second){
            tagging_decoder->threshold = 10000;
        }else{
            tagging_decoder->threshold = 0;
        }
        tagging_model = new permm::Model((prefix+"model_c_model.bin").c_str());
        tagging_dat = new DAT((prefix+"model_c_dat.bin").c_str());
        tagging_label_info = new char*[tagging_model->l_size];
        tagging_pocs_to_tags = new int*[16];
    
        get_label_info((prefix+"model_c_label.txt").c_str(), tagging_label_info, tagging_pocs_to_tags);
        tagging_decoder->init(tagging_model, tagging_dat, tagging_label_info, tagging_pocs_to_tags);
        tagging_decoder->set_label_trans();
    }
 
    
    //printf("%d\n",access("thulac.cc",0));
   
    POCGraph poc_cands;
    int rtn=1;
    thulac::RawSentence raw;
    thulac::RawSentence oiraw;
    thulac::RawSentence traw;
    thulac::SegmentedSentence segged;
    thulac::TaggedSentence tagged;
    Preprocesser* preprocesser = new Preprocesser();
    preprocesser->setT2SMap((prefix+"t2s.dat").c_str());
    Postprocesser* ns_dict = new Postprocesser((prefix+"ns.dat").c_str(), "ns", false);
    Postprocesser* idiom_dict = new Postprocesser((prefix+"idiom.dat").c_str(), "i", false);
    Postprocesser* user_dict = NULL;
    if(user_specified_dict_name){
        user_dict = new Postprocesser(user_specified_dict_name, "uw", true);
    }

    Punctuation* punctuation = new Punctuation((prefix+"singlepun.dat").c_str());

    NegWord* negword = new NegWord((prefix+"neg.dat").c_str());
    TimeWord* timeword = new TimeWord();

    Filter* filter = NULL;
    if(useFilter){
        filter = new Filter((prefix+"xu.dat").c_str(), (prefix+"time.dat").c_str());
    }

    clock_t start = clock();

   // tagging_decoder->cs();
    std::vector<thulac::RawSentence> vec;
    while(1){
        rtn=thulac::get_raw(oiraw);//读入生句子
        if(oiraw.size() > max_length) {
            thulac::cut_raw(oiraw, vec, max_length);    
        }
        else {
            vec.clear();
            vec.push_back(oiraw);
        }
        for(int vec_num = 0; vec_num < vec.size(); vec_num++) {
            if(useT2S) {
                preprocesser->clean(vec[vec_num],traw,poc_cands);
                preprocesser->T2S(traw, raw);
            }
            else {
                preprocesser -> clean(vec[vec_num],raw,poc_cands);
            }
            if(raw.size()){
            
                if(!seg_only) {
                    tagging_decoder->segment(raw,poc_cands,tagged);
                
                //后处理
                    ns_dict->adjust(tagged);
                    idiom_dict->adjust(tagged);
                    punctuation->adjust(tagged);
                    timeword->adjustDouble(tagged);
                    negword->adjust(tagged);
                    if(user_dict){
                        user_dict->adjust(tagged);
                    }
                    if(useFilter){
                        filter->adjust(tagged);
                    }

                    if(vec_num != 0) std::cout << " ";//  输出
                    std::cout<<tagged;//输出
                }
                else {

                    cws_decoder->segment(raw, poc_cands, tagged);
                    cws_decoder->get_seg_result(segged);
                    ns_dict->adjust(segged);
                    idiom_dict->adjust(segged);
                    punctuation->adjust(segged);
                    timeword->adjust(segged);
                    negword->adjust(segged);
                    if(user_dict){
                        user_dict->adjust(segged);
                    }
                    if(useFilter){
                        filter->adjust(segged);
                    }
                    if(vec_num != 0) std::cout << " ";
                    for(int j = 0; j < segged.size(); j++){
                        if(j!=0) std::cout<<" ";
                        std::cout<<segged[j];
                    }

                }
            }
        }
        if(rtn==-1) break;//如果到了文件末尾，退出
        putchar(rtn);//否则打印结尾符号
        std::cout.flush();
        //并继续
    }

    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    std::cerr<<duration<<" seconds"<<std::endl;
    delete tagging_decoder;


    delete preprocesser;
    delete ns_dict;
    delete idiom_dict;
    if(user_dict != NULL){
        delete user_dict;
    }

    delete negword;
    delete timeword;
    delete punctuation;
    if(useFilter){
        delete filter;
    }
    return 0;
}


