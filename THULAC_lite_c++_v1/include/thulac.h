#ifndef _THULAC_LIB_H
#define _THULAC_LIB_H
#include "preprocess.h"
#include "thulac_base.h"
#include "cb_tagging_decoder.h"
#include "postprocess.h"
#include "timeword.h"
#include "negword.h"
#include "punctuation.h"
#include "filter.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
using namespace thulac;

typedef std::vector<std::pair<std::string, std::string> > THULAC_result;


class THULAC {
public:
	int init(const char* model_path = NULL, const char* user_path = NULL, int just_seg = 0, int t2s = 0, int ufilter = 0, char separator = '_');
	void deinit();
	int cut(const std::string&, THULAC_result&);
    std::string toString(const THULAC_result&);
	THULAC() {
		user_specified_dict_name=NULL;
		model_path_char=NULL;
		separator = '_';
		useT2S = false;
		seg_only = false;
		useFilter = false;
		max_length = 50000;
		cws_decoder=NULL;
		cws_model = NULL;
		cws_dat = NULL;
		cws_label_info = NULL;
		cws_pocs_to_tags = NULL;

		tagging_decoder = NULL;
		tagging_model = NULL;
		tagging_dat = NULL;
		tagging_label_info = NULL;
		tagging_pocs_to_tags = NULL;
		preprocesser = NULL;

		ns_dict = NULL;
		idiom_dict = NULL;
		user_dict = NULL;

		punctuation = NULL;

		negword = NULL;
		timeword = NULL;

		filter = NULL;
	}
	~THULAC() {
		deinit();
	}

private:
	char* user_specified_dict_name;
	char* model_path_char;
	Character separator;
	bool useT2S;
	bool seg_only ;
	bool useFilter;
	int max_length;
	TaggingDecoder* cws_decoder;
	permm::Model* cws_model;
	DAT* cws_dat;
	char** cws_label_info;
	int** cws_pocs_to_tags;

	TaggingDecoder* tagging_decoder;
	permm::Model* tagging_model;
	DAT* tagging_dat;
	char** tagging_label_info;
	int** tagging_pocs_to_tags;
	Preprocesser* preprocesser;

	Postprocesser* ns_dict;
	Postprocesser* idiom_dict;
	Postprocesser* user_dict;

	Punctuation* punctuation;

	NegWord* negword;
	TimeWord* timeword;

	Filter* filter;
};

int THULAC::init(const char * model_path, const char* user_path, int just_seg, int t2s, int ufilter, char separator) {
	user_specified_dict_name = const_cast<char *>(user_path);
	model_path_char = const_cast<char *>(model_path);
	this->separator = separator;
	useT2S = bool(t2s);
	seg_only = bool(just_seg);
	useFilter = bool(ufilter);
	std::string prefix;
	if(model_path_char != NULL){
		prefix = model_path_char;
		if(*prefix.rbegin() != '/'){
			prefix += "/";
		}
	}else{
		prefix = "models/";
	}
	if(seg_only) {
		cws_decoder = new TaggingDecoder();
		cws_model = new permm::Model((prefix+"cws_model.bin").c_str());
		cws_dat = new DAT((prefix+"cws_dat.bin").c_str());
		cws_label_info = new char*[cws_model->l_size];
		cws_pocs_to_tags = new int*[16];
		get_label_info((prefix+"cws_label.txt").c_str(), cws_label_info, cws_pocs_to_tags);
		cws_decoder->init(cws_model, cws_dat, cws_label_info, cws_pocs_to_tags);
		cws_decoder->set_label_trans();
	}
	else {
		tagging_decoder = new TaggingDecoder();
		tagging_decoder->separator = separator;
		tagging_model = new permm::Model((prefix+"model_c_model.bin").c_str());
		tagging_dat = new DAT((prefix+"model_c_dat.bin").c_str());
		tagging_label_info = new char*[tagging_model->l_size];
		tagging_pocs_to_tags = new int*[16];
	
		get_label_info((prefix+"model_c_label.txt").c_str(), tagging_label_info, tagging_pocs_to_tags);
		tagging_decoder->init(tagging_model, tagging_dat, tagging_label_info, tagging_pocs_to_tags);
		tagging_decoder->set_label_trans();
	}


	preprocesser = new Preprocesser();
	preprocesser->setT2SMap((prefix+"t2s.dat").c_str());

	ns_dict = new Postprocesser((prefix+"ns.dat").c_str(), "ns", false);
	idiom_dict = new Postprocesser((prefix+"idiom.dat").c_str(), "i", false);

	if(user_specified_dict_name){
		user_dict = new Postprocesser(user_specified_dict_name, "uw", true);
	}

	punctuation = new Punctuation((prefix+"singlepun.dat").c_str());

	negword = new NegWord((prefix+"neg.dat").c_str());
	timeword = new TimeWord();

	filter = NULL;
	if(useFilter){
		filter = new Filter((prefix+"xu.dat").c_str(), (prefix+"time.dat").c_str());
	}
	printf("Model loaded succeed\n");
	return int(true);
}

void THULAC::deinit() {
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
	if(seg_only) {
		delete cws_decoder;
	}
	else {
		delete tagging_decoder;
	}
}

int THULAC::cut(const std::string &in, THULAC_result& result) {

	POCGraph poc_cands;
    int startraw = 0;
    thulac::RawSentence raw;
    thulac::RawSentence oiraw;
    thulac::RawSentence traw;
    thulac::SegmentedSentence segged;
    thulac::TaggedSentence tagged;
    size_t in_left=in.length();
	result.clear();
	std::ostringstream ous;
	std::vector<thulac::RawSentence> vec;
	while(true) {
		startraw = thulac::get_raw(oiraw, in, in_left, startraw);
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
	
				if(seg_only){
					cws_decoder->segment(raw, poc_cands, tagged);
	                cws_decoder->get_seg_result(segged);
	                //后处理
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
					for(int j = 0; j < segged.size(); j++){
						ous.str("");
						ous << segged[j];
						result.push_back(std::make_pair<std::string, std::string>(ous.str(), ""));
					}
				}else{
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
	                for(int j = 0; j < tagged.size(); j++) {
	                	ous.str("");
						ous << tagged[j].word;
                        std::string s = tagged[j].tag;
//                        char * ss = s;
	                	result.push_back(std::make_pair<std::string, std::string>(ous.str(), s.c_str()));
	                }
					
				}
			}
		}
		if(startraw == -1) {
			result.push_back(std::make_pair<std::string, std::string>("\n", ""));
			break;
		}
		else {
			ous.str("");
			ous << in[startraw];
			result.push_back(std::make_pair<std::string, std::string>(ous.str(), ""));
		}
		startraw++;
	}
	
	return 1;
}


std::string THULAC::toString(const THULAC_result& result) {
    std::string output = "";
    for(auto i : result) {
        if(i.first == "\n") continue;
        if(seg_only) output += i.first + i.second + " ";
        else output += i.first + char(separator) + i.second + " ";
    }
    output.erase(output.size() - 1, 1);
    return output;
}


#endif
