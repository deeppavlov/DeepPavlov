#pragma once
#include <fstream>
#include <cstring>
#include <fstream>
#include "dat.h"

namespace thulac{

class Postprocesser{
private:
    DAT* p_dat;
	std::string tag;
public:
    Postprocesser(const char* filename, std::string tag, bool isTxt){
		this->tag = tag;
		if(isTxt){
			std::ifstream is(filename, std::ifstream::in);
			std::vector<DATMaker::KeyValue> lexicon;
			lexicon.push_back(DATMaker::KeyValue());
			std::string str;
			int id = 0;
			while(std::getline(is,str,'\n'))
            {
				if(str.length()==0)continue;
				if(*str.rbegin() == '\r'){
					str.erase(str.end()-1);
				}
				string_to_raw(str,lexicon.back().key);
				lexicon.back().value=id;

				//init a new element
				lexicon.push_back(DATMaker::KeyValue());
				lexicon.back().key.clear();
				id+=1;
			}
			DATMaker* dm = new DATMaker();
			dm->make_dat(lexicon, 0);
			dm->shrink();
			
			/*
			p_dat = new DAT();	
			p_dat->dat_size = dm->dat_size;
			p_dat->dat = (DAT::Entry*)calloc(sizeof(DAT::Entry), p_dat->dat_size);
			memcpy(p_dat->dat, dm->dat, sizeof(DAT::Entry)*(p_dat->dat_size));
			*/
			
			/*
			dm->save("models/user.dat");
			p_dat = new DAT("models/user.dat");
			*/
			
			p_dat = new DAT(dm->dat_size, dm->dat);
			
			delete dm;
		}else{
			p_dat = new DAT(filename);
		}
    };

    void adjust(SegmentedSentence& sentence){
        if(!p_dat)return;
        std::vector<Word> tmpVec;
        for(int i = 0 ; i < sentence.size(); i ++){
            Word tmp = sentence[i];
            if(p_dat->get_info(tmp) >= 0) continue;

            tmpVec.clear();
            int j;
            for(j = i + 1; j < sentence.size(); j ++){
                tmp += sentence[j];
                if(p_dat->get_info(tmp) >= 0){
                    break;
                }
                tmpVec.push_back(tmp);
            }
            int vecSize = (int)tmpVec.size();
            
            for(int k = vecSize - 1; k >= 0; k--){
                tmp = tmpVec[k];
                if(p_dat->match(tmp) != -1){
                    for(j = i + 1; j < i + k + 2; j ++){
                        sentence[i] += sentence[j];
                    }
                    for(j = i + k + 1; j > i; j--){
                        sentence.erase(sentence.begin() + j);
                    }
                    break;
                }
            }

        }
        tmpVec.clear();
    };

    void adjust(TaggedSentence& sentence){
        if(!p_dat)return;
        std::vector<Word> tmpVec;
        for(int i = 0 ; i < sentence.size(); i ++){
            Word tmp = sentence[i].word;
            if(p_dat->get_info(tmp) >= 0) continue;

            //std::cout<<tmp<<std::endl;

            tmpVec.clear();
            int j;
            for(j = i + 1; j < sentence.size(); j ++){
                tmp += sentence[j].word;
                if(p_dat->get_info(tmp) >= 0){
                    break;
                }
                tmpVec.push_back(tmp);
            }
            int vecSize = (int)tmpVec.size();
            
            //std::cout<<vecSize<<std::endl;

            for(int k = vecSize - 1; k >= 0; k--){
                tmp = tmpVec[k];
                //std::cout<<k<<":"<<tmp<<std::endl;
                if(p_dat->match(tmp) != -1){
                    //std::cout<<p_dat->match(tmp)<<std::endl;
                    for(j = i + 1; j < i + k + 2; j ++){
                        sentence[i].word += sentence[j].word;
                    }
                    for(j = i + k + 1; j > i; j--){
                        sentence.erase(sentence.begin() + j);
                    }
                    sentence[i].tag = tag;
                    break;
                }
            }

        }
        tmpVec.clear();
    };

    void adjustSame(TaggedSentence& sentence){
        if(!p_dat)return;
        std::vector<Word> tmpVec;
        for(int i = 0 ; i < sentence.size(); i ++){
            Word tmp = sentence[i].word;
            if(p_dat->get_info(tmp) >= 0) continue;

            if(p_dat->match(sentence[i].word) != -1){
                sentence[i].tag = tag;
            }

        }
        tmpVec.clear();
    };

    ~Postprocesser(){
        if(p_dat) delete p_dat;
    };
};
}//end for thulac
