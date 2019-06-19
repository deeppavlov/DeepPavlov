#pragma once
#include <fstream>
#include <cstring>
#include "thulac_base.h"
#include "dat.h"

namespace thulac{

class Punctuation{
private:
    DAT* p_dat;
public:
    Punctuation(const char* filename){
		p_dat = new DAT(filename);
    };

    /*
    void adjust(TaggedSentence& sentence){
        if(!p_dat)return;
        for(int i = 0 ; i < sentence.size(); i ++){
            Word tmp = sentence[i].word;

			if(p_dat->match(tmp) != -1){
				sentence[i].tag = "w";
			}
        }
    };
    */

    void adjust(SegmentedSentence& sentence){
        if(!p_dat)return;
        std::vector<Word> tmpVec;
        for(int i = 0 ; i < sentence.size(); i ++){
            Word tmp = sentence[i];
            if(p_dat->get_info(tmp) >= 0) continue;

            //std::cout<<tmp<<std::endl;

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

            //std::cout<<vecSize<<std::endl;
            for(int k = vecSize - 1; k >= 0; k--){
                tmp = tmpVec[k];
                //std::cout<<k<<":"<<tmp<<std::endl;
                if(p_dat->match(tmp) != -1){
                    //std::cout<<p_dat->match(tmp)<<std::endl;
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
	bool findMulti = false;
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
            findMulti = false;
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
                    sentence[i].tag = "w";
                    findMulti = true;
                    break;
                }
            }

            if(!findMulti){
                if(p_dat->match(sentence[i].word) != -1){
		    sentence[i].tag = "w";
		}
            }

        }
        tmpVec.clear();
    };

    ~Punctuation(){
        if(p_dat) delete p_dat;
    };
};
}//end for thulac
