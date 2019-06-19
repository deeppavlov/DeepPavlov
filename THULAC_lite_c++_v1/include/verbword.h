#pragma once
#include <fstream>
#include <cstring>
#include "dat.h"

namespace thulac{

class VerbWord{
private:
    DAT* vM_dat;
	DAT* vD_dat;
	std::string tag_v;
public:
    VerbWord(const char* filename, const char* filename2){
		vM_dat = new DAT(filename);
		vD_dat = new DAT(filename2);
		tag_v = "v";
    };


    void adjust(TaggedSentence& sentence){
		if((!vM_dat)||(!vD_dat))return;
		for(int i=0;i<sentence.size()-1;i++){
			if((sentence[i].tag==tag_v)&&(sentence[i+1].tag==tag_v)){
				if(vM_dat->match(sentence[i].word)!=-1){
					sentence[i].tag="vm";
				}else if(vD_dat->match(sentence[i+1].word)!=-1){
					sentence[i+1].tag="vd";
				}
			}
		}
    };

    ~VerbWord(){
        if(vM_dat) delete vM_dat;
		if(vD_dat) delete vD_dat;
    };
};
}//end for thulac
