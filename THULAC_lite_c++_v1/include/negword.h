#pragma once
#include <fstream>
#include <cstring>
#include "dat.h"

namespace thulac{

class NegWord{
private:
    DAT* neg_dat;
public:
    NegWord(const char* filename){
		neg_dat = new DAT(filename);
    };

    void adjust(SegmentedSentence& sentence){
		if((!neg_dat))return;
		for(int i=sentence.size()-1;i>=0;i--){
			if(neg_dat->match(sentence[i]) != -1){
		    	Word tmpWord;
				tmpWord.push_back(sentence[i][1]);
				sentence.insert(sentence.begin() + i + 1, tmpWord);
				int tmpInt = sentence[i][0];
				sentence[i].clear();
				sentence[i].push_back(tmpInt);
			}
		}
    };

    void adjust(TaggedSentence& sentence){
		if((!neg_dat))return;	
		for(int i=sentence.size()-1;i>=0;i--){
			if(neg_dat->match(sentence[i].word) != -1){
		    	WordWithTag tmpWord(sentence[i].separator);
				tmpWord.word.clear();
				tmpWord.word.push_back(sentence[i].word[1]);
				tmpWord.tag = "v";
				sentence.insert(sentence.begin() + i + 1, tmpWord);
				int tmpInt = sentence[i].word[0];
				sentence[i].word.clear();
				sentence[i].word.push_back(tmpInt);
				sentence[i].tag = "d";
			}
		}
    };

    ~NegWord(){
        if(neg_dat) delete neg_dat;
    };
};
}//end for thulac
