#pragma once
#include <fstream>
#include <cstring>
#include <set>
#include "dat.h"

namespace thulac{

class Filter{
private:
    DAT* xu_dat;
	DAT* time_dat;
	std::set<std::string> posSet;
	std::set<int> arabicNumSet;
	std::set<int> chineseNumSet;
public:
    Filter(const char* xuWordFile, const char* timeWordFile){
		xu_dat = new DAT(xuWordFile);
		time_dat = new DAT(timeWordFile);
		std::string POS_RESERVES[] = {"n","np","ns","ni","nz","v","a","id","t","uw"};
		for(int i = 0; i < 10; i ++){
			posSet.insert(POS_RESERVES[i]);
		}
		for(int i = 48; i < 58; i ++){
			arabicNumSet.insert(i);
		}
		for(int i = 65296; i < 65306; i ++){
			arabicNumSet.insert(i);
		}
		int chineseNums[] = {12295,19968,20108,19977,22235,20116,20845,19971,20843,20061}; 
		for(int i = 0; i < 10; i ++){
			chineseNumSet.insert(chineseNums[i]);
		}
    };

    void adjust(SegmentedSentence& sentence){
        if(!xu_dat || !time_dat)return;
		int size = sentence.size();
		Word word;
		int count = 0;
		bool checkArabic = false;
		bool checkChinese = false;
		std::set<int>::iterator it;

		for(int i = size - 1; i >= 0; i --){
			word = sentence[i];
			//if((word.size() < 2) || (xu_dat->match(word) != -1)){
			if(xu_dat->match(word) != -1){
				sentence.erase(sentence.begin() + i);
				continue;
			}
			count = 0;
			checkArabic = false;
			checkChinese = false;
			
			for(int j = 0; j < word.size(); j ++){
				it = arabicNumSet.find(word[j]);
				if(it != arabicNumSet.end()){
					checkArabic = true;
					break;
				}
				it = chineseNumSet.find(word[j]);
				if(it != chineseNumSet.end()){
					count++;
					if(count == 2){
						checkChinese = true;
						break;
					}
				}
			}
			if(checkArabic || checkChinese || (time_dat->match(word) != -1)){
				sentence.erase(sentence.begin() + i);
				continue;
			}
		}

		word.clear();
    };

    void adjust(TaggedSentence& sentence){
		if(!xu_dat || !time_dat)return;
		int size = sentence.size();
		Word word;
		std::string tag;
		int count = 0;
		bool checkArabic = false;
		bool checkChinese = false;
		std::set<int>::iterator it;
		std::set<std::string>::iterator posIt;

		for(int i = size - 1; i >= 0; i --){
			word = sentence[i].word;
			/*
			if(word.size() < 2){
				sentence.erase(sentence.begin() + i);
				continue;
			}
			*/
			tag = sentence[i].tag;
			posIt = posSet.find(tag);
			if(posIt != posSet.end()){
				if(xu_dat->match(word) != -1){				
					sentence.erase(sentence.begin() + i);
					continue;
				}
				if(tag == "t"){	
					count = 0;
					checkArabic = false;
					checkChinese = false;
			
					for(int j = 0; j < word.size(); j ++){
						it = arabicNumSet.find(word[j]);
						if(it != arabicNumSet.end()){
							checkArabic = true;
							break;
						}
						it = chineseNumSet.find(word[j]);
						if(it != chineseNumSet.end()){
							count++;
							if(count == 2){
								checkChinese = true;
								break;
							}
						}
					}
					if(checkArabic || checkChinese || (time_dat->match(word) != -1)){
						sentence.erase(sentence.begin() + i);
						continue;
					}
				}
			}else{
				sentence.erase(sentence.begin() + i);
				continue;
			}

		}

		word.clear();
		tag.clear();
    };

    ~Filter(){
        if(xu_dat) delete xu_dat;
		if(time_dat) delete time_dat;
    };
};
}//end for thulac
