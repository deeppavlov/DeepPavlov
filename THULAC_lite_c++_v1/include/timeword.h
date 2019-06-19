#pragma once
#include <fstream>
#include <cstring>
#include <set>

namespace thulac{

class TimeWord{
private:
	std::set<int> arabicNumSet;
	//std::set<int> chineseNumSet;
	std::set<int> timeWordSet;
	std::set<int> otherSet;
public:
	TimeWord(){
		for(int i = 48; i < 58; i ++){
			arabicNumSet.insert(i);
		}
		for(int i = 65296; i < 65306; i ++){
			arabicNumSet.insert(i);
		}
		/*
		int chineseNums[] = {12295,19968,20108,19977,22235,20116,20845,19971,20843,20061}; 
		for(int i = 0; i < 10; i ++){
			chineseNumSet.insert(chineseNums[i]);
		}
		*/

		//年:24180 月:26376 日:26085 号:21495 时:26102 点:28857 分:20998 秒:31186
		int timeWord[] = {24180, 26376, 26085, 21495, 26102, 28857, 20998, 31186};
		int len = sizeof(timeWord) / sizeof(int);
		for(int i = 0; i < len; i ++){
			timeWordSet.insert(timeWord[i]);
		}

		for(int i = 65; i < 91; i ++){
			otherSet.insert(i);
		}
		for(int i = 97; i < 123; i ++){
			otherSet.insert(i);
		}
		for(int i = 48; i < 58; i ++){
			otherSet.insert(i);
		}

		int other[] = {65292, 12290, 65311, 65281, 65306, 65307, 8216, 8217, 8220, 8221, 12304, 12305,
					12289, 12298, 12299, 126, 183, 64, 124, 35, 65509, 37, 8230, 38, 42, 65288,
					65289, 8212, 45, 43, 61, 44, 46, 60, 62, 63, 47, 33, 59, 58, 39, 34, 123, 125,
					91, 93, 92, 124, 35, 36, 37, 94, 38, 42, 40, 41, 95, 45, 43, 61, 9700, 9734, 9733};
		len = sizeof(other) / sizeof(int);
		for(int i = 0; i < len; i ++){
			otherSet.insert(other[i]);
		}
	};

	inline bool isArabicNum(Word word){
		bool allArabic = true;
		std::set<int>::iterator it;
		for(int i = 0; i < word.size(); i ++){
			it = arabicNumSet.find(word[i]);
			if(it == arabicNumSet.end()){
				allArabic = false;
				break;
			}
		}
		return allArabic;
	}

	/*
	inline bool isChineseNum(Word word){
		bool allChineseNum = true;
		std::set<int>::iterator it;
		for(int i = 0; i < word.size(); i ++){
			it = chineseNumSet.find(word[i]);
			if(it == chineseNumSet.end()){
				allChineseNum = false;
				break;
			}
		}
		return allChineseNum;
	}
	*/

	inline bool isTimeWord(Word word){
		if(word.size() == 0 || word.size() > 1){
			return false;
		}
		std::set<int>::iterator it = timeWordSet.find(word[0]);
		if(it == timeWordSet.end()){
			return false;
		}else{
			return true;
		}
	}

	inline bool isDoubleWord(Word word, Word postWord){
		if(word.size() != 1 || postWord.size() != 1){
			return false;
		}else{
			int wordInt = word[0];
			int postWordInt = postWord[0];
			if(wordInt == postWordInt){
				std::set<int>::iterator it = otherSet.find(wordInt);
				if(it == otherSet.end()){
					return true;
				}else{
					return false;
				}
			}else{
				return false;
			}
		}
	}

	void adjust(SegmentedSentence& sentence){
		int size = sentence.size();
		Word word;
		int count = 0;
		bool hasTimeWord = false;
		std::set<int>::iterator it;

		for(int i = size - 1; i >= 0; i --){
			word = sentence[i];
			if(isTimeWord(word)){
				hasTimeWord = true;
			}else{
				if(hasTimeWord){
					//if(isArabicNum(word) || isChineseNum(word)){
					if(isArabicNum(word)){
						sentence[i] += sentence[i+1];
						sentence.erase(sentence.begin() + i + 1);
					}
					else {
						hasTimeWord = false;
					}
				}
			}
		}

		size = sentence.size();
		Word postWord;
		for(int i = size - 2; i >= 0; i --){
			word = sentence[i];
			postWord = sentence[i + 1];
			if(isDoubleWord(word, postWord)){
				sentence[i] += sentence[i+1];
				sentence.erase(sentence.begin() + i + 1);
			}
		}

		word.clear();
    };

	bool isHttpWord(Word word){
		if(word.size() < 5){
			return false;
		}else{
			if(word[0] == 'h' && word[1] == 't' && word[2] == 't' && word[3] == 'p' ){
				return true;
			}else{
				return false;
			}
		}
	}

	void adjust(TaggedSentence& sentence){
		int size = sentence.size();
		Word word;
		bool hasTimeWord = false;
		std::set<int>::iterator it;
		
		for(int i = size - 1; i >= 0; i --){
			word = sentence[i].word;
			if(isTimeWord(word)){
				hasTimeWord = true;
			}else{
				if(hasTimeWord){
					//if(isArabicNum(word) || isChineseNum(word)){
					if(isArabicNum(word)){
						sentence[i].word += sentence[i+1].word;
						sentence.erase(sentence.begin() + i + 1);
						sentence[i].tag = "t";
					}
					else {
						hasTimeWord = false;
					}
				}
			}
		}
		

		size = sentence.size();
		for(int i = 0; i < size; i ++){
			word = sentence[i].word;
			if(isHttpWord(word)){
				sentence[i].tag = "x";
			}
		}

		size = sentence.size();
		Word preWord;
		for(int i = 1; i < size; i ++){
			preWord = sentence[i-1].word;
			word = sentence[i].word;
			if(preWord.size() == 1 && preWord[0] == 64){
				if((word.size() != 1) || (word[0] != 64)){
					sentence[i].tag = "np";
				}
			}
		}

		word.clear();
    };

    void adjustDouble(TaggedSentence& sentence){
		int size = sentence.size();
		Word word;
		bool hasTimeWord = false;
		std::set<int>::iterator it;

		for(int i = size - 1; i >= 0; i --){
			word = sentence[i].word;
			if(isTimeWord(word)){
				hasTimeWord = true;
			}else{
				if(hasTimeWord){
					//if(isArabicNum(word) || isChineseNum(word)){
					if(isArabicNum(word)){
						sentence[i].word += sentence[i+1].word;
						sentence.erase(sentence.begin() + i + 1);
						sentence[i].tag = "t";
					}
					else {
						hasTimeWord = false;
					}
				}
			}
		}

		size = sentence.size();
		Word postWord;
		for(int i = size - 2; i >= 0; i --){
			word = sentence[i].word;
			postWord = sentence[i + 1].word;
			if(isDoubleWord(word, postWord)){
				sentence[i].word += sentence[i+1].word;
				sentence.erase(sentence.begin() + i + 1);
			}
		}

		size = sentence.size();
		for(int i = 0; i < size; i ++){
			word = sentence[i].word;
			if(isHttpWord(word)){
				sentence[i].tag = "x";
			}
		}

		size = sentence.size();
		Word preWord;
		for(int i = 1; i < size; i ++){
			preWord = sentence[i-1].word;
			word = sentence[i].word;
			if(preWord.size() == 1 && preWord[0] == 64){
				if((word.size() != 1) || (word[0] != 64)){
					sentence[i].tag = "np";
				}
			}
		}

		word.clear();
    };

    ~TimeWord(){
		arabicNumSet.clear();
		//chineseNumSet.clear();
		timeWordSet.clear();
    };
};
}//end for thulac
