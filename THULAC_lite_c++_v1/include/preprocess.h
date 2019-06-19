#pragma once
#include<vector>
#include<map>
#include<string>
#include<sstream>
#include "thulac_base.h"

namespace thulac{

class Preprocesser{
	std::map<int,int> t2s;
	std::map<int,int> s2t;
	std::set<int> otherSet;		// store all the number, digit and punctuation
	std::set<int> singlePunSet;		// store all the punctuation that need to split
	std::set<int> httpSet;			// store all the number, digit and punctuation that may be a charactor in a url
public:
	Preprocesser(){
		for(int i = 65; i < 91; i ++){
			otherSet.insert(i);
			httpSet.insert(i);
		}
		for(int i = 97; i < 123; i ++){
			otherSet.insert(i);
			httpSet.insert(i);
		}
		for(int i = 48; i < 58; i ++){
			otherSet.insert(i);
			httpSet.insert(i);
		}
		/*
		for(int i = 65296; i < 65306; i ++){
			otherSet.insert(i);
		}
		int chineseNums[] = {12295,19968,20108,19977,22235,20116,20845,19971,20843,20061};
		for(int i = 0; i < 10; i ++){
			otherSet.insert(chineseNums[i]);
		}
		*/

		// other correspond to ，:65292 。:12290 ？:65311 ！:65281 ：:65306 
		//						；:65307 ‘:8216 ’:8217 “:8220 ”:8221 【:12304 】:12305 、
		//						:12289 《:12298 》:12299 ~:126 ·:183 @:64 |:124 #:35 ￥:65509 
		//						%:37 ……:8230 8230 &:38 *:42 （:65288 ）:65289 ——:8212 8212 -:45 
		//						+:43 =:61 ...:46 46 46 。。。:12290 12290 12290 ,:44 .:46 <:60 >:62 
		//						?:63 /:47 ~:126 !:33 @:64 ;:59 ::58 ':39 ":34 {:123 }:125 [:91 ]:93 
		//						\:92 |:124 @:64 #:35 $:36 %:37 ^:94 &:38 *:42 (:40 ):41 _:95 -:45 +:43 =:61 
		//						◤:9700 ☆:9734 ★:9733
		int other[] = {65292, 12290, 65311, 65281, 65306, 65307, 8216, 8217, 8220, 8221, 12304, 12305,
						12289, 12298, 12299, 126, 183, 64, 124, 35, 65509, 37, 8230, 38, 42, 65288,
						65289, 8212, 45, 43, 61, 44, 46, 60, 62, 63, 47, 33, 59, 58, 39, 34, 123, 125,
						91, 93, 92, 124, 35, 36, 37, 94, 38, 42, 40, 41, 95, 45, 43, 61, 9700, 9734, 9733};
		int len = sizeof(other) / sizeof(int);
		for(int i = 0; i < len; i ++){
			otherSet.insert(other[i]);
		}

		// singlePun correspond to (see otherSet)
		int singlePun[] = {65292, 12290, 65311, 65281, 65306, 65307, 8216, 8217, 8220, 8221, 1230, 12304, 
							12305, 12289, 12298, 12299, 64,35, 65288, 65289, 34, 91, 93, 126, 47, 44, 58,
							63, 9700, 9734, 9733, 8230, 39, 33, 42, 43, 62, 40, 41, 59, 61};
		len = sizeof(singlePun) / sizeof(int);
		for(int i = 0 ; i < len; i ++){
			singlePunSet.insert(singlePun[i]);
		}

		char httpChar[] = {'/', '.', ':', '#', '"', '_', '-', '=', '+', '&', '$', ';', '?'};
		len = sizeof(httpChar) / sizeof(char);
		for(int i = 0; i < len; i ++){
			httpSet.insert((int)httpChar[i]);
		}
	}

	inline bool isOther(int c){
		std::set<int>::iterator it = otherSet.find(c);
		if(it != otherSet.end()){
			return true;
		}else{
			return false;
		}
	}
	
	inline bool isSinglePun(int c){
		std::set<int>::iterator it = singlePunSet.find(c);
		if(it != singlePunSet.end()){
			return true;
		}else{
			return false;
		}
	}
	
	inline bool isHttp(int c){
		std::set<int>::iterator it = httpSet.find(c);
		if(it != httpSet.end()){
			return true;
		}else{
			return false;
		}
	}

	void setT2SMap(const char* filename){
		FILE* pFile = fopen(filename, "r+b");
		if(!pFile){
			fprintf(stderr, "[ERROR] traditional Chinese to simplified Chinese : %s  data file not find ",filename);
			return;
		}
		fseek(pFile, 0, SEEK_END);
		int dat_size = ftell(pFile) / (2 * sizeof(int));
		rewind(pFile);
		int* tra = new int[dat_size];
		int* sim = new int[dat_size];
		int rtn = fread(tra, sizeof(int), dat_size, pFile);
		rtn = fread(sim, sizeof(int), dat_size, pFile);
		for(int i = 0; i < dat_size; i ++){
			t2s.insert(std::pair<int,int>(tra[i], sim[i]));
			s2t.insert(std::pair<int,int>(sim[i], tra[i]));
		}
		delete [] tra;
		delete [] sim;
		fclose(pFile);
	}

    int clean(RawSentence& sentence, RawSentence& senClean, POCGraph& graph){
        senClean.clear();
        graph.clear();
        bool hasSpace = false;		//use to check whether the char is a space 
		bool hasOther = false;		//use to check whether isOther(char);
		bool hasSinglePun = false;	//use to check whether isSinglePun(char);
		bool hasHttp = false;		//use to check whether isHttp(char);
		bool hasAt = false;			//use to check whether the char is @
		bool hasTitle = false;		//use to check whether the sentence has 《》
		std::vector<int> httpStartVec;
		int httpStart = -1;
		std::vector<Raw> httpVec;
        int c = -1;
		Raw tmpRaw;
		Raw npRaw;
		int npStart = -1;
		std::vector<int> npStartVec;
		std::vector<Raw> npVec;
		Raw titleRaw;
		int titleStart = -1;
		std::vector<int> titleStartVec;
		std::vector<Raw> titleVec;
		for(int i = 0; i < (int)sentence.size(); i++){
            c = sentence.at(i);
			// if the sentence has space
            if(c == 32 || c == 12288){
				hasOther = false;
                if(hasSpace){
                    continue;
                }else{
                    if(graph.size()){
                        graph.back()&=12;
                    }
                    hasSpace=true;
                    continue;
                }

				// if(hasAt){
				// 	npVec.push_back(npRaw);
				// 	npStartVec.push_back(npStart);
				// 	hasAt = false;
				// }
            }else if(isOther(c)){
				if(hasSpace){
					senClean.push_back(c);
					if(isSinglePun(c)){
						graph.push_back(8);
						hasSinglePun = true;
					}else{
						graph.push_back(9);
						hasSinglePun = false;
					}
					hasSpace = false;
				}else if(hasOther){
					if(isSinglePun(c)){
						if(graph.size()){
							graph.back() &= 12;
						}
						senClean.push_back(c);
						graph.push_back(8);
						hasSinglePun = true;

					}else{
						if(hasSinglePun){
							senClean.push_back(c);
							graph.push_back(9);
						}else{						
							if(!graph.back()) graph.back() = 7;
							senClean.push_back(c);
							graph.push_back(2);
						}
						hasSinglePun = false;
					}
				}else{
					senClean.push_back(c);
					graph.push_back(9);
					if(isSinglePun(c)){
						hasSinglePun = true;
					}else{
						hasSinglePun = false;
					}
				}
				// if(c == 41 || c == 65289){
				// 	if(hasAt){
				// 		npVec.push_back(npRaw);
				// 		npStartVec.push_back(npStart);
				// 		hasAt = false;
				// 	}
				// }
				if(c == 12299){
					if(hasTitle){
						titleVec.push_back(titleRaw);
						titleStartVec.push_back(titleStart);
						hasTitle = false;
					}
				}
				hasOther = true;
			}else{
				if(hasSpace){
					senClean.push_back(c);
					graph.push_back(9);
				}else if(hasOther){
					graph.back() &= 12;
					if(hasSinglePun){
						senClean.push_back(c);
						graph.push_back(9);
						hasSinglePun = false;
					}else{					
						senClean.push_back(c);
						graph.push_back(9);
					}
				}else{
					senClean.push_back(c);
					graph.push_back(15);
				}
				hasSpace = false;
				hasOther = false;
			}

			// if(isHttp(c)){
			// 	if(!hasHttp){
			// 		if(c == 'h'){
			// 			httpStart = graph.size() - 1;
			// 			tmpRaw.clear();
			// 			tmpRaw.push_back(c);
			// 			hasHttp = true;
			// 		}
			// 	}else{
			// 		tmpRaw.push_back(c);
			// 	}
			// }else{
			// 	if(hasHttp){
			// 		httpVec.push_back(tmpRaw);
			// 		httpStartVec.push_back(httpStart);
			// 		hasHttp = false;
			// 	}
			// }

			// if(c == 64){
			// 	if(hasAt){
			// 		npVec.push_back(npRaw);
			// 		npStartVec.push_back(npStart);
			// 		npRaw.clear();
			// 	}
			// 	hasAt = true;
			// 	npStart = graph.size();
			// 	npRaw.clear();
			// }else if(hasAt){
			// 	npRaw.push_back(c);
			// }

			if(c == 12298){
				hasTitle = true;
				titleStart = graph.size();
				titleRaw.clear();
			}else if(hasTitle){
				titleRaw.push_back(c);
			}
		}
		// if(tmpRaw.size() != 0){
		// 	httpVec.push_back(tmpRaw);
		// 	httpStartVec.push_back(httpStart);
		// }
		// if(npRaw.size() != 0){
		// 	npVec.push_back(npRaw);
		// 	npStartVec.push_back(npStart);
		// }

		// std::ostringstream ost;
		// std::string str;
		// for(int i = 0 ; i < httpVec.size(); i ++){
		// 	ost.str("");
		// 	ost<<httpVec[i];
		// 	str = ost.str();
		// 	std::cout << "====\n" << str << std::endl;
		// 	std::size_t found = str.find("http");
		// 	if(found != std::string::npos){
		// 		int start = httpStartVec[i];
		// 		int size = str.size();
				
				
		// 		graph[start] = 1;
		// 		for(int j = start + 1; j < start + size - 1; j ++){
		// 			graph[j] = 2;
		// 		}
		// 		graph[start + size - 1] = 4;
				
		// 	}
		// }

		// for(int i = 0; i < npVec.size(); i ++){
		// 	npRaw = npVec[i];
		// 	if(npRaw.size() > 0 && npRaw.size() < 15){
		// 		int start = npStartVec[i];
		// 		int size = npRaw.size();
				
		// 		graph[start] = 1;
		// 		for(int j = start + 1; j < start + size - 1; j ++){
		// 			graph[j] = 2;
		// 		}
		// 		graph[start + size - 1] = 4;
				
		// 	}
		// }

		for(int i = 0; i < titleVec.size(); i ++){
			titleRaw = titleVec[i];
			if(isPossibleTitle(titleRaw)){
				int start = titleStartVec[i];
				int size = titleRaw.size();
                                if(size == 1) {
					graph[start] = 9;
					continue;
				}
				graph[start] = 1;
				for(int j = start + 1; j < start + size - 1; j ++){
					graph[j] = 2;
				}
				graph[start + size - 1] = 4;
			}
		}
		
		if(graph.size()!=0){
            graph[0]&=9;
            graph.back()&=12;
		    if(!graph[0]) graph[0]=9;
	    	if(!graph.back()) graph.back()=12;
        }
        return 0;
    };

	bool isPossibleTitle(Raw titleRaw){
		if(titleRaw.size() > 10 || titleRaw.size() == 0){
			return false;
		}else{
			for(int i = 0; i < titleRaw.size(); i ++){
				if(isOther(titleRaw[i])){
					return false;
				}
			}
			return true;
		}
	}

	int getT2S(int c){
		std::map<int,int>::iterator it = t2s.find(c);
		if(it != t2s.end()){
			return it->second;
		}else{
			return c;
		}
	}

	int getS2T(int c){
		std::map<int,int>::iterator it = s2t.find(c);
		if(it != s2t.end()){
			return it->second;
		}else{
			return c;
		}
	}
	
	bool containsT(RawSentence& sentence){
		std::map<int,int>::iterator it;
		for(int i = 0; i < sentence.size(); i ++){
			it = t2s.find(sentence[i]);
			if(it != t2s.end()){
				return true;
			}
		}
		return false;
	}

	void T2S(RawSentence& sentence, RawSentence& newSentence){
		newSentence.clear();
		for(int i = 0; i < sentence.size();i ++){
			newSentence.push_back(getT2S(sentence[i]));
		}
	}
	
	void S2T(TaggedSentence& sentence, RawSentence& oriSentence){
		int count = 0;
		for(int i = 0; i < sentence.size();i ++){
			for(int j = 0; j < sentence[i].word.size(); j ++){
				sentence[i].word[j] = oriSentence[count];
				count ++;
			}
		}
	}

	/*
    int cleanAndT2S(RawSentence& sentence, RawSentence& senClean, POCGraph& graph){
        senClean.clear();
        graph.clear();
        bool hasSpace = false;		//use to check whether the char is a space 
		bool hasOther = false;		//use to check whether isOther(char);
		bool hasSinglePun = false;	//use to check whether isSinglePun(char);
		bool hasHttp = false;		//use to check whether isHttp(char);
		std::vector<int> httpStartVec;
		int httpStart = -1;
		std::vector<Raw> httpVec;
        int c = -1;
		Raw tmpRaw;
		for(int i = 0; i < (int)sentence.size(); i++){
            c = sentence.at(i);
			// if the sentence has space
            if(c == 32 || c == 12288){
				hasOther = false;
                if(hasSpace){
                    continue;
                }else{
                    if(graph.size()){
                        graph.back()&=12;
                    }
                    hasSpace=true;
                }
            }else if(isOther(c)){
				if(hasSpace){
					senClean.push_back(getT2S(c));
					if(isSinglePun(c)){
						graph.push_back(8);
						hasSinglePun = true;
					}else{
						graph.push_back(9);
						hasSinglePun = false;
					}
					hasSpace = false;
				}else if(hasOther){
					if(isSinglePun(c)){
						if(graph.size()){
							graph.back() &= 12;
						}
						senClean.push_back(getT2S(c));
						graph.push_back(8);
						hasSinglePun = true;
					}else{
						if(hasSinglePun){
							senClean.push_back(getT2S(c));
							graph.push_back(9);
						}else{						
							if(!graph.back()) graph.back() = 7;
							senClean.push_back(getT2S(c));
							graph.push_back(2);
						}
						hasSinglePun = false;
					}
				}else{
					senClean.push_back(getT2S(c));
					graph.push_back(9);
					if(isSinglePun(c)){
						hasSinglePun = true;
					}else{
						hasSinglePun = false;
					}
				}
				hasOther = true;
			}else{
				if(hasSpace){
					senClean.push_back(getT2S(c));
					graph.push_back(9);
				}else if(hasOther){
					graph.back() &= 12;
					if(hasSinglePun){
						senClean.push_back(getT2S(c));
						graph.push_back(9);
						hasSinglePun = false;
					}else{					
						senClean.push_back(getT2S(c));
						graph.push_back(15);
					}
				}else{
					senClean.push_back(getT2S(c));
					graph.push_back(15);
				}
				hasSpace = false;
				hasOther = false;
			}

			if(isHttp(c)){
				if(!hasHttp){
					if(c == 'h'){
						httpStart = graph.size() - 1;
						tmpRaw.clear();
						tmpRaw.push_back(c);
						hasHttp = true;
					}
				}else{
					tmpRaw.push_back(c);
				}
			}else{
				if(hasHttp){
					httpVec.push_back(tmpRaw);
					httpStartVec.push_back(httpStart);
					hasHttp = false;
				}
			}
		}
		if(tmpRaw.size() != 0){
			httpVec.push_back(tmpRaw);
			httpStartVec.push_back(httpStart);
		}
		
		std::ostringstream ost;
		std::string str;
		for(int i = 0 ; i < httpVec.size(); i ++){
			ost.str("");
			ost<<httpVec[i];
			str = ost.str();
			std::size_t found = str.find("http");
			if(found != std::string::npos){
				int start = httpStartVec[i];
				int size = str.size();
				//std::cout<<std::endl<<sentence<<":Here:"<<str<<":"<<start<<":"<<size<<":"<<graph.size()<<std::endl;
				
				graph[start] = 1;
				for(int j = start + 1; j < start + size - 1; j ++){
					graph[j] = 2;
				}
				graph[start + size - 1] = 4;
				
			}
		}
		
		if(graph.size()!=0){
            graph[0]&=9;
            graph.back()&=12;
		if(!graph[0]) graph[0]=9;
	    	if(!graph.back()) graph.back()=12;
        }
        return 0;
    };
	*/

    int cleanSpace(RawSentence& sentence, RawSentence& senClean, POCGraph& graph){
        senClean.clear();
        graph.clear();
        bool hasSpace=false;//use to check whether the char is a space 
        int c = -1;
		int wordLength = 0;
        for(int i=0;i<(int)sentence.size();i++){
            c = sentence.at(i);
		// if the sentence has space
            if(c==32 || c==12288){
                if(hasSpace){
                    continue;
                }else{
                    if(graph.size()){
						if(wordLength == 1){
							graph.back() = 8;
						}else{
							graph.back() = 4;
						}
                    }
                    hasSpace=true;
                }
				wordLength = 0;
            }else{
                if(hasSpace){
                    senClean.push_back(c);
                    graph.push_back(1);
                    hasSpace=false;
                }else{
                    senClean.push_back(c);
					if(graph.size() == 0){
						graph.push_back(1);
					}else{
						graph.push_back(2);
					}
                }
				wordLength ++;
            }
        }
        if(graph.size()){
			if(wordLength == 1){
				graph.back() = 8;
			}else{
				graph.back() = 4;
			}
        }
        return 0;
    };
};


}//end for thulac
