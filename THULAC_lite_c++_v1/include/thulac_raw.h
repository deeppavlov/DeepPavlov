#pragma once
#include <set>
#include "thulac_character.h"
namespace thulac{

class Raw:public std::vector<Character>{
public:
    Raw& operator+=(Raw& right){
        for(size_t i=0;i<right.size();i++){
            this->push_back(right[i]);
        };
        return *this;
    };
    Raw& operator+=(const char* right){
        while(*right){
            this->push_back(*(right++));
        }
        return *this;
    };

    Raw& operator+=(const char& right){
        this->push_back(right);
        return *this;
    };
    Raw& operator+=(const std::string& right){
        for(size_t i=0;i<right.size();i++){
            this->push_back(right[i]);
        };
        return *this;
    };

    /*
     * 从右边查找相应的字，返回下标，-1表示没有找到
     * */
    int rfind(const Character& sep){
        int ind=this->size()-1;
        while(ind>=0){
            if((*this)[ind]==sep)return ind;
            ind--;
        }
        return -1;
    };

    friend std::ostream& operator<< (std::ostream& os,const Raw& raw){
        for(size_t i=0;i<raw.size();i++){
            put_character(raw[i],os);
        }
        return os;    
    };
};

inline void put_raw(const Raw& r,FILE * pFile=stdout){
    for(size_t j=0;j<r.size();j++){
        put_character(r[j],pFile);
    }
}

inline int string_to_raw(const std::string& str,Raw& sent){
    sent.clear();
    int current_character=-1;
    int c;
    for(int i=0;i<str.length();i++){
        c=str.at(i);
        if(!(c&0x80)){//1个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=c;//+65248;//半角转全角，放入缓存
        }else if(!(c&0x40)){//not a beginning of a Character
            current_character=(current_character<<6)+(c&0x3f);
        }else if(!(c&0x20)){//2个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=(c&0x1f);
        }else if(!(c&0x10)){//3个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=(c&0x0f);
        }else if(!(c&0x80)){//4个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=(c&0x07);
        }else{//更大的unicode编码不能处理
            if(current_character!=-1)sent.push_back(current_character);
            current_character=0;
        }
    }
    if(current_character>0)sent.push_back(current_character);
    return 0;
}

inline int get_raw(Raw& sent,FILE* pFile=stdin,int min_char=33){
    sent.clear();
    int current_character=-1;
    int c;
    while(1){//反复读取输入流
        c=fgetc(pFile);
        if(c==EOF){
            if(current_character!=-1)sent.push_back(current_character);
            return c;//end of file
        }
        if(!(c&0x80)){//1个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            if(c<min_char){//非打印字符及空格
                return c;
            }else{//一般ascii字符
                current_character=c;//+65248;//半角转全角，放入缓存
            }
        }else if(!(c&0x40)){//not a beginning of a Character
            current_character=(current_character<<6)+(c&0x3f);
        }else if(!(c&0x20)){//2个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=(c&0x1f);
        }else if(!(c&0x10)){//3个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=(c&0x0f);
        }else if(!(c&0x80)){//4个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=(c&0x07);
        }else{//更大的unicode编码不能处理
            if(current_character!=-1)sent.push_back(current_character);
            current_character=0;
        }
    }
}
inline int get_raw(Character* seq,int max_len,int&len,FILE* pFile=stdin,int min_char=33){
    len=0;
    Character current_character=-1;
    int c;
    while(1){//反复读取输入流
        
        c=fgetc(pFile);
        if(c==EOF){
            if((current_character!=-1)&&(len<max_len))seq[len++]=current_character;
            return c;//end of file
        }
        if(!(c&0x80)){//1个byte的utf-8编码
            if((current_character!=-1)&&(len<max_len))seq[len++]=current_character;
            if(c<min_char){//非打印字符及空格
                return c;
            }else{//一般ascii字符
                current_character=c;//+65248;//半角转全角，放入缓存
            }
        }else if(!(c&0x40)){//not a beginning of a Character
            current_character=(current_character<<6)+(c&0x3f);
        }else if(!(c&0x20)){//2个byte的utf-8编码
            if((current_character!=-1)&&(len<max_len))seq[len++]=current_character;
            current_character=(c&0x1f);
        }else if(!(c&0x10)){//3个byte的utf-8编码
            if((current_character!=-1)&&(len<max_len))seq[len++]=current_character;
            current_character=(c&0x0f);
        }else if(!(c&0x80)){//4个byte的utf-8编码
            if((current_character!=-1)&&(len<max_len))seq[len++]=current_character;
            current_character=(c&0x07);
        }else{//更大的unicode编码不能处理
            if((current_character!=-1)&&(len<max_len))seq[len++]=current_character;
            current_character=0;
        }
    }
}

inline int get_raw(Raw& sent,const std::string& s,int len, int start, int min_char=33){
    sent.clear();
    int current_character=-1;
    int c;
    for(int i = start; i < len; i ++){//反复读取s
        c=s[i];
        if(!(c&0x80)){//1个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            if(c<min_char){//非打印字符及空格
                return i;
                // current_character=32;
            }else{//一般ascii字符
                current_character=c;//+65248;//半角转全角，放入缓存
            }
        }else if(!(c&0x40)){//not a beginning of a Character
            current_character=(current_character<<6)+(c&0x3f);
        }else if(!(c&0x20)){//2个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=(c&0x1f);
        }else if(!(c&0x10)){//3个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=(c&0x0f);
        }else if(!(c&0x80)){//4个byte的utf-8编码
            if(current_character!=-1)sent.push_back(current_character);
            current_character=(c&0x07);
        }else{//更大的unicode编码不能处理
            if(current_character!=-1)sent.push_back(current_character);
            current_character=0;
        }
    }
	if(current_character>0){
		sent.push_back(current_character);
	}
    return -1;
    //if(!(c&0x80))sent.push_back(current_character);
    //return 0;
}

inline int get_raw_vector(std::vector<Raw>& vec,FILE* pFile=stdin,int min_char=33){
    vec.clear();
	Raw sent;
	sent.clear();
	//std::vector<int> pun_vec;
	std::set<int> pun_set;
	std::set<int>::iterator it;
	//int punInts[] = {46,63,33,12290,65311,65281};
	int punInts[] = {63,33,59,12290,65311,65281,65307};
	for(int i = 0; i < 7; i ++){
		pun_set.insert(punInts[i]);
	}
    int current_character=-1;
    int c;
    while(1){//反复读取输入流
        c=fgetc(pFile);
        // std::cout << c;
        if(c==EOF){
            if(current_character!=-1)sent.push_back(current_character);
            if(sent.size()) vec.push_back(sent);
            return c;//end of file
        }
        if(!(c&0x80)){//1个byte的utf-8编码
            if(current_character!=-1){
				sent.push_back(current_character);
				it = pun_set.find(current_character);
				if(it != pun_set.end()){
					//pun_vec.push_back(sent.size());
					//std::cout<<"find pun at "<<sent.size()<<std::endl;
					if(sent.size() == 1){
						vec.push_back(sent);
						sent.clear();
					}else{
						int last_character = sent[sent.size()-2];
						it = pun_set.find(last_character);
						if(it == pun_set.end()){
							vec.push_back(sent);
							sent.clear();
						}
					}
				}
			}
            if(c<min_char){//非打印字符及空格
                if(sent.size()) vec.push_back(sent);
                // break;
                return c;
                // current_character=32;
            }else{//一般ascii字符
                current_character=c;//+65248;//半角转全角，放入缓存
            }
        }else if(!(c&0x40)){//not a beginning of a Character
            current_character=(current_character<<6)+(c&0x3f);
        }else if(!(c&0x20)){//2个byte的utf-8编码
            //if(current_character!=-1)sent.push_back(current_character);
			if(current_character!=-1){
                sent.push_back(current_character);
                it = pun_set.find(current_character);
                if(it != pun_set.end()){
                    //pun_vec.push_back(sent.size());
					//std::cout<<"find pun at "<<sent.size()<<std::endl;
                    if(sent.size() == 1){
                        vec.push_back(sent);
                        sent.clear();
                    }else{
                        int last_character = sent[sent.size()-2];
                        it = pun_set.find(last_character);
                        if(it == pun_set.end()){
                            vec.push_back(sent);
                            sent.clear();
                        }
                    }
                }
            }
            current_character=(c&0x1f);
        }else if(!(c&0x10)){//3个byte的utf-8编码
            //if(current_character!=-1)sent.push_back(current_character);
			if(current_character!=-1){
                sent.push_back(current_character);
                it = pun_set.find(current_character);
                if(it != pun_set.end()){
                    //pun_vec.push_back(sent.size());
					//std::cout<<"find pun at "<<sent.size()<<std::endl;
                    if(sent.size() == 1){
                        vec.push_back(sent);
                        sent.clear();
                    }else{
                        int last_character = sent[sent.size()-2];
                        it = pun_set.find(last_character);
                        if(it == pun_set.end()){
                            vec.push_back(sent);
                            sent.clear();
                        }
                    }
                }
            }
            current_character=(c&0x0f);
        }else if(!(c&0x80)){//4个byte的utf-8编码
            //if(current_character!=-1)sent.push_back(current_character);
			if(current_character!=-1){
                sent.push_back(current_character);
                it = pun_set.find(current_character);
                if(it != pun_set.end()){
                    //pun_vec.push_back(sent.size());
					//std::cout<<"find pun at "<<sent.size()<<std::endl;
                    if(sent.size() == 1){
                        vec.push_back(sent);
                        sent.clear();
                    }else{
                        int last_character = sent[sent.size()-2];
                        it = pun_set.find(last_character);
                        if(it == pun_set.end()){
                            vec.push_back(sent);
                            sent.clear();
                        }
                    }
                }
            }
            current_character=(c&0x07);
        }else{//更大的unicode编码不能处理
            //if(current_character!=-1)sent.push_back(current_character);
			if(current_character!=-1){
                sent.push_back(current_character);
                it = pun_set.find(current_character);
                if(it != pun_set.end()){
                    //pun_vec.push_back(sent.size());
                    if(sent.size() == 1){
                        vec.push_back(sent);
                        sent.clear();
                    }else{
                        int last_character = sent[sent.size()-2];
                        it = pun_set.find(last_character);
                        if(it == pun_set.end()){
                            vec.push_back(sent);
                            sent.clear();
                        }
                    }
                }
            }
            current_character=0;
        }
    }
    //if(current_character>0 && len != 9999)sent.push_back(current_character);
 //    std::cout << sent << std::endl;
	// if(current_character > 0){
 //        sent.push_back(current_character);
 //        vec.push_back(sent);
	// 	sent.clear();
 //    }else if(current_character > 0){
 //        vec.push_back(sent);
	// 	sent.clear();
	// }
    // return -1;
	// for(int i = 0; i < vec.size(); i ++){
		// std::cout<<"get_raw_vec:"<<vec[i]<<std::endl;
	// }
    // return c;
	// int startIndex = 0;
	// int endIndex = 0;
	// Raw tmpRaw;
	// for(int i = 0; i < pun_vec.size(); i ++){
	// 	startIndex = (i == 0) ? 0 : pun_vec[i - 1];
	// 	endIndex = pun_vec[i];
	// 	if(endIndex > 1 )
	// 	std::cout<<"get_raw_vec:"<<pun_vec[i]<<std::endl;
	// }
    //if(!(c&0x80))sent.push_back(current_character);
    //return 0;
}


inline void cut_raw(Raw& sent, std::vector<Raw>& vec, int max_len){
    vec.clear();
    //std::vector<int> pun_vec;
    Raw sent_tmp;
    std::set<int> pun_set;
    std::set<int>::iterator it;
    //int punInts[] = {46,63,33,12290,65311,65281};
    int punInts[] = {63,33,59,12290,65311,65281,65307};
    for(int i = 0; i < 7; i ++){
        pun_set.insert(punInts[i]);
    }
    int current_character=-1;
    int c, num = 0, last_pun = 0;
    sent_tmp.clear();
    for(int i = 0; i < sent.size(); i++){//反复读取输入流
        c = sent[i];
        num++;
        it = pun_set.find(c);
        if(it != pun_set.end() || i == sent.size()-1) {
            if(num > max_len) {
                vec.push_back(sent_tmp);
                sent_tmp.clear();
                num = i - last_pun + 1;
            }
            for(int j = last_pun; j <= i; j++) sent_tmp.push_back(sent[j]);
            last_pun = i+1;
        }
    }
    if(sent_tmp.size()) vec.push_back(sent_tmp);
}

}//for thulac
