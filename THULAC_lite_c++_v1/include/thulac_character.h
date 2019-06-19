#pragma once

namespace thulac{

typedef int Character;

inline void put_character(const Character c,FILE * pFile=stdout){
    if(c<128){//1个byte的utf-8
        fputc(c,pFile);
    }else if(c<0x800){//2个byte的utf-8
        fputc(0xc0|(c>>6),pFile);
        fputc(0x80|(c&0x3f),pFile);
    }else if(c<0x10000){//3个byte的utf-8
        fputc(0xe0|((c>>12)&0x0f),pFile);
        fputc(0x80|((c>>6)&0x3f),pFile);
        fputc(0x80|(c&0x3f),pFile);
    }else {//4个byte的utf-8
        fputc(0xf0|((c>>18)&0x07),pFile);
        fputc(0x80|((c>>12)&0x3f),pFile);
        fputc(0x80|((c>>6)&0x3f),pFile);
        fputc(0x80|(c&0x3f),pFile);
    }
}

inline void put_character(const Character c,std::ostream& os){
    if(c<128){//1个byte的utf-8
        os<<(char)c;
    }else if(c<0x800){//2个byte的utf-8
        os<<(char)(0xc0|(c>>6));
        os<<(char)(0x80|(c&0x3f));
    }else if(c<0x10000){//3个byte的utf-8
        os<<(char)(0xe0|((c>>12)&0x0f));
        os<<(char)(0x80|((c>>6)&0x3f));
        os<<(char)(0x80|(c&0x3f));
    }else {//4个byte的utf-8
        os<<(char)(0xf0|((c>>18)&0x07));
        os<<(char)(0x80|((c>>12)&0x3f));
        os<<(char)(0x80|((c>>6)&0x3f));
        os<<(char)(0x80|(c&0x3f));
    }
}

std::istream& operator>> (std::istream& is,Character& c){
    c=is.get();
    if(c==EOF){
        return is;
    }
    while(!(c&0x40)){
        if(c==EOF)return is;
        c=is.get();
    }//not a beginning of a Character
    
    if(!(c&0x80))return is;//one byte utf8
    if(!(c&0x20)){//2个byte的utf-8编码
        c&=0x1f;
        c=(c<<6)+(is.get()&0x3f);
    }else if(!(c&0x10)){//3个byte的utf-8编码
        c&=0x0f;
        c=(c<<6)+(is.get()&0x3f);
        c=(c<<6)+(is.get()&0x3f);
    }else if(!(c&0x80)){//4个byte的utf-8编码
        c&=0x07;
        c=(c<<6)+(is.get()&0x3f);
        c=(c<<6)+(is.get()&0x3f);
        c=(c<<6)+(is.get()&0x3f);
    }
    return is;
};


}//end of thulac

