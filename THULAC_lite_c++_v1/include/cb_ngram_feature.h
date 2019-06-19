#pragma once
#include<cstdio>
#include<iostream>
#include "dat.h"
#include "cb_model.h"
#include<map>
#include "thulac_base.h"
namespace thulac{//beginning of thulac

/*
what we need:
a model, a DAT
*/
class NGramFeature{
private:
    static const int SENTENCE_BOUNDARY='#';
    int SEPERATOR;
    int max_length;
    ///*特征、双数组相关*/
    int* uni_bases;
    int* bi_bases;

    int* values;
    ///*双数组*/
    int dat_size;//双数组大小
    DAT::Entry* dat;

    permm::Model* model;

public: 
    NGramFeature(){
        SEPERATOR=' ';
        uni_bases=NULL;
        bi_bases=NULL;
    };
    NGramFeature(DAT* dat,permm::Model* model,int* values){
        SEPERATOR=' ';
        this->dat=dat->dat;
        this->dat_size=dat->dat_size;
        this->model=model;
        max_length=50000;
        this->uni_bases=new int[this->max_length+2];
        this->bi_bases=new int[this->max_length+4];
        this->values=values;
    };
    ~NGramFeature(){
        if(uni_bases)delete[] uni_bases;
        if(bi_bases)delete[] bi_bases;
    };

    void UpdateModel(permm::Model* model) {
        this->model = model;
    }

    inline void feature_generation(RawSentence& seq,
                Indexer<RawSentence>& indexer,
                Counter<Word>* bigram_counter=NULL){
        int mid=0;
        int left=0;int left2=0;
        int right=0;int right2=0;
        RawSentence key;
        RawSentence bigram;
        for(int i=0;i<seq.size();i++){
            mid=seq[i];
            left=(i>0)?(seq[i-1]):(SENTENCE_BOUNDARY);
            left2=((i-2)>=0)?(seq[i-2]):(SENTENCE_BOUNDARY);
            right=((i+1)<seq.size())?(seq[i+1]):(SENTENCE_BOUNDARY);
            right2=((i+2)<seq.size())?(seq[i+2]):(SENTENCE_BOUNDARY);
            
            if(bigram_counter){
                if(i==0){
                    bigram.clear();
                    bigram.push_back(left2);bigram.push_back(left);
                    bigram_counter->update(bigram);
                    bigram.clear();
                    bigram.push_back(left);bigram.push_back(mid);
                    bigram_counter->update(bigram);
                    bigram.clear();
                    bigram.push_back(mid);bigram.push_back(right);
                    bigram_counter->update(bigram);
                }else{
                    bigram.clear();
                    bigram.push_back(right);bigram.push_back(right2);
                    bigram_counter->update(bigram);
                }
            }
            
            key.clear();
            key.push_back(mid);key.push_back(SEPERATOR);key.push_back('1');
            indexer.get_index(key);
            key.clear();
            key.push_back(left);key.push_back(SEPERATOR);key.push_back('2');
            indexer.get_index(key);
            key.clear();
            key.push_back(right);key.push_back(SEPERATOR);key.push_back('3');
            indexer.get_index(key);
            
            key.clear();
            key.push_back(left);key.push_back(mid);key.push_back(SEPERATOR);key.push_back('1');
            indexer.get_index(key);
            key.clear();
            key.push_back(mid);key.push_back(right);key.push_back(SEPERATOR);key.push_back('2');
            indexer.get_index(key);
            key.clear();
            key.push_back(left2);key.push_back(left);key.push_back(SEPERATOR);key.push_back('3');
            indexer.get_index(key);
            key.clear();
            key.push_back(right);key.push_back(right2);key.push_back(SEPERATOR);key.push_back('4');
            indexer.get_index(key);
        }
    }

    int put_values(int*sequence,int len){
        if(len>=this->max_length){
            fprintf(stderr,"larger than max\n");
            return 1;
        }
        find_bases(dat_size,SENTENCE_BOUNDARY,SENTENCE_BOUNDARY,uni_bases[0],bi_bases[0]);
        find_bases(dat_size,SENTENCE_BOUNDARY,sequence[0],uni_bases[0],bi_bases[1]);
        for(int i=0;i+1<len;i++)
            find_bases(dat_size,sequence[i],sequence[i+1],uni_bases[i+1],bi_bases[i+2]);
        //return 1;
        find_bases(dat_size,sequence[len-1],SENTENCE_BOUNDARY,uni_bases[len],bi_bases[len+1]);
        find_bases(dat_size,SENTENCE_BOUNDARY,SENTENCE_BOUNDARY,uni_bases[len+1],bi_bases[len+2]);
        int base=0;
        for(int i=0;i<len;i++){
            int* value_offset=values+i*model->l_size;
            if((base=uni_bases[i+1])!=-1){
                add_values(value_offset,base,49,NULL);
                //check_values(value_offset,base,49,NULL);
            }
            if((base=uni_bases[i])!=-1)
                add_values(value_offset,base,50,NULL);
            if((base=uni_bases[i+2])!=-1)
                add_values(value_offset,base,51,NULL);
            if((base=bi_bases[i+1])!=-1)
                add_values(value_offset,base,49,NULL);
            if((base=bi_bases[i+2])!=-1)
                add_values(value_offset,base,50,NULL);
            if((base=bi_bases[i])!=-1)
                add_values(value_offset,base,51,NULL);
            if((base=bi_bases[i+3])!=-1)
                add_values(value_offset,base,52,NULL);
        }
        return 0;
    }

    void update_weights(int*sequence,int len,int* results,int delta,long steps){
        find_bases(dat_size,SENTENCE_BOUNDARY,SENTENCE_BOUNDARY,uni_bases[0],bi_bases[0]);
        find_bases(dat_size,SENTENCE_BOUNDARY,sequence[0],uni_bases[0],bi_bases[1]);
        for(int i=0;i+1<len;i++)
            find_bases(dat_size,sequence[i],sequence[i+1],uni_bases[i+1],bi_bases[i+2]);
        find_bases(dat_size,sequence[len-1],SENTENCE_BOUNDARY,uni_bases[len],bi_bases[len+1]);
        find_bases(dat_size,SENTENCE_BOUNDARY,SENTENCE_BOUNDARY,uni_bases[len+1],bi_bases[len+2]);
        
        int base=0;
        for(int i=0;i<len;i++){
            int* value_offset=values+i*model->l_size;
            if((base=uni_bases[i+1])!=-1)
                update_weight(value_offset,base,49,results[i],delta,steps);
            if((base=uni_bases[i])!=-1)
                update_weight(value_offset,base,50,results[i],delta,steps);
            if((base=uni_bases[i+2])!=-1)
                update_weight(value_offset,base,51,results[i],delta,steps);
            if((base=bi_bases[i+1])!=-1)
                update_weight(value_offset,base,49,results[i],delta,steps);
            if((base=bi_bases[i+2])!=-1)
                update_weight(value_offset,base,50,results[i],delta,steps);
            if((base=bi_bases[i])!=-1)
                update_weight(value_offset,base,51,results[i],delta,steps);
            if((base=bi_bases[i+3])!=-1)
                update_weight(value_offset,base,52,results[i],delta,steps);
        }

    }
private:
    inline void update_weight(int *value_offset,int base,int del,int label,int delta,long steps){
        int ind=dat[base].base+del;
        if(ind>=dat_size||dat[ind].check!=base)return;
        register int offset=dat[dat[base].base+del].base;
        model->update_fl_weight(offset,label,delta,steps);
        //model->fl_weights[offset*model->l_size+label]+=delta;
    }
    /*只内部调用*/
    inline void add_values(int *value_offset,int base,int del,int* p_allowed_label=NULL){
        int ind=dat[base].base+del;
        if(ind>=dat_size||dat[ind].check!=base){
            return;
        }
        int offset=dat[dat[base].base+del].base;
        int* weight_offset=model->fl_weights+offset*model->l_size;
        int allowed_label;
        if(model->l_size==4){
            value_offset[0]+=weight_offset[0];
            value_offset[1]+=weight_offset[1];
            value_offset[2]+=weight_offset[2];
            value_offset[3]+=weight_offset[3];
        }else{
            if(p_allowed_label){
                while((allowed_label=(*(p_allowed_label++)))>=0){
                    value_offset[allowed_label]+=weight_offset[allowed_label];
                }
            }else{
                for(int j=0;j<model->l_size;j++){
                    value_offset[j]+=weight_offset[j];
                }
            }
        }
    };
    inline void check_values(int *value_offset,int base,int del,int* p_allowed_label=NULL){
        int ind=dat[base].base+del;
        if(ind>=dat_size||dat[ind].check!=base){
            return;
        }
        int offset=dat[dat[base].base+del].base;
        int* weight_offset=model->fl_weights+offset*model->l_size;
        int allowed_label;
        if(model->l_size==4){
            value_offset[0]+=weight_offset[0];
            //if(weight_offset[0]==0)value_offset[0]-=100000;
            value_offset[1]+=weight_offset[1];
            //if(weight_offset[1]==0)value_offset[1]-=100000;
            value_offset[2]+=weight_offset[2];
            //if(weight_offset[2]==0)value_offset[2]-=100000;
            value_offset[3]+=weight_offset[3];
            //if(weight_offset[3]==0)value_offset[3]-=100000;
        }else{
            if(p_allowed_label){
                while((allowed_label=(*(p_allowed_label++)))>=0){
                    value_offset[allowed_label]+=weight_offset[allowed_label];
                    if(weight_offset[allowed_label]==0)value_offset[allowed_label]-=100000;
                }
            }else{
                for(int j=0;j<model->l_size;j++){
                    value_offset[j]+=weight_offset[j];
                    if(weight_offset[j]==0)value_offset[j]-=100000;
                }
            }
        }
    };

    /*
     * 找出以ch1 ch2为字符的dat的下标
     * */
    inline void find_bases(int dat_size,int ch1,int ch2,int& uni_base,int&bi_base){
        if(ch1>32 &&ch1<128)ch1+=65248;
        if(ch2>32 &&ch2<128)ch2+=65248;
        if(dat[ch1].check){
            uni_base=-1;bi_base=-1;return;
        }
        uni_base=dat[ch1].base+SEPERATOR;
        int ind=dat[ch1].base+ch2;
        if(ind>=dat_size||dat[ind].check!=ch1){
            bi_base=-1;return;
        }
        bi_base=dat[ind].base+SEPERATOR;
    }


};


}//end of thulac
