#pragma once

#include <cstdio>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <algorithm> 

#include "cb_tagging_decoder.h"
#include "cb_ngram_feature.h"
#include "dat.h"


namespace thulac{




// TaggingDecoder
class TaggingLearner : public  TaggingDecoder {
public:
    Character* gold_standard;
    int T;
    Character separator;
    int seg_only;
    int bigram_threshold;
    void load_tagged_sentence(FILE* file);
    
    
    void train(const char*training_file,
            const char*dat_file,
            const char*model_file,
            const char*label_file);

    TaggingLearner(int T=15,Character separator='/',int seg_only=false){
        this->T=T;
        this->gold_standard=new int[this->max_length];
        this->separator=separator;
        this->seg_only=seg_only;
        bigram_threshold=1;
    };
    ~TaggingLearner(){
        delete this->gold_standard;
    };
    
    Indexer<RawSentence> tag_indexer;
    Indexer<RawSentence> ngram_indexer;
private:
    inline int get_char_tag(const int& poc,const RawSentence& tag){
        RawSentence char_tag;
        char_tag.clear();
        char_tag.push_back(poc);
        if((!seg_only)&&tag.size()){
            //char_tag.push_back('/');
            for(size_t j=0;j<tag.size();j++)char_tag.push_back(tag[j]);
        }
        return tag_indexer.get_index(char_tag);
    };
};


class TaggedFileLoader{
private:    
    std::string str;
    std::string item;

    char del;//词和词性之间的分隔符
    RawSentence tag;
    RawSentence word;

    RawSentence char_tag;
public:

    struct WordAndTag{
        Word word;
        Word tag;
    };
 
    std::ifstream* ifs;
    TaggedFileLoader(const char* filename,int del='_'){
        this->del=del;
        ifs=new std::ifstream(filename,std::ifstream::in);
    };
    void load(std::vector<WordAndTag>& seq){
        seq.clear();
        getline((*ifs),str);
        std::istringstream iss(str);
        while(iss){
            item.clear();
            iss>>item;
            if(!item.length())continue;
            int del_ind=item.find_first_of(del);

            const std::string&tag_string=item.substr(del_ind+1,item.length());
            const std::string&word_string=item.substr(0,del_ind);
            seq.push_back(WordAndTag());
            
            string_to_raw(tag_string,seq.back().tag);
            string_to_raw(word_string,seq.back().word);
            
        }
        
    };
    ~TaggedFileLoader(){
        delete ifs;
    };

};


bool compare_words (DATMaker::KeyValue first, DATMaker::KeyValue second)
{
    thulac::Word& first_word=first.key;
    thulac::Word& second_word=second.key;
    size_t min_size=(first_word.size()<second_word.size())?first_word.size():second_word.size();
    for(int i=0;i<min_size;i++){
        if(first_word[i]>second_word[i])return false;
        if(first_word[i]<second_word[i])return true;
    }
    
  return (first_word.size()<second_word.size());
}




void TaggingLearner::train(const char*training_file,
        const char*model_file,
        const char*dat_file,
        const char*label_file){
    

    RawSentence raw;
    RawSentence char_tag;
    NGramFeature ngram_feature;
    Counter<Word> bigram_counter;
    Word bigram;
    

        

    std::vector<TaggedFileLoader::WordAndTag> sent;
    TaggedFileLoader* tfl=new TaggedFileLoader(training_file,this->separator);

    
    std::cout<<"separator: ["<<(char)this->separator<<"]\n";
    while((*(tfl->ifs))){
        tfl->load(sent);
        if(!sent.size())continue;
        raw.clear();
        for(int i=0;i<sent.size();i++){
            const RawSentence& word=sent[i].word;
            const RawSentence& tag=sent[i].tag;
            for(int j=0;j<word.size();j++)raw.push_back(word[j]);
            
            if(word.size()==1){
                get_char_tag(kPOC_S,tag);
            }else{
                get_char_tag(kPOC_B,tag);
                if(word.size()>2){
                    get_char_tag(kPOC_M,tag);
                }
                get_char_tag(kPOC_E,tag);
            }
        }
        ngram_feature.feature_generation(raw,ngram_indexer,
                    (bigram_threshold>1)?(&bigram_counter):(NULL));
        
        
    }
    delete tfl;
    std::cerr<<"training file \""<<training_file<<"\" scanned\n";
    
    //dat_file

    std::vector<DATMaker::KeyValue> kv;
    int feature_ind=0;
    for(int i=0;i<ngram_indexer.list.size();i++){
        const Word& feature_raw=ngram_indexer.list[i];
        //filter some bigrams
        if((bigram_threshold>1)&&(feature_raw.size()>=2)){
            if((feature_raw[0]!=' ')&&(feature_raw[1]!=' ')){
                bigram.clear();
                bigram.push_back(feature_raw[0]);bigram.push_back(feature_raw[1]);
                if(bigram_counter[bigram]<bigram_threshold){
                    continue;
                }
            }
        }
        kv.push_back(DATMaker::KeyValue());
        kv.back().key=feature_raw;
        kv.back().value=feature_ind++;
        
    }
    
    std::sort(kv.begin(),kv.end(),compare_words);
    
    DATMaker* dm=new DATMaker();
    dm->make_dat(kv,1);
    dm->shrink();
    //dm->save_as(dat_file);
	dm->save(dat_file);
    delete dm;
    std::cerr<<"DAT (double array TRIE) file \""<<dat_file<<"\" created\n";
    
    
    //model_file
    int l_size=tag_indexer.list.size();
    int f_size=kv.size();
    printf("number of labels: %d\n",l_size);
    printf("number of features: %d\n",f_size);
    permm::Model* model=new permm::Model(l_size,f_size);
    model->save(model_file);
    delete model;
    
    std::cerr<<"model file \""<<model_file<<"\" created\n";
    
    
    
    //label_file
    FILE * pFile=fopen(label_file,"w");
    for(int i=0;i<tag_indexer.list.size();i++){
        put_raw(tag_indexer.list[i],pFile);
        fputc('\n',pFile);
    }
    fclose(pFile);
    std::cerr<<"label file \""<<label_file<<"\" created\n";
    

    //init decoder
    //init(model_file,dat_file,label_file);
	permm::Model* cws_model = new permm::Model(model_file);
    DAT* cws_dat = new DAT(dat_file);
    char** cws_label_info = new char*[cws_model->l_size];
    int** cws_pocs_to_tags = new int*[16];

    get_label_info(label_file, cws_label_info, cws_pocs_to_tags);
    init(cws_model, cws_dat, cws_label_info, cws_pocs_to_tags);
	set_label_trans();

    //do not use the original read-only model.
    delete this->model;
    //this->model=new permm::Model(model_file,false);
    this->model=new permm::Model(model_file);
    this->model->reset_ave_weights();
    this->ngram_feature->UpdateModel(this->model);
    fprintf(stderr,"decoder initialized\n");
    
    //learning !!!
    
    long steps=0;
    for(int t=0;t<this->T;t++){
        fprintf(stderr,"iteration %d\n",t+1);
        //continue;
        int number_nodes=0;
        int number_correct=0;
        tfl=new TaggedFileLoader(training_file,this->separator);
        while((*(tfl->ifs))){
            tfl->load(sent);
            if(!sent.size())continue;
            steps++;
            len=0;
            
            //putchar('\n');
            for(int i=0;i<sent.size();i++){
                const RawSentence& word=sent[i].word;
                const RawSentence& tag=sent[i].tag;
                //put_raw(word);putchar(' ');   
                for(int j=0;j<word.size();j++){
                    
                    this->sequence[len]=word[j];
                    if(word.size()==1){
                        gold_standard[len]=get_char_tag(kPOC_S,tag);
                    }else{
                        if(j==0){
                            gold_standard[len]=get_char_tag(kPOC_B,tag);
                        }else if((j+1)==word.size()){
                            gold_standard[len]=get_char_tag(kPOC_E,tag);
                        }else{
                            gold_standard[len]=get_char_tag(kPOC_M,tag);
                        }
                    }
                    len++;
                    if(len>=this->max_length){
                        //fprintf(stderr,"longer than max\n");
                        break;
                    }

                }
                if(len>=this->max_length){
                    fprintf(stderr,"longer than max\n");
                    break;
                }
            }
            if(len>=this->max_length){
                continue;
            }
            //printf("len: %d\n",len);
                        //decode
            put_values();
            //continue;
            dp();

            
            //update
            this->ngram_feature->update_weights(sequence,len,gold_standard,1,steps);
            this->ngram_feature->update_weights(sequence,len,result,-1,steps);
            for(int i=0;i<len-1;i++){
                this->model->update_ll_weight(gold_standard[i],gold_standard[i+1],1,steps);
                this->model->update_ll_weight(result[i],result[i+1],-1,steps);
            }

            
            for(int i=0;i<len;i++){
                
                number_nodes++;
                if(gold_standard[i]==result[i])number_correct++;
            }
            
        }
        std::cout<<number_correct<<" "<<number_nodes<<" "<<(double)number_correct/number_nodes<<"\n";
        delete tfl;
    }
    //average
    this->model->average(steps);
    //save model
    this->model->save(model_file);
}



}//end of thulac