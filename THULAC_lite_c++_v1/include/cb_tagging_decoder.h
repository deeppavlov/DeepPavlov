#pragma once
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <list>
#include "thulac_base.h"
#include "dat.h"
#include "cb_ngram_feature.h"
#include "cb_decoder.h"
#include "cb_model.h"


namespace thulac{

//用于分词和词性标注的类
class TaggingDecoder{
public:
    char separator;

    int max_length;
    /*句子*/
    int len;//句子长度
    int* sequence;//句子
    int** allowed_label_lists;///
    int** pocs_to_tags;///

    ///*特征*//
    NGramFeature* ngram_feature;

    

    ///*双数组*/
    DAT* dat;
    int is_old_type_dat;

    ///*模型参数*/
    permm::Model* model;

    ///*解码用*/
    permm::Node* nodes;//只用来存储解码用的节点的拓扑结构
    int* values;//存各个节点的权重
    permm::Alpha_Beta* alphas;//前向算法数据
    permm::Alpha_Beta* betas;//后向算法数据
    int best_score;
    int* result;//存储结果
    
    char** label_info;


    ///**合法转移矩阵*/
    int*label_trans;
    int**label_trans_pre;
    int**label_trans_post;

    ///*后处理用_ tagging*/
    int tag_size;//postag的个数
    int** label_looking_for;
    int* is_good_choice;
    
    /*构造函数*/
    TaggingDecoder();
    ~TaggingDecoder();
    
    
    /*初始化*/
    void init(permm::Model* model, DAT* dat, char** label_info, int** pocs_to_tags,
        char* label_trans=NULL);
    void set_label_trans();//
   
    /*解码*/
    void put_values();
    void dp();
    void cal_betas();
    
    /*接口*/
    
    int segment(RawSentence&,POCGraph&,TaggedSentence&);
    
    
    void get_result(TaggedSentence&);
    void get_seg_result(SegmentedSentence&);
    /*输入输出*/
    void output_raw_sentence();
    void output_sentence();
    void output_allow_tagging();

    void cs(); 
    
private:
    void load_label_trans(char*filename);
};




TaggingDecoder::TaggingDecoder(){
    this->separator='_';
    this->max_length=50000;   //就是这里！
    this->len=0;
    this->sequence=new int[this->max_length];
    this->allowed_label_lists=new int*[this->max_length];
    pocs_to_tags=NULL;

    ngram_feature=NULL;

    

    dat=NULL; 
    is_old_type_dat=false;
    
    nodes=new permm::Node[this->max_length];
    
    this->label_trans=NULL;
    label_trans_pre=NULL;
    label_trans_post=NULL;

    
    this->tag_size=0;
    //this->is_good_choice=NULL;
    
    this->model=NULL;
    
    alphas=NULL;
    betas=NULL;
    
}
TaggingDecoder::~TaggingDecoder(){
    delete[]sequence;
    delete[]allowed_label_lists;
    
    //delete 用于释放 new 分配的空间，free 有用释放 malloc 分配的空间。
    //关于delete和free的区别详情可以看http://www.cnblogs.com/zhuyp1015/archive/2012/07/20/2601698.html
    
    for(int i=0;i<max_length;i++){
        delete[](nodes[i].predecessors);
        delete[](nodes[i].successors);
    }
    delete[](nodes);
    free(values);
    free(alphas);
    free(betas);
    free(result);
    if(model!=NULL)for(int i=0;i<model->l_size;i++){
        if(label_info)delete[](label_info[i]);
    };
    delete[](label_info);
    
        
    
    free(label_trans);
    if(model!=NULL)for(int i=0;i<model->l_size;i++){
        if(label_trans_pre)free(label_trans_pre[i]);
        if(label_trans_post)free(label_trans_post[i]);
    }
    free(label_trans_pre);
    free(label_trans_post);
    
    
    if(model!=NULL)for(int i=0;i<model->l_size;i++){
        if(label_looking_for)delete[](label_looking_for[i]);
    };
    delete[](label_looking_for);
    
    if(pocs_to_tags){
        for(int i=1;i<16;i++){
            delete[]pocs_to_tags[i];
        }
    }
    delete[]pocs_to_tags;
    
    if(model!=NULL)delete model;
    delete dat;
}

void TaggingDecoder::init(
        permm::Model* model,
        DAT* dat,
        char** label_info,
        int** pocs_to_tags,
        char* label_trans
        ){
    /**模型*/
    this->model = model;
    
    /**解码用*/
    values=(int*)calloc(sizeof(int),max_length*model->l_size);
    alphas=(permm::Alpha_Beta*)calloc(sizeof(permm::Alpha_Beta),max_length*model->l_size);
    betas=(permm::Alpha_Beta*)calloc(sizeof(permm::Alpha_Beta),max_length*model->l_size);
    result=(int*)calloc(sizeof(int),max_length*model->l_size);
    this->label_info=label_info;
    
    for(int i=0;i<max_length;i++){
        int* pr=new int[2];
        pr[0]=i-1;
        pr[1]=-1;
        nodes[i].predecessors=pr;
        
        pr=new int[2];
        pr[0]=i+1;
        pr[1]=-1;
        nodes[i].successors=pr;
    };
    
    //DAT
    this->dat=dat;

    //Ngram Features
    ngram_feature=new NGramFeature(dat,model,values);

    /*pocs_to_tags*/
    this->pocs_to_tags=pocs_to_tags;
    
    label_looking_for=new int*[model->l_size];
    for(int i=0;i<model->l_size;i++)
        label_looking_for[i]=NULL;
    for(int i=0;i<model->l_size;i++){
        if(label_info[i][0]==kPOC_B || label_info[i][0]==kPOC_S)continue;
        
        for(int j=0;j<=i;j++){
            if((strcmp(label_info[i]+1,label_info[j]+1)==0)&&(label_info[j][0]==kPOC_B)){
                if(label_looking_for[j]==NULL){
                    label_looking_for[j]=new int[2];
                    label_looking_for[j][0]=-1;label_looking_for[j][1]=-1;
                    tag_size++;
                }
                label_looking_for[j][label_info[i][0]-'1']=i;
                break;
            }
        }
    }
    //printf("tagsize %d",tag_size);
    

    
    /**label_trans*/
    if(label_trans){
        load_label_trans(label_trans);
    }
    
   for(int i=0;i<max_length;i++)
       allowed_label_lists[i]=NULL;
    
    is_good_choice=new int[max_length*model->l_size];
    
}

void TaggingDecoder::dp(){//调用cb_decoder.h里的函数
    if(allowed_label_lists[0]==NULL){
        allowed_label_lists[0]=pocs_to_tags[9];
    }
    if(allowed_label_lists[len-1]==NULL){
        allowed_label_lists[len-1]=pocs_to_tags[12];
    }
    best_score=dp_decode(
            model->l_size,//check
            model->ll_weights,//check
            len,//check
            nodes,
            values,
            alphas,
            result,
            label_trans_pre,
            allowed_label_lists
        );
    allowed_label_lists[0]=NULL;
    allowed_label_lists[len-1]=NULL;
    /*for(int i=0;i<len;i++){
        printf("%s",label_info[result[i]]);

        std::cout<<" ";
    }std::cout<<"\n";*/
}


void TaggingDecoder::set_label_trans(){//不同位置可能出现的标签种类
    int l_size=this->model->l_size;
    std::list<int> *pre_labels;
    std::list<int> *post_labels;
    pre_labels=new std::list<int>[l_size];
    post_labels=new std::list<int>[l_size];

    for(int i=0;i<l_size;i++)
        for(int j=0;j<l_size;j++){
            // 0:B 1:M 2:E 3:S
            int ni=this->label_info[i][0]-'0';
            int nj=this->label_info[j][0]-'0';
            int i_is_end=((ni==2)//i is end of a word
                    ||(ni==3));
            int j_is_begin=((nj==0)//j is begin of a word
                    ||(nj==3));
            int same_tag=strcmp(this->label_info[i]+1,this->label_info[j]+1);
            
            if(same_tag==0){
                if((ni==0&&nj==1)||
                        (ni==0&&nj==2)||
                        (ni==1&&nj==2)||
                        (ni==1&&nj==1)||
                        (ni==2&&nj==0)||
                        (ni==2&&nj==3)||
                        (ni==3&&nj==3)||
                        (ni==3&&nj==0)){
                    pre_labels[j].push_back(i);
                    post_labels[i].push_back(j);
                    //printf("ok\n");
                }
            }else{
                //printf("%s <> %s\n",this->label_info[i],this->label_info[j]);
                if(i_is_end&&j_is_begin){
                    pre_labels[j].push_back(i);
                    post_labels[i].push_back(j);
                }
            }
        }
    label_trans_pre=new int*[l_size];
    for(int i=0;i<l_size;i++){
        label_trans_pre[i]=new int[(int)pre_labels[i].size()+1];
        int k=0;
        for(std::list<int>::iterator plist = pre_labels[i].begin();
                plist != pre_labels[i].end(); plist++){
            label_trans_pre[i][k]=*plist;
            k++;
        };
        label_trans_pre[i][k]=-1;
    }
    label_trans_post=new int*[l_size]; 
    for(int i=0;i<l_size;i++){
        label_trans_post[i]=new int[(int)post_labels[i].size()+1];
        int k=0;
        for(std::list<int>::iterator plist=post_labels[i].begin();
                plist!=post_labels[i].end();++plist){
            label_trans_post[i][k]=*plist;
            k++;
        };
        label_trans_post[i][k]=-1;
    }
    delete []pre_labels;
    delete []post_labels;
};

void TaggingDecoder::load_label_trans(char*filename){
    //打开文件
    FILE * pFile=fopen ( filename , "rb" );
    if(!pFile){
        fprintf(stderr,"[ERROR] DAT file %s not found\n",filename);
    }
    /*得到文件大小*/
    int remain_size=0;
    int rtn=fread (&remain_size,sizeof(int),1,pFile);
    /*得到矩阵数据*/
    label_trans=new int[remain_size];
    rtn=fread (label_trans,sizeof(int),remain_size,pFile);
    
    /*计算标签个数*/
    int label_size=0;
    for(int i=0;i<remain_size;i++){
        if(label_trans[i]==-1)label_size++;
    }
    label_size/=2;
    /*设定各个标签的指针*/
    label_trans_pre=new int*[label_size];
    label_trans_post=new int*[label_size];
    int ind=0;
    for(int i=0;i<label_size;i++){
        label_trans_pre[i]=label_trans+ind;
        while(label_trans[ind]!=-1)ind++;ind++;
        label_trans_post[i]=label_trans+ind;
        while(label_trans[ind]!=-1)ind++;ind++;
    }
    fclose (pFile);
    return;
}

void TaggingDecoder::put_values(){
    if(!len)return;
    /*nodes*/
    for(int i=0;i<len;i++){
        nodes[i].type=0;
    }
    nodes[0].type+=1;
    nodes[len-1].type+=2;
    /*values*/
    memset(values,0,sizeof(*values)*len*model->l_size);
    ngram_feature->put_values(sequence,len);
    //for(int i=0;i<len;i++) std::cout << values[i] << std::endl;
}


void TaggingDecoder::output_raw_sentence(){
    int c;
    for(int i=0;i<len;i++){
        thulac::put_character(sequence[i]);
        
    }
}
void TaggingDecoder::output_sentence(){
    int c;
    for(int i=0;i<len;i++){
        thulac::put_character(sequence[i]);
        
        if((i==len-1)||(label_info[result[i]][0]==kPOC_E)||(label_info[result[i]][0]==kPOC_S)){//分词位置
            if(*(label_info[result[i]]+1)){//输出标签（如果有的话）
                putchar(separator);
                printf("%s",label_info[result[i]]+1);
            }
            if((i+1)<len)putchar(' ');//在分词位置输出空格
        }
    }
}

int TaggingDecoder::segment(RawSentence& raw, POCGraph& graph, TaggedSentence& ts){
    if(raw.size()==0)return 0;

    for(int i=0;i<(int)raw.size();i++){
        int pocs = graph[i];
        if(pocs){
            allowed_label_lists[i]=pocs_to_tags[pocs];
        }else{
            allowed_label_lists[i]=pocs_to_tags[15];
        }
    }
    //std::cout<<"\n";
    for(int i=0;i<(int)raw.size();i++){
        sequence[i]=raw[i];
    }
    len=(int)raw.size();
    put_values();//检索出特征值并初始化放在values数组里
    dp();//动态规划搜索最优解放在result数组里

    for(int i=0;i<(int)raw.size();i++){
        allowed_label_lists[i]=NULL;
    }

    int c;
    int offset=0;
    ts.clear();
    for(int i=0;i<len;i++){
        if((i==len-1)||(label_info[result[i]][0]==kPOC_E)||(label_info[result[i]][0]==kPOC_S)){//分词位置
            ts.push_back(WordWithTag(separator));
            for(int j=offset;j<i+1;j++){
                ts.back().word.push_back(sequence[j]);
            }
            offset=i+1;
            if(*(label_info[result[i]]+1)){//输出标签（如果有的话）
                ts.back().tag=label_info[result[i]]+1;
                //printf("%s",label_info[result[i]]+1);
            }
            //if((i+1)<len)putchar(' ');//在分词位置输出空格
        }
    }
    return 1;
}
void TaggingDecoder::get_seg_result(SegmentedSentence& ss){
        ss.clear();
        /*
        Raw raw;
        for(int i = 0; i < len; i ++){
                raw.push_back(sequence[i]);
        }
        std::cerr<<raw<<std::endl;
        */
    for(int i=0;i<len;i++){
        if((i==0)||(label_info[result[i]][0]==kPOC_B)||(label_info[result[i]][0]==kPOC_S)){
            ss.push_back(Word());
        }
        ss.back().push_back(sequence[i]);
    }
};
void TaggingDecoder::cs()
    {   
        for(int i=0;i<1000;i++) std::cout << dat->dat[i].check << " " ;
        std::cout << std::endl;
    }
}



