#pragma once
#include <cstdio>
namespace permm{

/*the structure of the model, also the file of the model*/
template<typename ValueType=int>
class BasicModel{
public:
    static const int DEC=1000;
    int l_size;//size of the labels
    int f_size;//size of the features
    int* ll_weights;
    int* fl_weights;//weights of (feature,label)
    
    double* ave_ll_weights;
    double* ave_fl_weights;//weights of (feature,label)
    
    
    BasicModel(int l,int f):ave_ll_weights(NULL),ave_fl_weights(NULL){
        this->l_size=l;
        this->f_size=f;
        this->ll_weights=(int*)calloc(sizeof(int),l*l);
        this->fl_weights=(int*)calloc(sizeof(int),l*f);
        this->ave_ll_weights=(double*)calloc(sizeof(double),l*l);
        this->ave_fl_weights=(double*)calloc(sizeof(double),l*f);
    }
    ~BasicModel(){
		free(this->ll_weights);
        free(this->fl_weights);
        free(this->ave_ll_weights);
        free(this->ave_fl_weights);
        
    }
    void reset_ave_weights(){
        free(this->ave_ll_weights);
        free(this->ave_fl_weights);
        this->ave_ll_weights=(double*)calloc(sizeof(double),l_size*l_size);
        this->ave_fl_weights=(double*)calloc(sizeof(double),l_size*f_size);
    }
    
    void update_ll_weight(const int& i,const int& j,const int& delta,const long& steps){
        int ind=i*l_size+j;
        //std::cout<<i<<" "<<j<<" "<<ind<<"\n";
        this->ll_weights[ind]+=delta;
        this->ave_ll_weights[ind]+=steps*delta;
    }
    void update_fl_weight(const int& i,const int& j,const int& delta,const long& steps){
        int ind=i*l_size+j;
        this->fl_weights[ind]+=delta;
        this->ave_fl_weights[ind]+=steps*delta;
    }
    
    void average(int step){
        int l_size=this->l_size;
        int f_size=this->f_size;
        for(int i=0;i<l_size*f_size;i++){
            this->fl_weights[i]=(int)
                    (
                     (
                        (double)(this->fl_weights[i])
                         -
                        (this->ave_fl_weights[i])/step
                      )
                     *DEC+0.5
                     )
            ;
        }
        for(int i=0;i<l_size*l_size;i++){
            this->ll_weights[i]=(int)
                    (
                        (
                         (double)(this->ll_weights[i])
                         -
                        (this->ave_ll_weights[i])/step   
                        )
                    *DEC+0.5
                    )
            ;
        }
    }
    
    BasicModel(const char* filename):ave_ll_weights(NULL),ave_fl_weights(NULL){
        FILE* pFile;
        size_t rtn_value;
        pFile=fopen(filename,"rb");
        if(!pFile){
            fprintf(stderr,"[ERROR] models path is incorrect, please check the \"models_dir\" parameter or make sure \"models\" is included in your root directory.\n",filename);
        }
        rtn_value=fread(&(this->l_size),4,1,pFile);
        rtn_value=fread(&(this->f_size),4,1,pFile);
        int l_size=this->l_size;
        int f_size=this->f_size;

        this->ll_weights=(int*)malloc(sizeof(int)*l_size*l_size);
        this->fl_weights=(int*)malloc(sizeof(int)*l_size*f_size);
            
        rtn_value=fread((this->ll_weights),4,l_size*l_size,pFile);
        rtn_value=fread((this->fl_weights),4,l_size*f_size,pFile);
        fclose(pFile);

    }
    
    void save(const char* filename){
        FILE* pFile=fopen(filename,"wb");
        int l_size=this->l_size;
        int f_size=this->f_size;
        
        fwrite(&(this->l_size),4,1,pFile);
        fwrite(&(this->f_size),4,1,pFile);
        //printf("%d %d\n",l_size,f_size);
        fwrite((this->ll_weights),4,l_size*l_size,pFile);
        fwrite((this->fl_weights),4,l_size*f_size,pFile);
        fclose(pFile);
    }
    
    
};


typedef BasicModel<> Model;

}
