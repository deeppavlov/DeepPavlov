#ifndef __DP_H__
#define __DP_H__
#include<cstdlib>

namespace permm{


/**
 * topological information about a node
 * type的定义： 默认0，如果是开始节点+1，如果是结尾节点+2
 * */
struct Node{
    int type;///默认0，如果是开始节点+1，如果是结尾节点+2
    int* predecessors;///ends with a -1，用于记录可能的前继结点
    int* successors;///ends with a -1，用于记录可能的后继结点
};

///**given prececessors, calculate successors*/
//int* dp_cal_successors(int node_count,Node* nodes);

//a structure for alphas and betas，用于保存Alpha(i,label)的分值或者Beta(i，label)的分值
struct Alpha_Beta{
    int value;
    int node_id;
    int label_id;
};

/** The DP algorithm(s) for path labeling */
inline int dp_decode(
        int l_size,///标签个数
        int* ll_weights,///标签间权重
        int node_count,///节点个数
        Node* nodes,///节点数据
        int* values,///value for i-th node with j-th label
        Alpha_Beta* alphas,///alpha value (and the pointer) for i-th node with j-th label
        int* result,
        int** pre_labels=NULL,///每种标签可能的前导标签（以-1结尾）
        int** allowed_label_lists=NULL///每个节点可能的标签列表
        ){
    //calculate alphas
    int node_id;
    int* p_node_id;
    int* p_pre_label;
    int* p_allowed_label;//指向当前字所有可能标签的数组的指针
    register int k;//当前字的前一个节点可能的标签（的编号）
    register int j;//当前字某一个可能的标签（的编号）
    register Alpha_Beta* tmp;
    Alpha_Beta best;best.node_id=-1;
    Alpha_Beta* pre_alpha;
    int score;
    
    for(int i=0;i<node_count*l_size;i++)alphas[i].node_id=-2;
    for(int i=0;i<node_count;i++){//for each node
        p_allowed_label=allowed_label_lists?allowed_label_lists[i]:NULL;
        j=-1;
        int max_value=0;
        int has_max_value=0;
	//为每一个节点分配当前具有最大值的标签
        while((p_allowed_label?
                    ((j=(*(p_allowed_label++)))!=-1)://如果有指定，则按照列表来
                    ((++j)!=l_size))){//否则枚举
            if((!has_max_value) || (max_value<values[i*l_size+j])){
                has_max_value=1;
                max_value=values[i*l_size+j];
            }
        }
        p_allowed_label=allowed_label_lists?allowed_label_lists[i]:NULL;
        j=-1;
        while((p_allowed_label?
                    ((j=(*(p_allowed_label++)))!=-1)://如果有指定，则按照列表来
                    ((++j)!=l_size))){//否则枚举
            //if(max_value-20000>values[i*l_size+j])continue;//
            tmp=&alphas[i*l_size+j];
            tmp->value=0;
            p_node_id=nodes[i].predecessors;//前继结点
            p_pre_label=pre_labels?pre_labels[j]:NULL;
            while((node_id=*(p_node_id++))>=0){//枚举前继节点
                k=-1;
                while(p_pre_label?
                        ((k=(*p_pre_label++))!=-1):
                        ((++k)!=l_size)
                        ){
                    pre_alpha=alphas+node_id*l_size+k;//获得alpha(node_id,k)的得分
                    if(pre_alpha->node_id==-2)continue;//not reachable
                    score=pre_alpha->value+ll_weights[k*l_size+j];//score=alpha(node_id,k)+ll_weights(k,j)
                    if((tmp->node_id<0)||(score>tmp->value)){
                        tmp->value=score;
                        tmp->node_id=node_id;
                        tmp->label_id=k;
                    }
                }
            }
            tmp->value+=values[i*l_size+j];//加上本身的分数
            
            if((nodes[i].type==1)||(nodes[i].type==3))//如果第i个节点原来的type为1或3
                tmp->node_id=-1;
            if(nodes[i].type>=2){//更新最大得分
                if((best.node_id==-1)||(best.value<tmp->value)){
                    best.value=tmp->value;
                    best.node_id=i;
                    best.label_id=j;
                }
            }
        }
    }
    //find the path and label the nodes of it.
    tmp=&best;
    while(tmp->node_id>=0){
        result[tmp->node_id]=tmp->label_id;
        tmp=&(alphas[(tmp->node_id)*l_size+(tmp->label_id)]);
    }
    //debug
    /*(for(int i=0;i<node_count;i++){//for each node
        p_allowed_label=allowed_label_lists?allowed_label_lists[i]:NULL;   
        j=-1;
        std::cerr<<values[i*l_size+result[i]]<<" ";
        while((p_allowed_label?
                    ((j=(*(p_allowed_label++)))!=-1)://如果有指定，则按照列表来
                    ((++j)!=l_size))){//否则枚举
            tmp=&alphas[i*l_size+j];
            std::cerr<<values[i*l_size+j]<<" ";  
        }
        std::cerr<<"\n";
    }
    std::cerr<<"\n";*/
    //end of debug
    return best.value;
};


}
#endif
