#include <iostream>
#include <string>
#include "thulac.h"
#include "thulac_test.h"


int main() {
    THULAC lac;
    const char *model_path = "/program/python/python-git/thulac/models";
    std::string case_name;
    {
        
        case_name = "检查单分词模型模型";
        lac.init(model_path,NULL,1);
        THULAC_TEST test = THULAC_TEST(&lac, case_name);
        test.testEqual("我爱北京天安门", "我 爱 北京 天安门");
        test.testEqual("小明喜欢玩炉石传说", "小明 喜欢 玩 炉石 传说");
    }
    
    {
        case_name = "检查带词性标注的模型";
        lac.init(model_path,NULL,0);
        THULAC_TEST test = THULAC_TEST(&lac, case_name);
        test.testEqual("我爱北京天安门", "我_r 爱_v 北京_ns 天安门_ns");
        test.testEqual("小明喜欢玩炉石传说", "小明_np 喜欢_v 玩_v 炉石_n 传说_n");
    }
    
    {
        case_name = "检查deli分隔符参数";
        lac.init(model_path,NULL,0, 0, 0,'#');
        THULAC_TEST test = THULAC_TEST(&lac, case_name);
        test.testEqual("我爱北京天安门", "我#r 爱#v 北京#ns 天安门#ns");
        test.testEqual("小明喜欢玩炉石传说", "小明#np 喜欢#v 玩#v 炉石#n 传说#n");
    }
    
    {
        case_name = "检查T2S分隔符参数";
        lac.init(model_path,NULL,1, 1);
        THULAC_TEST test = THULAC_TEST(&lac, case_name);
        test.testEqual("我愛北京天安門", "我 爱 北京 天安门");
        test.testEqual("小明喜歡玩爐石傳說", "小明 喜欢 玩 炉石 传说");
    }
    
    {
        case_name = "检查ufilter参数";
        lac.init(model_path,NULL,1, 0, 1);
        THULAC_TEST test = THULAC_TEST(&lac, case_name);
        test.testEqual("我可以爱北京天安门", "我 爱 北京 天安门");
    }
    
//    std::cout << "下面测试程序报错的错误提示" << endl;
//    
//    {
//        case_name = "model_dir不存在";
//        try {
//            lac.init("/program/python/python-git/thulac/model",NULL,1, 0, 1);
//        }
//        catch e{
//            std::cout << e;
//        }
//    }
    THULAC_TEST::reportAll();
    return 0;
}



