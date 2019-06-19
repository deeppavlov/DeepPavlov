#include <string>
#include <vector>
#include <iostream>
#include <utility>
#include "../include/thulac.h"
class THULAC_TEST {
public:
    void report(const std::string &s);
    void testEqual(const std::string &raw, const std::string &standard);
    THULAC_TEST(THULAC *lac);
    THULAC_TEST(THULAC *lac, std::string name);
    static void reportAll();
private:
    std::vector<std::string> errorMsg;
    THULAC *lac;
    std::string caseName;
    bool equals(const std::string &result, const std::string &standard);
    void report_error(const std::string &e);
    static std::vector<std::pair<std::string, std::string> > allErrorMsg;
    
};

THULAC_TEST::THULAC_TEST(THULAC *lac) {
    this->lac = lac;
    
    
}

THULAC_TEST::THULAC_TEST(THULAC *lac, std::string caseName) {
    this->lac = lac;
    this->caseName = caseName;
}

bool THULAC_TEST::equals(const std::string &result, const std::string &standard) {
    return result == standard;
}

void THULAC_TEST::report_error(const std::string &e) {
    errorMsg.push_back(e);
    allErrorMsg.push_back(std::make_pair(caseName, e));
}

void THULAC_TEST::testEqual(const std::string &raw, const std::string &standard) {
    THULAC_result result;
    lac->cut(raw, result);
    std::string s_result = lac->toString(result);
    if(equals(s_result, standard)) {
        std::cout << ".";
        return;
    }
    std::cout << "E";
    std::string error = "不匹配：" + s_result + "  与  " + standard;
    report_error(error);
    return;
}

void THULAC_TEST::report(const std::string &s) {
    std::cout << std::endl << s << ":" << std::endl;
    if(errorMsg.size() == 0) {
        std::cout << "恭喜！所有case通过测试" << std::endl;
    }
    else{
        for(auto i : errorMsg) {
            
            std::cout << i << std::endl;
        }
        
    }
    return;
}

void THULAC_TEST::reportAll() {
    std::cout << "\n测试结果：\n";
    if(allErrorMsg.size() == 0) std::cout << "恭喜！所有case通过测试" << std::endl;
    else {
        for (auto i : allErrorMsg) {
            std::cout << i.first << "case中：\n" << i.second << std::endl;
        }
    }
}

std::vector<std::pair<std::string, std::string> > THULAC_TEST::allErrorMsg = std::vector<std::pair<std::string, std::string> >();
