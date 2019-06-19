cmake_minimum_required(VERSION 2.8)
project(THULAC)

set(SOURCE_FILES
    src/cb_decoder.h
    src/cb_model.h
    src/cb_ngram_feature.h
    src/cb_tagging_decoder.h
    src/cb_tagging_learner.h
    src/dat.h
    src/filter.h
    src/negword.h
    src/postprocess.h
    src/preprocess.h
    src/punctuation.h
    src/thulac_base.h
    src/thulac_character.h
    src/thulac_raw.h
    src/timeword.h
    src/verbword.h)

add_executable(thulac src/thulac.cc ${SOURCE_FILES})
add_executable(train src/train_c.cc ${SOURCE_FILES})