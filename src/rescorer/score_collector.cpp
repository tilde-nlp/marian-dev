#include <cassert>

#include "common/file_stream.h"
#include "common/logging.h"
#include "rescorer/score_collector.h"

namespace marian {

ScoreCollector::ScoreCollector()
    : nextId_(0), outStrm_(new OutputFileStream(std::cout)) {}

void ScoreCollector::Write(long id, float score) {
  boost::mutex::scoped_lock lock(mutex_);
  if(id == nextId_) {
    ((std::ostream&)*outStrm_) << score << std::endl;

    ++nextId_;

    Outputs::const_iterator iter, iterNext;
    iter = outputs_.begin();
    while(iter != outputs_.end()) {
      long currId = iter->first;

      if(currId == nextId_) {
        // 1st element in the map is the next
        ((std::ostream&)*outStrm_) << iter->second.first << std::endl;

        ++nextId_;

        // delete current record, move iter on 1
        iterNext = iter;
        ++iterNext;
        outputs_.erase(iter);
        iter = iterNext;
      } else {
        // not the nextr, stop iterating
        assert(nextId_ < currId);
        break;
      }
    }

  } else {
    // save for later
    outputs_[id] = std::make_pair(score, "");
  }
}

void ScoreCollector::WriteNBest(long id,
                                float score,
                                const std::string& sentence,
                                const std::string& feature) {
  boost::mutex::scoped_lock lock(mutex_);
  if(id == nextId_) {
    ((std::ostream&)*outStrm_) << id << " ||| " << sentence << " ||| "
                               << feature << "= " << score << " ||| " << score
                               << std::endl;

    ++nextId_;

    Outputs::const_iterator iter, iterNext;
    iter = outputs_.begin();
    while(iter != outputs_.end()) {
      long currId = iter->first;

      if(currId == nextId_) {
        // 1st element in the map is the next
        ((std::ostream&)*outStrm_) << iter->first << iter->second.second << std::endl;

        ++nextId_;

        // delete current record, move iter on 1
        iterNext = iter;
        ++iterNext;
        outputs_.erase(iter);
        iter = iterNext;
      } else {
        // not the nextr, stop iterating
        assert(nextId_ < currId);
        break;
      }
    }

  } else {
    // @TODO: outputs_ of that type is ugly!
    // save for later
    outputs_[id] = std::make_pair(score,
                                  " ||| " + sentence + " ||| " + feature + "= "
                                      + std::to_string(score)
                                      + " ||| "
                                      + std::to_string(score));
  }
}
}
