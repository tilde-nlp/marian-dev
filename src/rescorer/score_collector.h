#pragma once

#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <iostream>
#include <map>

#include "common/definitions.h"
#include "common/file_stream.h"

namespace marian {

class ScoreCollector {
public:
  ScoreCollector();
  ScoreCollector(const ScoreCollector&) = delete;

  void Write(long id, float score);
  void WriteNBest(long id,
                  float score,
                  const std::string& sentence,
                  const std::string& feature);

protected:
  long nextId_{0};
  UPtr<OutputFileStream> outStrm_;
  boost::mutex mutex_;

  // @TODO: outputs_ of that type is ugly!
  typedef std::map<long, std::pair<float, std::string>> Outputs;
  Outputs outputs_;
};
}
