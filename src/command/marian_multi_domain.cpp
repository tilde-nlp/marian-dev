#include "marian.h"

#include "training/training.h"
#include "training/multi_domain.h"
#include "common/file_stream.h"
#include "common/utils.h"


int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  auto task = New<TrainMultiDomain>(options);
  task->init();

  std::string line;
  while(std::getline(std::cin, line)) {
    std::vector<std::string> inputs;
    Split(line, inputs, "\t");

    std::string text(inputs.back());
    std::vector<std::string> trainSet(inputs.begin(), inputs.end() - 1);

    task->run(text, trainSet);
  }

  return 0;
}
