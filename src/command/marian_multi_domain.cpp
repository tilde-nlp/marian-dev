#include "marian.h"

#include "training/training.h"
#include "training/multi_domain.h"
#include "common/file_stream.h"
#include "common/utils.h"


int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  New<TrainMultiDomain>(options)->run();

  return 0;
}
