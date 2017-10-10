#include "marian.h"

#include "training/training.h"
#include "training/multi_domain.h"
#include "training/graph_group_async.h"
#include "training/graph_group_sync.h"
#include "training/graph_group_singleton.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  New<TrainMultiDomain>(options)->run();

  return 0;
}
