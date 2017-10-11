#pragma once

#include "common/config.h"
#include "data/batch_generator.h"
#include "data/text_input.h"
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"


namespace marian {

using namespace data;

class TrainMultiDomain : public ModelTask {
private:
  Ptr<Config> options_;

  std::vector<Ptr<Vocab>> vocabs_;
  std::vector<Ptr<Scorer>> scorers_;

  Ptr<models::ModelBase> builder_;
  Ptr<models::ModelBase> builderTrans_;
  Ptr<ExpressionGraph> graph_;
  Ptr<ExpressionGraph> graphTemp_;

  Ptr<OptimizerBase> optimizer_;

  float train(Ptr<data::Batch> batch) {
    auto costNode = builder_->build(graphTemp_, batch);
    graphTemp_->forward();
    float cost = costNode->scalar();
    graphTemp_->backward();

    optimizer_->update(graphTemp_);

    return cost;
  }

  void translate(Ptr<data::CorpusBatch> batch) {
    graphTemp_->setInference(true);
    {
      auto collector = New<OutputCollector>();
      size_t sentenceId = 0;

      graphTemp_->clear();
      auto search = New<BeamSearch>(options_, scorers_);
      auto history = search->search(graphTemp_, batch, sentenceId);

      std::stringstream best1;
      std::stringstream bestn;
      Printer(options_, vocabs_.back(), history, best1, bestn);

      collector->Write(history->GetLineNum(),
                       best1.str(),
                       bestn.str(),
                       options_->get<bool>("n-best"));

      ++sentenceId;
    }
    graphTemp_->setInference(false);
  }

public:
  TrainMultiDomain(Ptr<Config> options) : options_(options) {
    size_t device = options_->get<std::vector<size_t>>("devices")[0];

    graph_ = New<ExpressionGraph>();
    graph_->setDevice(device);
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    builder_ = models::from_config(options_);

    optimizer_ = Optimizer(options_);

    Ptr<Options> opts = New<Options>();
    opts->merge(options_);
    opts->set("inference", true);
    builderTrans_ = models::from_options(opts);

    auto model = options_->get<std::string>("model");
    scorers_.push_back(New<ScorerWrapper>(builderTrans_, "", 1.0f, model));
  }

  void init() {
    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");

    for(size_t i = 0; i < vocabPaths.size(); ++i) {
      Ptr<Vocab> vocab = New<Vocab>();
      vocab->load(vocabPaths[i], maxVocabs[i]);
      vocabs_.emplace_back(vocab);
    }

    std::string name = options_->get<std::string>("model");
    builder_->load(graph_, name);
  }

  void run(std::string text, std::vector<std::string> trainSet) {
    // Training
    auto state = New<TrainingState>(options_->get<float>("learn-rate"));
    auto scheduler = New<Scheduler>(options_, state);
    scheduler->registerTrainingObserver(scheduler);
    scheduler->registerTrainingObserver(optimizer_);

    auto opts = New<Config>(*options_);
    opts->set<size_t>("max-length", 1000);

    auto trainset = New<data::TextInput>(trainSet, vocabs_, opts);
    auto trainBG = New<BatchGenerator<data::TextInput>>(trainset, opts);

    bool first = true;

    scheduler->started();
    while(scheduler->keepGoing()) {
      trainBG->prepare(false);

      while(*trainBG && scheduler->keepGoing()) {
        auto batch = trainBG->next();

        if(first) {
          builder_->build(graph_, batch);
          graph_->forward();

          graphTemp_ = New<ExpressionGraph>();
          graphTemp_->setDevice(graph_->getDevice());
          graphTemp_->reuseWorkspace(graph_);

          graphTemp_->copyParams(graph_);
          first = false;
        }

        auto cost = train(batch);
        scheduler->update(cost, batch);
      }
      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();

    // Translation
    opts->set<size_t>("mini-batch", 1);
    opts->set<size_t>("maxi-batch", 1);

    auto testset = New<data::TextInput>(text, vocabs_.front(), opts);
    auto testBG = New<BatchGenerator<TextInput>>(testset, opts);
    testBG->prepare(false);
    auto batch = testBG->next();

    translate(batch);
  }

  void run() {}

};
}
