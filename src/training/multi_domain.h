#pragma once

#include "common/config.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"


namespace marian {

using namespace data;

class TrainMultiDomain : public ModelTask {
private:
  Ptr<Config> options_;
  std::vector<Ptr<Vocab>> vocabs_;

  Ptr<models::ModelBase> builder_;
  Ptr<models::ModelBase> builderTrans_;
  Ptr<ExpressionGraph> graph_;
  Ptr<OptimizerBase> opt_;


  void train(Ptr<data::Batch> batch) {
    auto costNode = builder_->build(graph_, batch);

    graph_->forward();
    float cost = costNode->scalar();
    graph_->backward();

    opt_->update(graph_);
  }

  void translate(Ptr<data::Batch> batch = nullptr) {
    // Temporary options for translation
    auto cfgs = New<Config>(*options_);
    cfgs->set("mini-batch", 1);
    cfgs->set("maxi-batch", 1);
    cfgs->set("max-length", 1000);

    // Create corpus
    auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
    std::vector<std::string> srcPaths(validPaths.begin(), validPaths.end() - 1);
    std::vector<Ptr<Vocab>> srcVocabs(vocabs_.begin(), vocabs_.end() - 1);
    auto corpus = New<data::Corpus>(srcPaths, srcVocabs, cfgs);

    // Generate batches
    auto batchGenerator = New<BatchGenerator<data::Corpus>>(corpus, cfgs);
    batchGenerator->prepare(false);

    // Create scorer
    auto model = options_->get<std::string>("model");
    Ptr<Scorer> scorer = New<ScorerWrapper>(builderTrans_, "", 1.0f, model);
    std::vector<Ptr<Scorer>> scorers = { scorer };

    LOG(valid)->info("Translating...");

    graph_->setInference(true);
    boost::timer::cpu_timer timer;

    {
      auto collector = New<OutputCollector>();
      size_t sentenceId = 0;

      while(*batchGenerator) {
        auto batch = batchGenerator->next();

        graph_->clear();
        auto search = New<BeamSearch>(options_, scorers);
        auto history = search->search(graph_, batch, sentenceId);

        std::stringstream best1;
        std::stringstream bestn;
        Printer(options_, vocabs_.back(), history, best1, bestn);
        collector->Write(history->GetLineNum(),
                         best1.str(),
                         bestn.str(),
                         options_->get<bool>("n-best"));

        //int id = batch->getSentenceIds()[0];
        //LOG(valid)->info("Best translation {}: {}", id, best1.str());

        sentenceId++;
      }
    }

    LOG(valid)->info("Total translation time: {}", timer.format(5, "%ws"));
    graph_->setInference(false);
  }

public:
  TrainMultiDomain(Ptr<Config> options) : options_(options) {
    size_t device = options_->get<std::vector<size_t>>("devices")[0];

    graph_ = New<ExpressionGraph>();
    graph_->setDevice(device);
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    opt_ = Optimizer(options_);

    builder_ = models::from_config(options_);

    Ptr<Options> opts = New<Options>();
    opts->merge(options_);
    opts->set("inference", true);
    builderTrans_ = models::from_options(opts);
  }

  void run() {
    auto state = New<TrainingState>(options_->get<float>("learn-rate"));
    auto scheduler = New<Scheduler>(options_, state);
    scheduler->registerTrainingObserver(scheduler);
    scheduler->registerTrainingObserver(opt_);

    auto dataset = New<data::Corpus>(options_);
    dataset->prepare();
    vocabs_ = dataset->getVocabs();

    std::string name = options_->get<std::string>("model");
    builder_->load(graph_, name);

    auto batchGenerator = New<BatchGenerator<Corpus>>(dataset, options_);

    scheduler->started();
    while(scheduler->keepGoing()) {
      batchGenerator->prepare(false);
      while(*batchGenerator && scheduler->keepGoing()) {
        auto batch = batchGenerator->next();
        train(batch);
      }
      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();

    translate();
  }
};
}
