#pragma once

#include <fstream>
#include <iostream>
#include <random>

#include <boost/algorithm/string.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/batch.h"
#include "data/dataset.h"
#include "data/vocab.h"

namespace marian {
namespace data {

class SentenceTuple {
private:
  size_t id_;
  std::vector<Words> tuple_;

public:
  SentenceTuple(size_t id) : id_(id) {}

  ~SentenceTuple() { tuple_.clear(); }

  void push_back(const Words& words) { tuple_.push_back(words); }

  size_t size() const { return tuple_.size(); }

  Words& operator[](size_t i) { return tuple_[i]; }

  Words& back() { return tuple_.back(); }
  const Words& back() const { return tuple_.back(); }

  const Words& operator[](size_t i) const { return tuple_[i]; }

  bool empty() const { return tuple_.empty(); }

  auto begin() -> decltype(tuple_.begin()) { return tuple_.begin(); }
  auto end() -> decltype(tuple_.end()) { return tuple_.end(); }

  size_t getId() const { return id_; }
};

class SubBatch {
private:
  std::vector<Word> indices_;
  std::vector<float> mask_;

  size_t size_;
  size_t width_;
  size_t words_;

public:
  SubBatch(int size, int width)
      : indices_(size * width, 0),
        mask_(size * width, 0),
        size_(size),
        width_(width),
        words_(0) {}

  std::vector<Word>& indices() { return indices_; }
  std::vector<float>& mask() { return mask_; }

  size_t batchSize() { return size_; }
  size_t batchWidth() { return width_; };
  size_t batchWords() { return words_; }

  std::vector<Ptr<SubBatch>> split(size_t n) {
    std::vector<Ptr<SubBatch>> splits;

    size_t subSize = std::ceil(size_ / (float)n);
    size_t totSize = size_;

    int pos = 0;
    for(int k = 0; k < n; ++k) {
      size_t __size__ = std::min(subSize, totSize);

      auto sb = New<SubBatch>(__size__, width_);

      size_t __words__ = 0;
      for(int j = 0; j < width_; ++j) {
        for(int i = 0; i < __size__; ++i) {
          sb->indices()[j * __size__ + i] = indices_[j * size_ + pos + i];
          sb->mask()[j * __size__ + i] = mask_[j * size_ + pos + i];
          if(mask_[j * size_ + pos + i] != 0)
            __words__++;
        }
      }

      sb->setWords(__words__);
      splits.push_back(sb);

      totSize -= __size__;
      pos += __size__;
    }
    return splits;
  }

  void setWords(size_t words) { words_ = words; }
};

class CorpusBatch : public Batch {
private:
  std::vector<Ptr<SubBatch>> batches_;
  std::vector<float> guidedAlignment_;
  std::vector<float> editDiffs_;

public:
  CorpusBatch(const std::vector<Ptr<SubBatch>>& batches) : batches_(batches) {}

  Ptr<SubBatch> operator[](size_t i) const { return batches_[i]; }

  Ptr<SubBatch> front() { return batches_.front(); }

  Ptr<SubBatch> back() { return batches_.back(); }

  void debug() {
    std::cerr << "batches: " << sets() << std::endl;

    if(!sentenceIds_.empty()) {
      std::cerr << "indexes: ";
      for(auto id : sentenceIds_)
        std::cerr << id << " ";
      std::cerr << std::endl;
    }

    size_t b = 0;
    for(auto sb : batches_) {
      std::cerr << "batch " << b++ << ": " << std::endl;
      for(size_t i = 0; i < sb->batchWidth(); i++) {
        std::cerr << "\t w: ";
        for(size_t j = 0; j < sb->batchSize(); j++) {
          Word w = sb->indices()[i * sb->batchSize() + j];
          std::cerr << w << " ";
        }
        std::cerr << std::endl;
      }
    }
  }

  std::vector<Ptr<Batch>> split(size_t n) {
    std::vector<Ptr<Batch>> splits;

    std::vector<std::vector<Ptr<SubBatch>>> subs(n);

    for(auto subBatch : batches_) {
      size_t i = 0;
      for(auto splitSubBatch : subBatch->split(n))
        subs[i++].push_back(splitSubBatch);
    }

    for(auto subBatches : subs)
      splits.push_back(New<CorpusBatch>(subBatches));

    size_t pos = 0;
    for(auto split : splits) {
      std::vector<size_t> ids;
      for(int i = pos; i < pos + split->size(); ++i)
        ids.push_back(sentenceIds_[i]);
      split->setSentenceIds(ids);
      pos += split->size();
    }

    return splits;
  }

  size_t size() const { return batches_[0]->batchSize(); }

  size_t words() const { return batches_[0]->batchWords(); }

  size_t sets() const { return batches_.size(); }

  static Ptr<CorpusBatch> fakeBatch(std::vector<size_t>& lengths,
                                    size_t batchSize,
                                    bool guidedAlignment = false,
                                    bool editAlignment = false) {
    std::vector<Ptr<SubBatch>> batches;

    for(auto len : lengths) {
      auto sb = New<SubBatch>(batchSize, len);
      std::fill(sb->mask().begin(), sb->mask().end(), 1);

      batches.push_back(sb);
    }

    auto batch = New<CorpusBatch>(batches);

    if(guidedAlignment) {
      std::vector<float> guided(batchSize * lengths.front() * lengths.back(),
                                0.f);
      batch->setGuidedAlignment(guided);
    }
    if(editAlignment) {
      std::vector<float> edits(batchSize * lengths.back(), 0.f);
      batch->setEditDiffs(edits);
    }

    return batch;
  }

  std::vector<float>& getGuidedAlignment() { return guidedAlignment_; }

  void setGuidedAlignment(const std::vector<float>& aln) {
    guidedAlignment_ = aln;
  }

  std::vector<float>& getEditDiffs() { return editDiffs_; }

  void setEditDiffs(const std::vector<float>& editDiffs) {
    editDiffs_ = editDiffs;
  }
};

class Corpus;

class CorpusIterator
    : public boost::iterator_facade<CorpusIterator,
                                    SentenceTuple const,
                                    boost::forward_traversal_tag> {
public:
  CorpusIterator();
  explicit CorpusIterator(Corpus& corpus);

private:
  friend class boost::iterator_core_access;

  void increment();

  bool equal(CorpusIterator const& other) const;

  const SentenceTuple& dereference() const;

  Corpus* corpus_;

  long long int pos_;
  SentenceTuple tup_;
};

class WordAlignment {
private:
  typedef std::pair<int, int> Point;
  typedef std::vector<Point> Alignment;

  std::vector<Alignment> data_;

public:
  WordAlignment(const std::string& fname) {
    InputFileStream aStream(fname);
    std::string line;
    size_t c = 0;

    LOG(info, "[data] Loading word alignment from {}", fname);

    while(std::getline((std::istream&)aStream, line)) {
      data_.emplace_back();
      if(!line.empty()) {
        std::vector<std::string> atok = split(line, " -");
        for(size_t i = 0; i < atok.size(); i += 2)
          data_.back().emplace_back(std::stoi(atok[i]), std::stoi(atok[i + 1]));
      }
      c++;
    }

    LOG(info, "[data] Done");
  }

  std::vector<std::string> split(const std::string& input,
                                 const std::string& chars) {
    std::vector<std::string> output;
    boost::split(output, input, boost::is_any_of(chars));
    return output;
  }

  void guidedAlignment(Ptr<CorpusBatch> batch) {
    int srcWords = batch->front()->batchWidth();
    int trgWords = batch->back()->batchWidth();

    int dimBatch = batch->getSentenceIds().size();
    std::vector<float> guided(dimBatch * srcWords * trgWords, 0.f);

    for(int b = 0; b < dimBatch; ++b) {
      auto& alignment = data_[batch->getSentenceIds()[b]];
      for(auto& p : alignment) {
        int sid, tid;
        std::tie(sid, tid) = p;

        size_t idx = b + sid * dimBatch + tid * srcWords * dimBatch;
        guided[idx] = 1.f;
      }
    }
    batch->setGuidedAlignment(guided);
  }
};

class Corpus : public DatasetBase<SentenceTuple, CorpusIterator, CorpusBatch> {
private:
  Ptr<Config> options_;

  std::vector<UPtr<TemporaryFile>> tempFiles_;
  std::vector<UPtr<InputFileStream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;
  size_t maxLength_;
  bool maxLengthCrop_;

  std::mt19937 g_;
  std::vector<size_t> ids_;
  size_t pos_{0};

  Ptr<WordAlignment> wordAlignment_;

  void shuffleFiles(const std::vector<std::string>& paths);

public:
  Corpus(Ptr<Config> options, bool translate = false);

  Corpus(std::vector<std::string> paths,
         std::vector<Ptr<Vocab>> vocabs,
         Ptr<Config> options,
         size_t maxLength = 0);

  /**
   * @brief Iterates sentence tuples in the corpus.
   *
   * A sentence tuple is skipped with no warning if any sentence in the tuple
   * (e.g. a source or target) is longer than the maximum allowed sentence
   * length in words.
   *
   * @return A tuple representing parallel sentences.
   */
  sample next();

  void shuffle();

  void reset();

  iterator begin() { return iterator(*this); }

  iterator end() { return iterator(); }

  std::vector<Ptr<Vocab>>& getVocabs() { return vocabs_; }

  void editDiffs(Ptr<CorpusBatch> batch, float weight = 1.0) {
    int dimBatch = batch->getSentenceIds().size();
    int trgWords = batch->back()->batchWidth();
    std::vector<float> edits(dimBatch * trgWords, 1.0);

    for(int i = 0; i < batch->back()->indices().size(); ++i)
      if(batch->front()->indices()[i] != batch->back()->indices()[i])
        edits[i] = weight;

    batch->setEditDiffs(edits);
  }


  //template <class Distribution>
  //void distribution(std::vector<float>& vals, float a, float b) {
  //  std::default_random_engine engine(Config::seed++);
  //  Distribution dist(a, b);
  //  auto gen = std::bind(dist, engine);
  //  std::generate(begin(vals), end(vals), gen);
  //}

  //distribution<std::normal_distribution<float>>(t, 0, scale);
  //distribution<std::uniform_real_distribution<float>>(t, -scale, scale);

  template <class Gen>
  void sub(std::vector<Word>& words, int sid, int dimBatch, int dimWords, Gen gen) {
    int wid = gen() * (dimWords - 1);
    int i = wid * dimBatch + sid;

    while(words[i] == 0) {
      wid = gen() * (dimWords - 1);
      i = wid * dimBatch + sid;
    }

    int j = gen() * (words.size() - 1);
    while(words[j] == 0 || i == j)
      j = gen() * (words.size() - 1);

    words[i] = words[j];
  }

  template <class Gen>
  void del(std::vector<Word>& words, int sid, int dimBatch, int dimWords, Gen gen) {
    int wid = gen() * (dimWords - 1);
    int i = wid * dimBatch + sid;

    while(words[i] == 0) {
      wid = gen() * (dimWords - 1);
      i = wid * dimBatch + sid;
    }

    for(int w = wid + 1; w < dimWords; ++w) {
      int j = w * dimBatch + sid;
      words[i] = words[j];
      i = j;
    }
  }

  template <class Gen>
  void ins(std::vector<Word>& words, int sid, int dimBatch, int dimWords, Gen gen) {
    int wid = gen() * (dimWords - 1);
    int i = wid * dimBatch + sid;

    while(words[i] == 0) {
      wid = gen() * (dimWords - 1);
      i = wid * dimBatch + sid;
    }

    for(int w = dimWords - 1; w > wid ; --w) {
      int j = w * dimBatch + sid;
      int k = (w - 1) * dimBatch + sid;
      words[j] = words[k];
    }

    int j = gen() * (words.size() - 1);
    while(words[j] == 0 || i == j)
      j = gen() * (words.size() - 1);

    words[i] = words[j];
  }

  template <class Gen>
  void swp(std::vector<Word>& words, int sid, int dimBatch, int dimWords, Gen gen) {
    int wid = gen() * (dimWords - 1);
    int i = wid * dimBatch + sid;

    while(words[i] == 0) {
      wid = gen() * (dimWords - 1);
      i = wid * dimBatch + sid;
    }

    int j = (wid + 1) * dimBatch + sid;

    if(j < words.size() && words[j] != 0) {
      Word temp = words[i];
      words[i] = words[j];
      words[j] = temp;
    }
  }

  template <class Gen>
  void corruptSent(std::vector<Word>& words, int sid,
                   int dimBatch, int dimWords,
                   int actions, Gen gen) {
    float subProb = 0.5;
    float delProb = 0.166;
    float insProb = 0.166;
    float swpProb = 0.166;

    int total = 100;
    std::vector<int> wheel;
    for(int i = 0; i < total * subProb; i++)
      wheel.push_back(0);
    for(int i = 0; i < total * delProb; i++)
      wheel.push_back(1);
    for(int i = 0; i < total * insProb; i++)
      wheel.push_back(2);
    for(int i = 0; i < total * swpProb; i++)
      wheel.push_back(3);

    for(int i = 0; i < actions; i++) {
      float p = gen();
      int act = wheel[p * (wheel.size() - 1)];
      switch (act) {
        case 0: sub(words, sid, dimBatch, dimWords, gen); break;
        case 1: del(words, sid, dimBatch, dimWords, gen); break;
        case 2: ins(words, sid, dimBatch, dimWords, gen); break;
        case 3: swp(words, sid, dimBatch, dimWords, gen); break;
        default: break;
      }
    }
  }

  void corrupt(Ptr<CorpusBatch> batch) {
    float senProb = 0.6;
    float errProb = 0.2;

    int dimBatch = batch->getSentenceIds().size();
    int srcWords = batch->front()->batchWidth();

    std::vector<Word>& words = batch->front()->indices();

    std::default_random_engine engine(Config::seed++);

    std::normal_distribution<float> distNormal(errProb / senProb, 0.2);
    std::uniform_real_distribution<float> distUniform(0.0, 1.0);

    auto genNormal = std::bind(distNormal, engine);
    auto genUniform = std::bind(distUniform, engine);

    std::vector<float> senProbs(dimBatch);
    std::vector<float> errProbs(dimBatch);

    std::generate(std::begin(senProbs), std::end(senProbs), genUniform);
    std::generate(std::begin(errProbs), std::end(errProbs), genNormal);

    for(int sid = 0; sid < dimBatch; sid++) {
      if(senProbs[sid] < 0.6) {
        corruptSent(words, sid,
                    dimBatch, srcWords,
                    srcWords * fabs(errProbs[sid]),
                    genUniform);
      }
    }
  }

  batch_ptr toBatch(const std::vector<sample>& batchVector) {
    int batchSize = batchVector.size();

    std::vector<size_t> sentenceIds;

    std::vector<int> maxDims;
    for(auto& ex : batchVector) {
      if(maxDims.size() < ex.size())
        maxDims.resize(ex.size(), 0);
      for(size_t i = 0; i < ex.size(); ++i) {
        if(ex[i].size() > (size_t)maxDims[i])
          maxDims[i] = ex[i].size();
      }
      sentenceIds.push_back(ex.getId());
    }

    std::vector<Ptr<SubBatch>> subBatches;
    for(auto m : maxDims) {
      subBatches.emplace_back(New<SubBatch>(batchSize, m));
    }

    std::vector<size_t> words(maxDims.size(), 0);
    for(int i = 0; i < batchSize; ++i) {
      for(int j = 0; j < maxDims.size(); ++j) {
        for(int k = 0; k < batchVector[i][j].size(); ++k) {
          subBatches[j]->indices()[k * batchSize + i] = batchVector[i][j][k];
          subBatches[j]->mask()[k * batchSize + i] = 1.f;
          words[j]++;
        }
      }
    }

    for(size_t j = 0; j < maxDims.size(); ++j)
      subBatches[j]->setWords(words[j]);

    auto batch = batch_ptr(new batch_type(subBatches));
    batch->setSentenceIds(sentenceIds);

    if(options_->get<bool>("edit-corrupt"))
      corrupt(batch);

    if(options_->has("guided-alignment") && wordAlignment_)
      wordAlignment_->guidedAlignment(batch);

    if(options_->has("edit-weight"))
      editDiffs(batch, options_->get<float>("edit-weight"));

    return batch;
  }

  void prepare() {
    if(options_->has("guided-alignment"))
      setWordAlignment(options_->get<std::string>("guided-alignment"));
  }

private:
  void setWordAlignment(const std::string& path) {
    wordAlignment_ = New<WordAlignment>(path);
  }
};
}
}
